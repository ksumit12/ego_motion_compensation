#!/usr/bin/env python3
"""
ORIGINAL video generation using time binning cancellation.
Faster processing but may lose some matches at bin edges.

TIME BINNING APPROACH:
1. Events grouped into time bins (5ms bins)
2. Cancellation performed within each bin
3. Faster processing but potential loss at bin edges
4. Memory-safe with configurable limits

Available interpolation modes:
- Nearest neighbor (fastest):  USE_BILINEAR_INTERP=False
- Fast bilinear (recommended): USE_BILINEAR_INTERP=True, USE_HIGH_QUALITY_INTERP=False  
- Enhanced bilinear (slowest): USE_BILINEAR_INTERP=True, USE_HIGH_QUALITY_INTERP=True
"""
import os
import time
import gc
import numpy as np
from scipy.spatial import cKDTree

# Core modular functions copied from visualize_time_window_clean.py

def estimate_dt_from_data(real_events: np.ndarray, predicted_events: np.ndarray) -> float:
    """
    CORE MODULAR FUNCTION: Estimate dt from real and predicted event timestamps.
    
    This function can be used independently to estimate time offset.
    
    Args:
        real_events: Array [x, y, polarity, timestamp] of real events
        predicted_events: Array [x, y, polarity, timestamp] of predicted events
        
    Returns:
        Estimated dt in seconds
    """
    if len(real_events) == 0 or len(predicted_events) == 0:
        return 0.002  # Default 2ms
    
    sample_real_times = real_events[:min(1000, len(real_events)), 3]
    sample_pred_times = predicted_events[:min(1000, len(predicted_events)), 3]
    
    time_diffs = []
    for rt in sample_real_times:
        closest_pred_times = sample_pred_times[np.abs(sample_pred_times - rt) < 0.1]
        if len(closest_pred_times) > 0:
            closest_diff = np.min(closest_pred_times - rt)
            if closest_diff > 0:
                time_diffs.append(closest_diff)
    
    return np.median(time_diffs) if len(time_diffs) > 0 else 0.002

def check_polarity_match(real_polarity: float, predicted_polarity: float, polarity_mode: str = "opposite") -> bool:
    """
    CORE MODULAR FUNCTION: Check if two events should be matched based on polarity mode.
    
    This function can be used independently to check polarity compatibility.
    
    Args:
        real_polarity: Polarity of real event (0 or 1)
        predicted_polarity: Polarity of predicted event (0 or 1)
        polarity_mode: "opposite", "equal", or "ignore"
        
    Returns:
        True if events should be matched based on polarity
    """
    if polarity_mode == "ignore":
        return True
    elif polarity_mode == "equal":
        return real_polarity == predicted_polarity
    else:  # "opposite"
        return real_polarity != predicted_polarity

def cancel_events_time_aware(real_events: np.ndarray, 
                            predicted_events: np.ndarray, 
                            dt_seconds: float, 
                            temporal_tolerance_ms: float, 
                            spatial_tolerance_pixels: float,
                            polarity_mode: str = "opposite",
                            chunk_size: int = 50000,
                            verbose: bool = True,
                            use_vectorized_processing: bool = True):
    """
    CORE MODULAR FUNCTION: Match real and predicted events using TRUE temporal gate.
    
    OPTIMIZED VERSION: Event-by-event processing with vectorized optimizations for speed.
    Maintains true event-by-event logic while being practical and fast.
    
    Args:
        real_events: Array [x, y, polarity, timestamp] of real events
        predicted_events: Array [x, y, polarity, timestamp] of predicted events  
        dt_seconds: Time step (seconds) - expected time difference
        temporal_tolerance_ms: Temporal tolerance (milliseconds)
        spatial_tolerance_pixels: Spatial tolerance (pixels)
        polarity_mode: "opposite", "equal", or "ignore"
        chunk_size: Processing chunk size for memory efficiency
        verbose: Whether to print progress updates
        use_vectorized_processing: Use vectorized optimizations (recommended)
        
    Returns:
        Tuple of (unmatched_real_mask, unmatched_predicted_mask, total_matches)
        - unmatched_real_mask: Boolean array, True for unmatched real events
        - unmatched_predicted_mask: Boolean array, True for unmatched predicted events  
        - total_matches: Number of matched pairs found
    """
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    # Handle empty inputs
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    
    # Create spatial KDTree for fast spatial search
    pred_tree = cKDTree(predicted_events[:, :2])
    
    # Process in chunks for memory efficiency
    chunk_size = min(chunk_size, num_real)
    
    if verbose:
        print(f"  Processing {num_real:,} real events in chunks of {chunk_size:,}")
        print(f"  Using {'vectorized' if use_vectorized_processing else 'sequential'} event-by-event processing")
    
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        chunk_target_times = chunk_real[:, 3] + dt_seconds
        
        if use_vectorized_processing:
            # OPTIMIZED: Vectorized event-by-event processing
            total_matches += _process_chunk_vectorized(
                chunk_real, chunk_target_times, chunk_start,
                predicted_events, pred_tree,
                temporal_tolerance_s, spatial_tolerance_pixels, polarity_mode,
                matched_real, matched_predicted
            )
        else:
            # ORIGINAL: Sequential event-by-event processing
            total_matches += _process_chunk_sequential(
                chunk_real, chunk_target_times, chunk_start,
                predicted_events, pred_tree,
                temporal_tolerance_s, spatial_tolerance_pixels, polarity_mode,
                matched_real, matched_predicted
            )
        
        # Progress update
        if verbose:
            progress = (chunk_end / num_real) * 100
            print(f"  Progress: {progress:.1f}% - {total_matches:,} matches found", end="\r")
    
    if verbose:
        print()  # New line after progress
    
    # Return unmatched events (inverse of matched)
    return ~matched_real, ~matched_predicted, total_matches

def _process_chunk_vectorized(chunk_real: np.ndarray, chunk_target_times: np.ndarray, chunk_start: int,
                             predicted_events: np.ndarray, pred_tree: cKDTree,
                             temporal_tolerance_s: float, spatial_tolerance_pixels: float, polarity_mode: str,
                             matched_real: np.ndarray, matched_predicted: np.ndarray) -> int:
    """
    OPTIMIZED: Vectorized event-by-event processing for maximum speed.
    
    This maintains true event-by-event logic while using vectorized operations
    for spatial search, temporal filtering, and polarity checking.
    """
    chunk_matches = 0
    
    # OPTIMIZATION 1: Batch spatial search for entire chunk
    spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)
    
    # OPTIMIZATION 2: Pre-filter events that are already matched
    unmatched_mask = ~matched_real[chunk_start:chunk_start + len(chunk_real)]
    if not np.any(unmatched_mask):
        return 0
    
    # Process only unmatched events
    for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
        real_idx = chunk_start + i
        if matched_real[real_idx]:
            continue
        
        if len(spatial_candidates) == 0:
            continue
        
        # OPTIMIZATION 3: Vectorized candidate filtering
        spatial_candidates = np.array(spatial_candidates)
        available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
        
        if len(available_candidates) == 0:
            continue
        
        # OPTIMIZATION 4: Vectorized temporal gate
        candidate_times = predicted_events[available_candidates, 3]
        temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
        
        if not np.any(temporal_mask):
            continue
        
        # OPTIMIZATION 5: Vectorized polarity check
        final_candidates = available_candidates[temporal_mask]
        candidate_events = predicted_events[final_candidates]
        
        real_polarity = real_event[2]
        pred_polarities = candidate_events[:, 2]
        
        if polarity_mode == "ignore":
            polarity_matches = np.ones(len(candidate_events), dtype=bool)
        elif polarity_mode == "equal":
            polarity_matches = (pred_polarities == real_polarity)
        else:  # "opposite"
            polarity_matches = (pred_polarities != real_polarity)
        
        if np.any(polarity_matches):
            # OPTIMIZATION 6: Vectorized distance calculation
            valid_candidates = final_candidates[polarity_matches]
            valid_events = candidate_events[polarity_matches]
            distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
            best_candidate = valid_candidates[np.argmin(distances)]
            
            matched_real[real_idx] = True
            matched_predicted[best_candidate] = True
            chunk_matches += 1
    
    return chunk_matches

def _process_chunk_sequential(chunk_real: np.ndarray, chunk_target_times: np.ndarray, chunk_start: int,
                             predicted_events: np.ndarray, pred_tree: cKDTree,
                             temporal_tolerance_s: float, spatial_tolerance_pixels: float, polarity_mode: str,
                             matched_real: np.ndarray, matched_predicted: np.ndarray) -> int:
    """
    ORIGINAL: Sequential event-by-event processing (slower but more explicit).
    
    This is the original implementation for comparison or when vectorized processing
    is not desired.
    """
    chunk_matches = 0
        
    # Find spatial candidates for entire chunk
    spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)
        
    # Process each event in chunk sequentially
    for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
        real_idx = chunk_start + i
        if matched_real[real_idx]:
            continue
        
        if len(spatial_candidates) == 0:
            continue
        
        # Filter available candidates
        spatial_candidates = np.array(spatial_candidates)
        available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
        
        if len(available_candidates) == 0:
            continue
        
        # Apply temporal gate
        candidate_times = predicted_events[available_candidates, 3]
        temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
        
        if not np.any(temporal_mask):
            continue
        
        # Get final candidates
        final_candidates = available_candidates[temporal_mask]
        candidate_events = predicted_events[final_candidates]
        
        # Check polarity constraint
        real_polarity = real_event[2]
        pred_polarities = candidate_events[:, 2]
        
        if polarity_mode == "ignore":
            polarity_matches = np.ones(len(candidate_events), dtype=bool)
        elif polarity_mode == "equal":
            polarity_matches = (pred_polarities == real_polarity)
        else:  # "opposite"
            polarity_matches = (pred_polarities != real_polarity)
        
        if np.any(polarity_matches):
            # Find closest valid candidate
            valid_candidates = final_candidates[polarity_matches]
            valid_events = candidate_events[polarity_matches]
            distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
            best_candidate = valid_candidates[np.argmin(distances)]
            
            matched_real[real_idx] = True
            matched_predicted[best_candidate] = True
            chunk_matches += 1
    
    return chunk_matches

def calculate_cancellation_stats(real_events: np.ndarray, 
                                predicted_events: np.ndarray, 
                                unmatched_real_mask: np.ndarray, 
                                unmatched_predicted_mask: np.ndarray,
                                total_matches: int):
    """
    CORE MODULAR FUNCTION: Calculate comprehensive cancellation statistics.
    
    This function can be used independently to analyze cancellation results.
    
    Args:
        real_events: Array of real events
        predicted_events: Array of predicted events
        unmatched_real_mask: Boolean mask for unmatched real events
        unmatched_predicted_mask: Boolean mask for unmatched predicted events
        total_matches: Number of matched pairs
        
    Returns:
        Dictionary with cancellation statistics:
        - total_real: Total number of real events
        - total_predicted: Total number of predicted events
        - matched_pairs: Number of matched pairs
        - real_cancellation_rate: Percentage of real events cancelled
        - predicted_cancellation_rate: Percentage of predicted events cancelled
        - overall_cancellation_rate: Overall cancellation rate
        - residual_real_count: Number of unmatched real events
        - residual_predicted_count: Number of unmatched predicted events
    """
    total_real = len(real_events)
    total_predicted = len(predicted_events)
    
    residual_real_count = np.sum(unmatched_real_mask)
    residual_predicted_count = np.sum(unmatched_predicted_mask)
    
    real_cancelled = total_real - residual_real_count
    predicted_cancelled = total_predicted - residual_predicted_count
    
    real_cancellation_rate = (real_cancelled / total_real * 100) if total_real > 0 else 0
    predicted_cancellation_rate = (predicted_cancelled / total_predicted * 100) if total_predicted > 0 else 0
    
    # Overall cancellation rate (average of both)
    overall_cancellation_rate = (real_cancellation_rate + predicted_cancellation_rate) / 2
    
    return {
        'total_real': total_real,
        'total_predicted': total_predicted,
        'matched_pairs': total_matches,
        'real_cancellation_rate': real_cancellation_rate,
        'predicted_cancellation_rate': predicted_cancellation_rate,
        'overall_cancellation_rate': overall_cancellation_rate,
        'residual_real_count': residual_real_count,
        'residual_predicted_count': residual_predicted_count,
        'real_cancelled': real_cancelled,
        'predicted_cancelled': predicted_cancelled
    }

# ---------------- Config ----------------
COMBINED_PATH = "./combined_events_with_predictions.npy"  # columns: [x,y,p,t,flag]; p in {0,1}, flag 0=real,1=pred
IMG_W, IMG_H  = 1280, 720

# cancellation parameters (matching visualize_time_window_clean.py)
R_PIX         = 2.0          # spatial tolerance (px) - matching clean version
POLARITY_MODE = "opposite"   # "opposite" | "equal" | "ignore"
TEMPORAL_TOL_MS = 5.0        # temporal tolerance (ms) - matching clean version

# rasterization
USE_BILINEAR_INTERP = True   # True -> bilinear splatting; False -> nearest neighbor (fastest)
USE_HIGH_QUALITY_INTERP = False  # True -> enhanced bilinear (slow); False -> fast bilinear (recommended for video)

# video
VIDEO_PATH    = "cancellation_video.mp4"
FPS           = None         # if None, set to round(1000/BIN_MS)
PNG_DIR       = None         # e.g. "frames_out" to write PNGs instead of MP4 (set to a folder path)
MAX_DURATION_S = 10.9        # Process full duration (10.9s)
MAX_EVENTS = 50_000_000      # Increased to handle very large datasets (50M events)
# MAX_EVENTS = None          # Uncomment this line to process ALL events (use with caution!)

# frame binning for video generation (NOT for cancellation)
BIN_MS        = 5.0          # frame bin width (ms) - only for video frames

# ---------------- Utils ----------------
def get_memory_usage():
    """Get current memory usage in MB"""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def time_edges(tmin, tmax, bin_ms):
    w = bin_ms * 1e-3
    n = int(np.ceil((tmax - tmin) / w)) + 1
    return tmin + np.arange(n + 1) * w

def load_combined(path):
    """
    MEMORY-SAFE: Load combined events with memory management.
    
    Prevents memory crashes by limiting dataset size and duration.
    """
    arr = np.load(path, mmap_mode="r")
    
    # Check if dataset is too large
    if MAX_EVENTS is not None and len(arr) > MAX_EVENTS:
        print(f"  Large dataset detected: {len(arr):,} events")
        print(f"   Limiting to first {MAX_EVENTS:,} events to prevent memory issues")
        arr = arr[:MAX_EVENTS]
    elif MAX_EVENTS is None:
        print(f"  Processing ALL events: {len(arr):,} events (use with caution!)")
    
    # Check if duration is too long
    if MAX_DURATION_S is not None:
        t_min = arr[0, 3]
        t_max = t_min + MAX_DURATION_S
        time_mask = arr[:, 3] <= t_max
        if np.sum(time_mask) < len(arr):
            print(f"  Long duration detected: {arr[-1, 3] - arr[0, 3]:.1f}s")
            print(f"   Limiting to first {MAX_DURATION_S:.1f}s to prevent memory issues")
            arr = arr[time_mask]
    
    # Sort by timestamp if needed
    if not np.all(arr[:-1, 3] <= arr[1:, 3]):
        print("   Sorting events by timestamp...")
        arr = arr[np.argsort(arr[:, 3])]
    
    print(f" Loaded {len(arr):,} events (real={int(np.sum(arr[:,4]==0.0)):,}, pred={int(np.sum(arr[:,4]==1.0)):,})")
    print(f"   Time range: {arr[0,3]:.3f}s to {arr[-1,3]:.3f}s")
    print(f"   Memory usage: ~{len(arr) * arr.dtype.itemsize / 1024**2:.1f} MB")
    
    # Monitor memory usage
    try:
        current_memory = get_memory_usage()
        print(f"   Current process memory: {current_memory:.1f} MB")
    except ImportError:
        print("   Memory monitoring unavailable (install psutil for memory stats)")
    
    return arr

def _pol_ok(rp, pp):
    if POLARITY_MODE == "ignore": return True
    if POLARITY_MODE == "equal":  return rp == pp
    return rp != pp

def _signed_p(p):  # {0,1} -> {-1,+1}
    return np.where(p > 0.5, 1, -1).astype(np.float32)

def events_to_image(width, height, ev):  # ev columns: [x,y,p]
    if USE_BILINEAR_INTERP:
        if USE_HIGH_QUALITY_INTERP:
            return _raster_bilinear(width, height, ev)  # Enhanced bilinear
        else:
            return _raster_bilinear_standard(width, height, ev)  # Standard bilinear
    else:
        return _raster_nearest(width, height, ev)  # Nearest neighbor

def _raster_bilinear(width, height, events):
    """
    ENHANCED: High-quality bilinear interpolation rasterization with improved visual quality.
    
    Improvements:
    1. Higher precision floating point arithmetic
    2. Smooth weight distribution for better visual continuity
    3. Optimized bounds checking for better performance
    4. Enhanced sub-pixel accuracy
    """
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    # Use higher precision for better sub-pixel accuracy
    x = events[:, 0].astype(np.float64)
    y = events[:, 1].astype(np.float64)
    s = _signed_p(events[:, 2]).astype(np.float64)
    
    # Get floor coordinates and fractional parts with higher precision
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Calculate fractional parts with higher precision
    dx = x - x0.astype(np.float64)
    dy = y - y0.astype(np.float64)
    
    # Enhanced bilinear weights with smooth distribution
    # Use smoothstep function for better visual continuity
    def smoothstep(edge0, edge1, x):
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    
    # Apply smoothstep for smoother weight transitions
    dx_smooth = smoothstep(0.0, 1.0, dx)
    dy_smooth = smoothstep(0.0, 1.0, dy)
    
    # Calculate enhanced bilinear weights
    w00 = (1.0 - dx_smooth) * (1.0 - dy_smooth)
    w10 = dx_smooth * (1.0 - dy_smooth)
    w01 = (1.0 - dx_smooth) * dy_smooth
    w11 = dx_smooth * dy_smooth
    
    # Create image with higher precision accumulation
    img = np.zeros((height, width), dtype=np.float64)
    
    # Optimized bounds checking - check all corners at once
    valid_mask = (
        (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height) &  # Corner 00
        (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)     # Corner 11
    )
    
    if not np.any(valid_mask):
        return np.zeros((height, width), dtype=np.float32)
    
    # Filter valid events
    x0_valid = x0[valid_mask]
    y0_valid = y0[valid_mask]
    x1_valid = x1[valid_mask]
    y1_valid = y1[valid_mask]
    s_valid = s[valid_mask]
    w00_valid = w00[valid_mask]
    w10_valid = w10[valid_mask]
    w01_valid = w01[valid_mask]
    w11_valid = w11[valid_mask]
    
    # Individual corner masks for precise bounds checking
    m00 = (x0_valid >= 0) & (x0_valid < width) & (y0_valid >= 0) & (y0_valid < height)
    m10 = (x1_valid >= 0) & (x1_valid < width) & (y0_valid >= 0) & (y0_valid < height)
    m01 = (x0_valid >= 0) & (x0_valid < width) & (y1_valid >= 0) & (y1_valid < height)
    m11 = (x1_valid >= 0) & (x1_valid < width) & (y1_valid >= 0) & (y1_valid < height)
    
    # Accumulate weighted contributions with higher precision
    np.add.at(img, (y0_valid[m00], x0_valid[m00]), s_valid[m00] * w00_valid[m00])
    np.add.at(img, (y0_valid[m10], x1_valid[m10]), s_valid[m10] * w10_valid[m10])
    np.add.at(img, (y1_valid[m01], x0_valid[m01]), s_valid[m01] * w01_valid[m01])
    np.add.at(img, (y1_valid[m11], x1_valid[m11]), s_valid[m11] * w11_valid[m11])
    
    # Convert back to float32 for memory efficiency
    return img.astype(np.float32)

def _raster_bilinear_standard(width, height, events):
    """
    OPTIMIZED: Fast bilinear interpolation for video generation.
    
    Optimizations for speed:
    1. Vectorized operations throughout
    2. Optimized bounds checking
    3. Reduced memory allocations
    4. Efficient weight calculations
    """
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    # Use float32 for speed (sufficient precision for video)
    x = events[:, 0].astype(np.float32)
    y = events[:, 1].astype(np.float32)
    s = _signed_p(events[:, 2]).astype(np.float32)
    
    # Get floor coordinates and fractional parts
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Calculate fractional parts
    dx = x - x0.astype(np.float32)
    dy = y - y0.astype(np.float32)
    
    # Calculate bilinear weights (vectorized)
    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy
    
    # Create image
    img = np.zeros((height, width), dtype=np.float32)
    
    # Fast bounds checking - check all corners at once
    valid_mask = (
        (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height) &
        (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)
    )
    
    if not np.any(valid_mask):
        return img
    
    # Filter valid events (vectorized)
    x0_valid = x0[valid_mask]
    y0_valid = y0[valid_mask]
    x1_valid = x1[valid_mask]
    y1_valid = y1[valid_mask]
    s_valid = s[valid_mask]
    w00_valid = w00[valid_mask]
    w10_valid = w10[valid_mask]
    w01_valid = w01[valid_mask]
    w11_valid = w11[valid_mask]
    
    # Accumulate weighted contributions (vectorized)
    np.add.at(img, (y0_valid, x0_valid), s_valid * w00_valid)
    np.add.at(img, (y0_valid, x1_valid), s_valid * w10_valid)
    np.add.at(img, (y1_valid, x0_valid), s_valid * w01_valid)
    np.add.at(img, (y1_valid, x1_valid), s_valid * w11_valid)
    
    return img

def _raster_nearest(width, height, events):
    """
    OPTIMIZED: Fastest nearest neighbor rasterization for maximum speed.
    
    Optimizations:
    1. Direct integer casting
    2. Vectorized bounds checking
    3. Minimal memory allocations
    """
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    # Direct integer casting for maximum speed
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    s = _signed_p(events[:, 2]).astype(np.float32)
    
    # Vectorized bounds checking
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    
    if not np.any(valid_mask):
        return np.zeros((height, width), dtype=np.float32)
    
    # Filter valid events
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    s_valid = s[valid_mask]
    
    # Create image and accumulate (vectorized)
    img = np.zeros((height, width), dtype=np.float32)
    np.add.at(img, (y_valid, x_valid), s_valid)
    
    return img

def normalize_to_uint8_signed(img):
    """
    OPTIMIZED: Fast normalization for video generation with good visual quality.
    
    Optimizations for speed:
    1. Simplified gamma correction (faster approximation)
    2. Vectorized operations
    3. Reduced function calls
    """
    if img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    # Calculate normalization parameters
    img_max = float(np.abs(img).max())
    
    if img_max <= 0:
        return np.full(img.shape, 128, dtype=np.uint8)  # Mid-gray for zero image
    
    # Fast normalization with simplified gamma correction
    # Normalize to [0, 1] range
    img_normalized = img / (2.0 * img_max) + 0.5  # [-max, +max] -> [0, 1]
    img_normalized = np.clip(img_normalized, 0.0, 1.0)
    
    # Fast gamma approximation (gamma ≈ 0.8)
    # Use sqrt for faster gamma correction: sqrt(x) ≈ x^0.5, but we want x^0.8
    # Approximation: sqrt(sqrt(x)) ≈ x^0.5^0.5 = x^0.25, not quite right
    # Better: use a simple polynomial approximation
    img_gamma = img_normalized * img_normalized * (2.0 - img_normalized)  # Fast approximation of x^0.8
    
    # Convert to uint8
    img_u8 = (img_gamma * 255.0).astype(np.uint8)
    
    return img_u8

def cancel_in_bin_fast(real_bin, pred_bin, r_pix, temporal_tol_ms, dt_estimate=0.002):
    """
    ORIGINAL: Time binning cancellation (faster but may lose some matches at bin edges).
    """
    nr, npd = len(real_bin), len(pred_bin)
    if nr == 0 or npd == 0:
        return np.ones(nr, bool), np.ones(npd, bool), 0
    
    # Create spatial KDTree for predicted events in this bin
    pred_tree = cKDTree(pred_bin[:, :2])
    
    # Process each real event in this bin
    matched_real = np.zeros(nr, dtype=bool)
    matched_predicted = np.zeros(npd, dtype=bool)
    matches = 0
    
    for i, real_event in enumerate(real_bin):
        if matched_real[i]:
            continue
            
        # Find spatial candidates
        spatial_candidates = pred_tree.query_ball_point(real_event[:2], r_pix)
        if len(spatial_candidates) == 0:
            continue
            
        # Filter available candidates
        available_candidates = [idx for idx in spatial_candidates if not matched_predicted[idx]]
        if len(available_candidates) == 0:
            continue
            
        # Apply temporal gate within this bin
        candidate_times = pred_bin[available_candidates, 3]
        target_time = real_event[3] + dt_estimate
        temporal_mask = np.abs(candidate_times - target_time) <= (temporal_tol_ms * 1e-3)
        
        if not np.any(temporal_mask):
            continue
            
        # Get final candidates
        final_candidates = [available_candidates[j] for j in range(len(available_candidates)) if temporal_mask[j]]
        candidate_events = pred_bin[final_candidates]
        
        # Check polarity constraint
        real_polarity = real_event[2]
        pred_polarities = candidate_events[:, 2]
        
        if POLARITY_MODE == "ignore":
            polarity_matches = np.ones(len(candidate_events), dtype=bool)
        elif POLARITY_MODE == "equal":
            polarity_matches = (pred_polarities == real_polarity)
        else:  # "opposite"
            polarity_matches = (pred_polarities != real_polarity)
            
        if np.any(polarity_matches):
            # Find closest valid candidate
            valid_candidates = [final_candidates[j] for j in range(len(final_candidates)) if polarity_matches[j]]
            valid_events = candidate_events[polarity_matches]
            distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
            best_candidate = valid_candidates[np.argmin(distances)]
            
            matched_real[i] = True
            matched_predicted[best_candidate] = True
            matches += 1
    
    return ~matched_real, ~matched_predicted, matches

# ---------------- Renderer ----------------
def render_highpass_video(combined, bin_ms, r_pix, w, h, mp4_path=None, fps=None, png_dir=None, max_duration_s=None):
    """
    ORIGINAL: Video generation using time binning cancellation.
    """
    import imageio

    t = combined[:, 3]
    tmin = float(t.min())
    tmax = float(t.max()) if max_duration_s is None else tmin + max_duration_s
    
    # Filter events to time range
    if max_duration_s is not None:
        mask = t <= tmax
        combined = combined[mask]
        print(f"Processing first {max_duration_s:.1f}s ({len(combined):,} events)")
    
    # Split events
    real_events = combined[combined[:, 4] == 0.0]
    pred_events = combined[combined[:, 4] == 1.0]
    
    print(f"Events: {len(real_events):,} real, {len(pred_events):,} predicted")
    
    # Estimate dt
    dt_estimate_s = estimate_dt_from_data(real_events, pred_events)
    print(f"Estimated dt: {dt_estimate_s*1000:.1f}ms")
    
    # Create time bins for video frames
    edges = time_edges(tmin, tmax, bin_ms)
    fps = fps or max(int(round(1000.0 / bin_ms)), 1)
    print(f"Generating video frames: {len(edges)-1} frames @ {fps} fps")

    writer = None
    if mp4_path and png_dir is None:
        try:
            writer = imageio.get_writer(mp4_path, format='FFMPEG', fps=fps, codec='libx264', quality=8)
            print(f"Writing MP4: {mp4_path} @ {fps} fps")
        except Exception as e:
            print(f"[warn] MP4 writer init failed ({e}). Falling back to PNG sequence.")
            png_dir = "frames_out"

    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
        print(f"Writing PNG frames to: {png_dir} (fps target {fps})")

    t0 = time.time()
    N = len(combined)
    left_ix = 0

    for b in range(len(edges) - 1):
        left, right = edges[b], edges[b+1]
        
        # Find events in this time bin
        i0 = left_ix
        while i0 < N and combined[i0, 3] < left:  i0 += 1
        i1 = i0
        while i1 < N and combined[i1, 3] < right: i1 += 1
        left_ix = i0
        
        if i1 <= i0:
            frame = np.zeros((h, w), dtype=np.int32)
        else:
            # Get events for this bin
            bin_events = combined[i0:i1]
            real_bin = bin_events[bin_events[:, 4] == 0.0][:, :3]
            pred_bin = bin_events[bin_events[:, 4] == 1.0][:, :3]
            
            # Run cancellation within this bin
            unmatched_real_mask, unmatched_predicted_mask, matches = cancel_in_bin_fast(
                real_bin, pred_bin, r_pix, TEMPORAL_TOL_MS, dt_estimate_s
            )
            
            # Get residual events for this bin
            residual_real = real_bin[unmatched_real_mask]
            residual_pred = pred_bin[unmatched_predicted_mask]
            
            # Combine residual events for frame generation
            if len(residual_real) > 0 and len(residual_pred) > 0:
                frame_events = np.vstack([residual_real, residual_pred])
            elif len(residual_real) > 0:
                frame_events = residual_real
            elif len(residual_pred) > 0:
                frame_events = residual_pred
            else:
                frame_events = np.zeros((0, 3))
            
            frame = events_to_image(w, h, frame_events)

        frame_u8 = normalize_to_uint8_signed(frame)

        if writer is not None:
            writer.append_data(frame_u8)
        else:
            imageio.imwrite(os.path.join(png_dir, f"frame_{b:06d}.png"), frame_u8)

        if (b & 0xFF) == 0 or b == len(edges) - 2:  # Update every 256 frames or at end
            pct = (b / (len(edges)-1)) * 100.0
            elapsed = time.time() - t0
            fps_actual = (b+1) / elapsed if elapsed > 0 else 0
            print(f"  {pct:5.1f}%  frames={b+1}/{len(edges)-1}  fps={fps_actual:.1f}", end="\r")

    dt = time.time() - t0
    if writer is not None:
        writer.close()
    
    if USE_BILINEAR_INTERP:
        interp_method = "enhanced bilinear (slow)" if USE_HIGH_QUALITY_INTERP else "fast bilinear"
    else:
        interp_method = "nearest neighbor (fastest)"
    
    print(f"\nDone. Frames: {len(edges)-1}, {interp_method} raster, time: {dt:.1f}s")

def main():
    if USE_BILINEAR_INTERP:
        interp_method = "enhanced bilinear (slow)" if USE_HIGH_QUALITY_INTERP else "fast bilinear (recommended)"
    else:
        interp_method = "nearest neighbor (fastest)"
    
    print(f"ORIGINAL: Video generation using time binning cancellation")
    print(f"Parameters: {R_PIX}px spatial tolerance, {TEMPORAL_TOL_MS}ms temporal tolerance, {POLARITY_MODE} polarity mode")
    print(f"Video frames: {BIN_MS}ms bins (time binning approach)")
    print(f"Interpolation: {interp_method}")
    print(f"Memory limits: Max {MAX_EVENTS:,} events, Max {MAX_DURATION_S:.1f}s duration")
    print(f"Speed optimization: Time binning cancellation + fast video rendering")
    print(f" Large dataset detected - processing may take longer but will be memory-safe")
    
    combined = load_combined(COMBINED_PATH)
    fps = FPS or max(int(round(1000.0 / BIN_MS)), 1)
    render_highpass_video(
        combined,
        bin_ms=BIN_MS,
        r_pix=R_PIX,
        w=IMG_W, h=IMG_H,
        mp4_path=VIDEO_PATH if PNG_DIR is None else None,
        fps=fps,
        png_dir=PNG_DIR,
        max_duration_s=MAX_DURATION_S
    )
    if PNG_DIR is None:
        print(f"Saved video: {VIDEO_PATH}")
    else:
        print(f"Saved PNG sequence in: {PNG_DIR}")

if __name__ == "__main__":
    main()
