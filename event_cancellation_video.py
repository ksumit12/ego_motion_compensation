#!/usr/bin/env python3
"""
Concise Event Cancellation and Video Generation Script
=====================================================

This script uses core functions directly to:
1. Load combined events numpy array
2. Perform event cancellation using true temporal gate
3. Apply bilinear interpolation and rasterization
4. Generate video output

Usage:
    python event_cancellation_video.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import imageio
from typing import Tuple, Dict

# =============== Configuration ===============
COMBINED_PATH = "./combined_events_with_predictions.npy"
IMG_W, IMG_H = 1280, 720
R_PIX = 2.0  # spatial tolerance (pixels)
TEMPORAL_TOL_MS = 5.0  # temporal tolerance (milliseconds)
POLARITY_MODE = "opposite"  # "opposite", "equal", or "ignore"
BIN_MS = 5.0  # frame bin width (ms)
VIDEO_PATH = "cancellation_video.mp4"
FPS = None  # Auto-calculate from BIN_MS
MAX_DURATION_S = 10.0  # Process first 10 seconds
USE_BILINEAR_INTERP = True  # Use bilinear interpolation

# =============== Core Functions (copied from visualize_time_window_clean.py) ===============

def load_combined(path: str) -> np.ndarray:
    """Load combined events data from file."""
    arr = np.load(path, mmap_mode="r")
    if not np.all(arr[:-1, 3] <= arr[1:, 3]):
        arr = arr[np.argsort(arr[:, 3])]
    print(f"Loaded {len(arr):,} events (real={int(np.sum(arr[:,4]==0.0)):,}, pred={int(np.sum(arr[:,4]==1.0)):,})")
    return arr

def estimate_dt_from_data(real_events: np.ndarray, predicted_events: np.ndarray) -> float:
    """Estimate dt from real and predicted event timestamps."""
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

def cancel_events_time_aware(real_events: np.ndarray, 
                            predicted_events: np.ndarray, 
                            dt_seconds: float, 
                            temporal_tolerance_ms: float, 
                            spatial_tolerance_pixels: float,
                            polarity_mode: str = "opposite",
                            chunk_size: int = 50000,
                            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    """Match real and predicted events using TRUE temporal gate."""
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    
    pred_tree = cKDTree(predicted_events[:, :2])
    chunk_size = min(chunk_size, num_real)
    
    if verbose:
        print(f"  Processing {num_real:,} real events in chunks of {chunk_size:,}")
    
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        chunk_target_times = chunk_real[:, 3] + dt_seconds
        
        total_matches += _process_chunk_vectorized(
            chunk_real, chunk_target_times, chunk_start,
            predicted_events, pred_tree,
            temporal_tolerance_s, spatial_tolerance_pixels, polarity_mode,
            matched_real, matched_predicted
        )
        
        if verbose:
            progress = (chunk_end / num_real) * 100
            print(f"  Progress: {progress:.1f}% - {total_matches:,} matches found", end="\r")
    
    if verbose:
        print()
    
    return ~matched_real, ~matched_predicted, total_matches

def _process_chunk_vectorized(chunk_real: np.ndarray, chunk_target_times: np.ndarray, chunk_start: int,
                             predicted_events: np.ndarray, pred_tree: cKDTree,
                             temporal_tolerance_s: float, spatial_tolerance_pixels: float, polarity_mode: str,
                             matched_real: np.ndarray, matched_predicted: np.ndarray) -> int:
    """Vectorized event-by-event processing for maximum speed."""
    chunk_matches = 0
    
    spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)
    
    for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
        real_idx = chunk_start + i
        if matched_real[real_idx]:
            continue
        
        if len(spatial_candidates) == 0:
            continue
        
        spatial_candidates = np.array(spatial_candidates)
        available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
        
        if len(available_candidates) == 0:
            continue
        
        candidate_times = predicted_events[available_candidates, 3]
        temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
        
        if not np.any(temporal_mask):
            continue
        
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
            valid_candidates = final_candidates[polarity_matches]
            valid_events = candidate_events[polarity_matches]
            distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
            best_candidate = valid_candidates[np.argmin(distances)]
            
            matched_real[real_idx] = True
            matched_predicted[best_candidate] = True
            chunk_matches += 1
    
    return chunk_matches

def create_per_pixel_count_image(width: int, height: int, events: np.ndarray) -> np.ndarray:
    """Create per-pixel signed count image using bilinear interpolation."""
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    x = events[:, 0].astype(np.float32)
    y = events[:, 1].astype(np.float32)
    s = np.where(events[:, 2] > 0.5, 1, -1).astype(np.float32)
    
    # Bilinear interpolation
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = x - x0
    dy = y - y0
    
    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy
    
    img = np.zeros((height, width), dtype=np.float32)
    
    # Bounds checking masks
    m00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    m10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    m01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    m11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)
    
    # Accumulate weighted contributions
    np.add.at(img, (y0[m00], x0[m00]), s[m00] * w00[m00])
    np.add.at(img, (y0[m10], x1[m10]), s[m10] * w10[m10])
    np.add.at(img, (y1[m01], x0[m01]), s[m01] * w01[m01])
    np.add.at(img, (y1[m11], x1[m11]), s[m11] * w11[m11])
    
    return img

def normalize_to_uint8_signed(img: np.ndarray) -> np.ndarray:
    """Normalize signed image to uint8 for video output."""
    if img.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    
    img_max = float(np.abs(img).max())
    
    if img_max <= 0:
        return np.full(img.shape, 128, dtype=np.uint8)
    
    # Normalize to [0, 1] range
    img_normalized = img / (2.0 * img_max) + 0.5
    img_normalized = np.clip(img_normalized, 0.0, 1.0)
    
    # Convert to uint8
    img_u8 = (img_normalized * 255.0).astype(np.uint8)
    
    return img_u8

def time_edges(tmin: float, tmax: float, bin_ms: float) -> np.ndarray:
    """Generate time bin edges for video frames."""
    w = bin_ms * 1e-3
    n = int(np.ceil((tmax - tmin) / w)) + 1
    return tmin + np.arange(n + 1) * w

# =============== Main Processing Function ===============

def process_and_render_video():
    """Main function to process events and render video."""
    print("Loading combined events data...")
    combined = load_combined(COMBINED_PATH)
    
    # Limit duration if specified
    if MAX_DURATION_S is not None:
        t_min = combined[0, 3]
        t_max = t_min + MAX_DURATION_S
        time_mask = combined[:, 3] <= t_max
        combined = combined[time_mask]
        print(f"Limited to first {MAX_DURATION_S:.1f}s: {len(combined):,} events")
    
    # Split events
    real_events = combined[combined[:, 4] == 0.0]
    pred_events = combined[combined[:, 4] == 1.0]
    
    print(f"Events: {len(real_events):,} real, {len(pred_events):,} predicted")
    
    # Estimate dt
    dt_seconds = estimate_dt_from_data(real_events, pred_events)
    print(f"Estimated dt: {dt_seconds*1000:.1f}ms")
    
    # Run cancellation
    print("Running event cancellation...")
    start_time = time.time()
    
    unmatched_real_mask, unmatched_predicted_mask, total_matches = cancel_events_time_aware(
        real_events, pred_events, dt_seconds, TEMPORAL_TOL_MS, R_PIX,
        polarity_mode=POLARITY_MODE, verbose=True
    )
    
    residual_real_events = real_events[unmatched_real_mask]
    residual_predicted_events = pred_events[unmatched_predicted_mask]
    
    elapsed_time = time.time() - start_time
    print(f"Cancellation completed in {elapsed_time:.1f}s")
    print(f"Matched pairs: {total_matches:,}")
    print(f"Residual events: {len(residual_real_events):,} real, {len(residual_predicted_events):,} predicted")
    
    # Generate video
    print("Generating video...")
    render_video(combined, residual_real_events, residual_predicted_events)

def render_video(combined_events: np.ndarray, 
                residual_real_events: np.ndarray, 
                residual_predicted_events: np.ndarray):
    """Render video from residual events."""
    t = combined_events[:, 3]
    tmin = float(t.min())
    tmax = float(t.max())
    
    # Create time bins for video frames
    edges = time_edges(tmin, tmax, BIN_MS)
    fps = FPS or max(int(round(1000.0 / BIN_MS)), 1)
    print(f"Generating {len(edges)-1} frames @ {fps} fps")
    
    # Initialize video writer
    writer = imageio.get_writer(VIDEO_PATH, format='FFMPEG', fps=fps, codec='libx264', quality=8)
    
    start_time = time.time()
    N = len(combined_events)
    left_ix = 0
    
    for b in range(len(edges) - 1):
        left, right = edges[b], edges[b+1]
        
        # Find events in this time bin
        i0 = left_ix
        while i0 < N and combined_events[i0, 3] < left:
            i0 += 1
        i1 = i0
        while i1 < N and combined_events[i1, 3] < right:
            i1 += 1
        left_ix = i0
        
        if i1 <= i0:
            frame = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        else:
            # Get residual events for this bin
            bin_events = combined_events[i0:i1]
            
            # Filter residual events to this time bin
            real_mask = (residual_real_events[:, 3] >= left) & (residual_real_events[:, 3] < right)
            pred_mask = (residual_predicted_events[:, 3] >= left) & (residual_predicted_events[:, 3] < right)
            
            frame_real = residual_real_events[real_mask][:, :3] if np.any(real_mask) else np.zeros((0, 3))
            frame_pred = residual_predicted_events[pred_mask][:, :3] if np.any(pred_mask) else np.zeros((0, 3))
            
            # Combine residual events for frame generation
            if len(frame_real) > 0 and len(frame_pred) > 0:
                frame_events = np.vstack([frame_real, frame_pred])
            elif len(frame_real) > 0:
                frame_events = frame_real
            elif len(frame_pred) > 0:
                frame_events = frame_pred
            else:
                frame_events = np.zeros((0, 3))
            
            # Create frame using bilinear interpolation
            if USE_BILINEAR_INTERP:
                frame = create_per_pixel_count_image(IMG_W, IMG_H, frame_events)
            else:
                # Simple nearest neighbor fallback
                frame = np.zeros((IMG_H, IMG_W), dtype=np.float32)
                if len(frame_events) > 0:
                    x = frame_events[:, 0].astype(np.int32)
                    y = frame_events[:, 1].astype(np.int32)
                    s = np.where(frame_events[:, 2] > 0.5, 1, -1).astype(np.float32)
                    
                    valid_mask = (x >= 0) & (x < IMG_W) & (y >= 0) & (y < IMG_H)
                    if np.any(valid_mask):
                        np.add.at(frame, (y[valid_mask], x[valid_mask]), s[valid_mask])
        
        # Normalize and write frame
        frame_u8 = normalize_to_uint8_signed(frame)
        writer.append_data(frame_u8)
        
        # Progress update
        if (b & 0xFF) == 0 or b == len(edges) - 2:
            pct = (b / (len(edges)-1)) * 100.0
            elapsed = time.time() - start_time
            fps_actual = (b+1) / elapsed if elapsed > 0 else 0
            print(f"  {pct:5.1f}%  frames={b+1}/{len(edges)-1}  fps={fps_actual:.1f}", end="\r")
    
    writer.close()
    
    elapsed = time.time() - start_time
    print(f"\nVideo generation completed in {elapsed:.1f}s")
    print(f"Saved video: {VIDEO_PATH}")

# =============== Main Execution ===============

if __name__ == "__main__":
    print("Event Cancellation and Video Generation")
    print("=" * 50)
    print(f"Parameters:")
    print(f"  Spatial tolerance: {R_PIX}px")
    print(f"  Temporal tolerance: {TEMPORAL_TOL_MS}ms")
    print(f"  Polarity mode: {POLARITY_MODE}")
    print(f"  Frame bin width: {BIN_MS}ms")
    print(f"  Bilinear interpolation: {USE_BILINEAR_INTERP}")
    print(f"  Max duration: {MAX_DURATION_S}s")
    print()
    
    process_and_render_video()
    
    print("\nProcessing complete!")











