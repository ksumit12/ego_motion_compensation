#!/usr/bin/env python3
"""
Ultra-Fast Event Cancellation and Video Generation (Test Version)
================================================================

This script processes a small subset first to test the concept:
1. Loads only first 1 second of data
2. Uses time binning for both cancellation and video generation
3. Applies bilinear interpolation and rasterization
4. Generates video output

Perfect for testing the pipeline with large datasets.
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
BIN_MS = 5.0  # time bin width (ms) - used for BOTH cancellation AND video frames
VIDEO_PATH = "cancellation_video_test.mp4"
FPS = None  # Auto-calculate from BIN_MS
MAX_DURATION_S = 1.0  # Process only first 1 second for testing
USE_BILINEAR_INTERP = True  # Use bilinear interpolation

# =============== Core Functions ===============

def load_combined_subset(path: str, max_duration_s: float = 1.0) -> np.ndarray:
    """Load combined events data from file - subset only."""
    print(f"Loading events from {path}...")
    arr = np.load(path, mmap_mode="r")
    
    # Get first max_duration_s seconds
    t_min = arr[0, 3]
    t_max = t_min + max_duration_s
    time_mask = arr[:, 3] <= t_max
    arr = arr[time_mask]
    
    print(f"Loaded subset: {len(arr):,} events (real={int(np.sum(arr[:,4]==0.0)):,}, pred={int(np.sum(arr[:,4]==1.0)):,})")
    print(f"Time range: {arr[0,3]:.3f}s to {arr[-1,3]:.3f}s")
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

def cancel_in_bin(real_bin: np.ndarray, pred_bin: np.ndarray, r_pix: float, 
                 temporal_tol_ms: float, dt_estimate: float = 0.002) -> Tuple[np.ndarray, np.ndarray, int]:
    """Cancel events within a single time bin."""
    nr, npd = len(real_bin), len(pred_bin)
    if nr == 0 or npd == 0:
        return np.ones(nr, bool), np.ones(npd, bool), 0
    
    pred_tree = cKDTree(pred_bin[:, :2])
    matched_real = np.zeros(nr, dtype=bool)
    matched_predicted = np.zeros(npd, dtype=bool)
    matches = 0
    
    for i, real_event in enumerate(real_bin):
        if matched_real[i]:
            continue
            
        spatial_candidates = pred_tree.query_ball_point(real_event[:2], r_pix)
        if len(spatial_candidates) == 0:
            continue
            
        available_candidates = [idx for idx in spatial_candidates if not matched_predicted[idx]]
        if len(available_candidates) == 0:
            continue
            
        candidate_times = pred_bin[available_candidates, 3]
        target_time = real_event[3] + dt_estimate
        temporal_mask = np.abs(candidate_times - target_time) <= (temporal_tol_ms * 1e-3)
        
        if not np.any(temporal_mask):
            continue
            
        final_candidates = [available_candidates[j] for j in range(len(available_candidates)) if temporal_mask[j]]
        candidate_events = pred_bin[final_candidates]
        
        real_polarity = real_event[2]
        pred_polarities = candidate_events[:, 2]
        
        if POLARITY_MODE == "ignore":
            polarity_matches = np.ones(len(candidate_events), dtype=bool)
        elif POLARITY_MODE == "equal":
            polarity_matches = (pred_polarities == real_polarity)
        else:  # "opposite"
            polarity_matches = (pred_polarities != real_polarity)
            
        if np.any(polarity_matches):
            valid_candidates = [final_candidates[j] for j in range(len(final_candidates)) if polarity_matches[j]]
            valid_events = candidate_events[polarity_matches]
            distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
            best_candidate = valid_candidates[np.argmin(distances)]
            
            matched_real[i] = True
            matched_predicted[best_candidate] = True
            matches += 1
    
    return ~matched_real, ~matched_predicted, matches

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
    
    img_normalized = img / (2.0 * img_max) + 0.5
    img_normalized = np.clip(img_normalized, 0.0, 1.0)
    img_u8 = (img_normalized * 255.0).astype(np.uint8)
    
    return img_u8

def time_edges(tmin: float, tmax: float, bin_ms: float) -> np.ndarray:
    """Generate time bin edges for video frames."""
    w = bin_ms * 1e-3
    n = int(np.ceil((tmax - tmin) / w)) + 1
    return tmin + np.arange(n + 1) * w

# =============== Main Processing Function ===============

def process_and_render_video():
    """Main function to process events and render video using time binning."""
    print("Loading combined events data (subset only)...")
    combined = load_combined_subset(COMBINED_PATH, MAX_DURATION_S)
    
    # Split events
    real_events = combined[combined[:, 4] == 0.0]
    pred_events = combined[combined[:, 4] == 1.0]
    
    print(f"Events: {len(real_events):,} real, {len(pred_events):,} predicted")
    
    # Estimate dt
    dt_seconds = estimate_dt_from_data(real_events, pred_events)
    print(f"Estimated dt: {dt_seconds*1000:.1f}ms")
    
    # Generate video using time binning approach
    print("Generating video using time binning cancellation...")
    render_video_with_binning(combined, dt_seconds)

def render_video_with_binning(combined_events: np.ndarray, dt_estimate: float):
    """Render video using time binning approach."""
    t = combined_events[:, 3]
    tmin = float(t.min())
    tmax = float(t.max())
    
    # Create time bins for video frames
    edges = time_edges(tmin, tmax, BIN_MS)
    fps = FPS or max(int(round(1000.0 / BIN_MS)), 1)
    print(f"Generating {len(edges)-1} frames @ {fps} fps using {BIN_MS}ms bins")
    
    # Initialize video writer
    writer = imageio.get_writer(VIDEO_PATH, format='FFMPEG', fps=fps, codec='libx264', quality=8)
    
    start_time = time.time()
    N = len(combined_events)
    left_ix = 0
    total_matches = 0
    
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
            # Get events for this bin
            bin_events = combined_events[i0:i1]
            real_bin = bin_events[bin_events[:, 4] == 0.0][:, :3]
            pred_bin = bin_events[bin_events[:, 4] == 1.0][:, :3]
            
            # Run cancellation within this bin
            unmatched_real_mask, unmatched_predicted_mask, matches = cancel_in_bin(
                real_bin, pred_bin, R_PIX, TEMPORAL_TOL_MS, dt_estimate
            )
            
            total_matches += matches
            
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
            
            # Create frame using bilinear interpolation
            frame = create_per_pixel_count_image(IMG_W, IMG_H, frame_events)
        
        # Normalize and write frame
        frame_u8 = normalize_to_uint8_signed(frame)
        writer.append_data(frame_u8)
        
        # Progress update
        if b % 10 == 0 or b == len(edges) - 2:
            pct = (b / (len(edges)-1)) * 100.0
            elapsed = time.time() - start_time
            fps_actual = (b+1) / elapsed if elapsed > 0 else 0
            print(f"  {pct:5.1f}%  frames={b+1}/{len(edges)-1}  fps={fps_actual:.1f}  matches={total_matches:,}", end="\r")
    
    writer.close()
    
    elapsed = time.time() - start_time
    print(f"\nVideo generation completed in {elapsed:.1f}s")
    print(f"Total matches found: {total_matches:,}")
    print(f"Saved video: {VIDEO_PATH}")

# =============== Main Execution ===============

if __name__ == "__main__":
    print("Ultra-Fast Event Cancellation and Video Generation (Test Version)")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Spatial tolerance: {R_PIX}px")
    print(f"  Temporal tolerance: {TEMPORAL_TOL_MS}ms")
    print(f"  Polarity mode: {POLARITY_MODE}")
    print(f"  Time bin width: {BIN_MS}ms (used for BOTH cancellation AND video frames)")
    print(f"  Bilinear interpolation: {USE_BILINEAR_INTERP}")
    print(f"  Max duration: {MAX_DURATION_S}s (TEST MODE - small subset)")
    print()
    print("APPROACH: Time binning cancellation + video generation")
    print("  - Processes only first {MAX_DURATION_S}s for testing")
    print("  - Divides timeline into {BIN_MS}ms bins")
    print("  - Performs cancellation within each bin")
    print("  - Creates video frame from residual events in each bin")
    print("  - Ultra-fast for testing pipeline with large datasets")
    print()
    
    process_and_render_video()
    
    print("\nProcessing complete!")











