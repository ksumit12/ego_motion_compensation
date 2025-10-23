#!/usr/bin/env python3
"""
Comprehensive analysis of both dt values and tolerance combinations.
This script tests different dt values AND different spatial/temporal tolerance combinations
to find the optimal parameters for maximum cancellation rate.

Interactive 3D Visualization Dependencies (optional):
    pip install pyvista          # Recommended - creates VTK, PLY, and HTML formats
    pip install mayavi           # Alternative - creates Mayavi scene files
    
The script will work without these packages, but won't save interactive 3D formats.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
from scipy.spatial import cKDTree
from itertools import product
from math import ceil

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# Optional imports for interactive 3D visualization
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    from mayavi import mlab
    HAS_MAYAVI = True
except ImportError:
    HAS_MAYAVI = False

# =============== Configuration ===============
# Point to the 5s prediction set on the Windows partition by default
WINDOW_PREDICTIONS_DIR = "./window_predictions"

# DT values to test (excluding 0ms as it gives unrealistic 100% cancellation)
DT_VALUES_MS = [1, 2, 3, 4, 5]

VERBOSE = True
USE_TQDM = True

# Tolerance ranges to test
SPATIAL_TOLERANCE_RANGE = (0.5, 5.0, 0.5)  # (min, max, step) in pixels
TEMPORAL_TOLERANCE_RANGE = (1.0, 10.0, 1.0)  # (min, max, step) in milliseconds

# Optimization settings
MAX_SPATIAL_TOLERANCE = 5.0  # Maximum spatial tolerance for caching
SPATIAL_QUERY_K = 32  # Number of nearest neighbors to query
MAX_EVENTS_PER_COMBINATION = 50000  # Memory limit for large datasets

# Polarity mode
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# Disc center coordinates and radius
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

# Time windows
WINDOWS = [
    (5.000, 5.010),
    (8.200, 8.210),
    (9.000, 9.010),
]

# Output settings
OUTPUT_DIR = "./dt_tolerance_analysis_results"
PLOT_DPI = 150
INTERACTIVE_3D = True  # Show interactive 3D surface after saving
SAVE_INTERACTIVE_3D = True  # Save interactive 3D formats (PyVista/Mayavi)
SHOW_4D_PLOTS_REALTIME = True  # Show 4D correlation plots in real-time (enabled for interactive viewing)

# =============== Utility Functions ===============
def circle_mask(x, y, cx, cy, r, scale=1.05):
    """Create boolean mask for points inside scaled circle"""
    return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

def check_polarity_match(real_polarity, predicted_polarity):
    """Check if two events should be matched based on polarity mode"""
    if POLARITY_MODE == "ignore":
        return True
    elif POLARITY_MODE == "equal":
        return real_polarity == predicted_polarity
    else:  # "opposite" mode
        return real_polarity != predicted_polarity

def load_npy_or_npz(path, key=None):
    """Load .npy or .npz. If .npz, use provided key or the first array."""
    if path.endswith('.npz'):
        with np.load(path) as z:
            if key is not None and key in z:
                return z[key]
            first_key = list(z.files)[0]
            return z[first_key]
    else:
        return np.load(path, mmap_mode='r')

def try_load_combined_or_split(window_dir, dt_ms):
    """Try combined; else use real+pred split and combine on the fly."""
    combined_candidates = [
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:02d}ms.npz"),
    ]
    for path in combined_candidates:
        if os.path.exists(path):
            arr = load_npy_or_npz(path, key='combined')
            return arr
    real_candidates = [
        os.path.join(window_dir, 'real_events.npy'),
        os.path.join(window_dir, 'real_events.npz'),
    ]
    pred_candidates = [
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npy"),
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npz"),
    ]
    real = None
    for p in real_candidates:
        if os.path.exists(p):
            real = load_npy_or_npz(p, key='real')
            break
    if real is None:
        raise FileNotFoundError(f"Missing real_events in {window_dir}")
    pred = None
    for p in pred_candidates:
        if os.path.exists(p):
            pred = load_npy_or_npz(p, key='pred')
            break
    if pred is None:
        raise FileNotFoundError(f"Missing predictions for dt={dt_ms}ms in {window_dir}")
    # Combine into 5-col array with flags and sort by t
    real_flag = np.zeros((len(real), 1), dtype=np.float32)
    pred_flag = np.ones((len(pred), 1), dtype=np.float32)
    stacks = []
    if len(real) > 0:
        stacks.append(np.column_stack([real, real_flag]))
    if len(pred) > 0:
        stacks.append(np.column_stack([pred, pred_flag]))
    combined = np.vstack(stacks) if stacks else np.zeros((0, 5), dtype=np.float32)
    return combined[np.argsort(combined[:, 3])]

def build_spatial_cache(real_events, predicted_events, dt_seconds, max_spatial_tolerance, k=32):
    """
    Build spatial neighbor cache once per (dt, window) combination.
    Returns precomputed spatial neighbors, distances, and derived data.
    """
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    if num_real == 0 or num_predicted == 0:
        return None
    
    # Limit events for memory efficiency
    if num_real > MAX_EVENTS_PER_COMBINATION:
        indices = np.random.choice(num_real, MAX_EVENTS_PER_COMBINATION, replace=False)
        real_events = real_events[indices]
        num_real = len(real_events)
    
    # Build spatial tree
    pred_tree = cKDTree(predicted_events[:, :2])
    
    # Single spatial query with maximum tolerance
    distances, indices = pred_tree.query(
        real_events[:, :2], 
        k=k, 
        distance_upper_bound=max_spatial_tolerance,
        workers=-1
    )
    
    # Precompute derived data
    target_times = real_events[:, 3] + dt_seconds
    real_polarities = real_events[:, 2]
    
    # Handle k=1 case (ensure 2D arrays)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    
    cache = {
        'real_events': real_events,
        'predicted_events': predicted_events,
        'distances': distances,
        'indices': indices,
        'target_times': target_times,
        'real_polarities': real_polarities,
        'num_real': num_real,
        'num_predicted': num_predicted,
        'dt_seconds': dt_seconds
    }
    
    return cache

def cancel_events_from_cache(cache, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Fast cancellation using precomputed spatial cache.
    Just applies tolerance masks to cached data.
    """
    if cache is None:
        return np.array([]), np.array([]), 0
    
    distances = cache['distances']
    indices = cache['indices']
    target_times = cache['target_times']
    real_polarities = cache['real_polarities']
    predicted_events = cache['predicted_events']
    num_real = cache['num_real']
    num_predicted = cache['num_predicted']
    
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    
    # Process each real event
    for i in range(num_real):
        if matched_real[i]:
            continue
            
        real_target_time = target_times[i]
        real_polarity = real_polarities[i]
        
        # Get spatial candidates within tolerance
        spatial_mask = distances[i] <= spatial_tolerance_pixels
        valid_indices_mask = indices[i] < num_predicted
        combined_mask = spatial_mask & valid_indices_mask
        
        if not np.any(combined_mask):
            continue
            
        candidate_indices = indices[i][combined_mask]
        
        # Filter already matched candidates
        available_mask = ~matched_predicted[candidate_indices]
        if not np.any(available_mask):
            continue
            
        candidate_indices = candidate_indices[available_mask]
        candidate_events = predicted_events[candidate_indices]
        
        # Apply temporal constraint
        candidate_times = candidate_events[:, 3]
        temporal_mask = np.abs(candidate_times - real_target_time) <= temporal_tolerance_s
        
        if not np.any(temporal_mask):
            continue
            
        # Apply polarity constraint
        candidate_polarities = candidate_events[temporal_mask, 2]
        if POLARITY_MODE == "ignore":
            polarity_mask = np.ones(len(candidate_polarities), dtype=bool)
        elif POLARITY_MODE == "equal":
            polarity_mask = (candidate_polarities == real_polarity)
        else:  # "opposite"
            polarity_mask = (candidate_polarities != real_polarity)
            
        if not np.any(polarity_mask):
            continue
            
        # Get final valid candidates
        final_candidates = candidate_indices[temporal_mask][polarity_mask]
        final_events = candidate_events[temporal_mask][polarity_mask]
        
        # Choose closest spatial match
        final_distances = np.sum((final_events[:, :2] - cache['real_events'][i, :2])**2, axis=1)
        best_idx = np.argmin(final_distances)
        best_candidate = final_candidates[best_idx]
        
        # Mark as matched
        matched_real[i] = True
        matched_predicted[best_candidate] = True
        total_matches += 1
    
    # Return unmatched events
    unmatched_real = cache['real_events'][~matched_real]
    unmatched_predicted = predicted_events[~matched_predicted]
    
    return unmatched_real, unmatched_predicted, total_matches

def cancel_events_time_aware(real_events, predicted_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """True temporal gate with spatial KDTree (k-NN, bounded radius, parallel)."""
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0

    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0

    pred_tree = cKDTree(predicted_events[:, :2])
    chunk_size = min(50000, num_real)
    num_chunks = ceil(max(num_real, 1) / max(chunk_size, 1))
    pbar = None
    if VERBOSE and USE_TQDM and tqdm is not None:
        pbar = tqdm(total=num_chunks, desc="    matching chunks", leave=False)

    K = 8
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        chunk_target_times = chunk_real[:, 3] + dt_seconds

        dists, inds = pred_tree.query(chunk_real[:, :2], k=K, distance_upper_bound=spatial_tolerance_pixels, workers=-1)
        if K == 1:
            dists = dists[:, None]
            inds = inds[:, None]

        for i, (real_event, target_time) in enumerate(zip(chunk_real, chunk_target_times)):
            real_idx = chunk_start + i
            if matched_real[real_idx]:
                continue
            spatial_candidates = inds[i]
            valid_mask = (spatial_candidates < num_predicted)
            if not np.any(valid_mask):
                continue
            spatial_candidates = spatial_candidates[valid_mask]
            avail_mask = ~matched_predicted[spatial_candidates]
            if not np.any(avail_mask):
                continue
            spatial_candidates = spatial_candidates[avail_mask]
            candidate_times = predicted_events[spatial_candidates, 3]
            temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
            if not np.any(temporal_mask):
                continue
            final_candidates = spatial_candidates[temporal_mask]
            cand_events = predicted_events[final_candidates]
            # Polarity
            real_pol = real_event[2]
            if POLARITY_MODE == "ignore":
                pol_mask = np.ones(len(cand_events), dtype=bool)
            elif POLARITY_MODE == "equal":
                pol_mask = (cand_events[:, 2] == real_pol)
            else:
                pol_mask = (cand_events[:, 2] != real_pol)
            if not np.any(pol_mask):
                continue
            valid_candidates = final_candidates[pol_mask]
            valid_events = cand_events[pol_mask]
            distances = np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1)
            best_candidate = valid_candidates[int(np.argmin(distances))]
            matched_real[real_idx] = True
            matched_predicted[best_candidate] = True
            total_matches += 1
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    return ~matched_real, ~matched_predicted, total_matches

def cancel_events_in_time_bin(real_events, predicted_events, spatial_tolerance_pixels):
    """Match real and predicted events within a time bin and return unmatched events"""
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    # Create a tree for fast spatial lookup of predicted events
    predicted_tree = cKDTree(predicted_events[:, :2])
    
    # Find closest predicted event for each real event within tolerance
    distances, closest_indices = predicted_tree.query(
        real_events[:, :2], k=1, distance_upper_bound=spatial_tolerance_pixels
    )
    
    # Find real events that have a valid match within tolerance
    valid_matches = np.where(closest_indices < num_predicted)[0]
    
    if len(valid_matches) == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    # Create list of potential matches with distance, real index, predicted index
    potential_matches = []
    for real_idx in valid_matches:
        predicted_idx = int(closest_indices[real_idx])
        distance = float(distances[real_idx])
        
        # Check if polarities are compatible
        if check_polarity_match(real_events[real_idx, 2], predicted_events[predicted_idx, 2]):
            potential_matches.append((distance, real_idx, predicted_idx))
    
    if len(potential_matches) == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    # Sort by distance (closest matches first)
    potential_matches.sort(key=lambda match: match[0])
    
    # Greedily assign matches
    used_predicted = set()
    matched_real = np.zeros(num_real, bool)
    matched_predicted = np.zeros(num_predicted, bool)
    
    for distance, real_idx, predicted_idx in potential_matches:
        if predicted_idx not in used_predicted:
            used_predicted.add(predicted_idx)
            matched_real[real_idx] = True
            matched_predicted[predicted_idx] = True
    
    # Return unmatched events (inverse of matched)
    unmatched_real = ~matched_real
    unmatched_predicted = ~matched_predicted
    num_matches = int(matched_real.sum())
    
    return unmatched_real, unmatched_predicted, num_matches

def time_edges(tmin, tmax, bin_ms):
    """Generate time bin edges"""
    w = bin_ms * 1e-3
    n = int(np.ceil((tmax - tmin) / w)) + 1
    return tmin + np.arange(n+1) * w

def run_cancellation_cached(cache, temporal_tolerance_ms, spatial_tolerance_pixels):
    """Run ego-motion cancellation using cached spatial data."""
    if cache is None:
        return np.zeros((0, 5)), np.zeros((0, 5)), 0
    
    unmatched_real, unmatched_pred, total_matches = cancel_events_from_cache(
        cache, temporal_tolerance_ms, spatial_tolerance_pixels
    )
    
    return unmatched_real, unmatched_pred, total_matches

def run_cancellation_true_gate(combined_events, temporal_tolerance_ms, spatial_tolerance_pixels):
    """Run ego-motion cancellation using TRUE temporal gate (no binning)."""
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    if len(real_events) == 0 or len(pred_events) == 0:
        return np.zeros((0, 5), dtype=combined_events.dtype), np.zeros((0, 5), dtype=combined_events.dtype), 0

    # Estimate dt from data (robust median of positive deltas)
    sample_real_times = real_events[:min(1000, len(real_events)), 3]
    sample_pred_times = pred_events[:min(1000, len(pred_events)), 3]
    time_diffs = []
    for rt in sample_real_times:
        closest_pred_times = sample_pred_times[np.abs(sample_pred_times - rt) < 0.1]
        if len(closest_pred_times) > 0:
            closest_diff = np.min(closest_pred_times - rt)
            if closest_diff > 0:
                time_diffs.append(closest_diff)
    dt_seconds = np.median(time_diffs) if len(time_diffs) > 0 else 0.002

    # Pre-filter ROI to accelerate
    cx, cy, r = DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS
    r_pred = r + spatial_tolerance_pixels + 2.0
    def _circle_mask_xy(arr, rad):
        return (arr[:, 0] - cx)**2 + (arr[:, 1] - cy)**2 <= (rad * 1.05)**2
    real_events_roi = real_events[_circle_mask_xy(real_events, r)]
    pred_events_roi = pred_events[_circle_mask_xy(pred_events, r_pred)]

    unmatched_real_mask, unmatched_pred_mask, total_matches = cancel_events_time_aware(
        real_events_roi, pred_events_roi, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels
    )

    residual_real = real_events_roi[unmatched_real_mask]
    residual_pred = pred_events_roi[unmatched_pred_mask]
    return residual_real, residual_pred, total_matches

def calculate_roi_cancellation_rate(combined_events, residual_real_events, disc_center, disc_radius):
    """Calculate cancellation rate specifically for the circular ROI"""
    # Extract real events from combined data
    real_events = combined_events[combined_events[:, 4] == 0.0]
    
    if len(real_events) == 0:
        return 0.0, 0, 0
    
    # Filter real events to ROI
    cx, cy = disc_center
    roi_mask = circle_mask(real_events[:, 0], real_events[:, 1], cx, cy, disc_radius)
    roi_real_events = real_events[roi_mask]
    
    if len(roi_real_events) == 0:
        return 0.0, 0, 0
    
    # Filter residual events to ROI
    if len(residual_real_events) > 0:
        roi_residual_mask = circle_mask(residual_real_events[:, 0], residual_real_events[:, 1], cx, cy, disc_radius)
        roi_residual_events = residual_real_events[roi_residual_mask]
    else:
        roi_residual_events = np.zeros((0, 5), dtype=combined_events.dtype)
    
    # Calculate cancellation rate
    total_roi_real = len(roi_real_events)
    total_roi_residual = len(roi_residual_events)
    total_roi_cancelled = total_roi_real - total_roi_residual
    cancellation_rate = (total_roi_cancelled / total_roi_real * 100) if total_roi_real > 0 else 0.0
    
    return cancellation_rate, total_roi_real, total_roi_cancelled

def analyze_all_combinations():
    """Analyze all combinations of dt values and tolerance parameters using optimized caching"""
    print("=== Comprehensive DT and Tolerance Analysis (Optimized) ===")
    
    # Generate tolerance values
    spatial_values = np.arange(SPATIAL_TOLERANCE_RANGE[0], 
                              SPATIAL_TOLERANCE_RANGE[1] + SPATIAL_TOLERANCE_RANGE[2], 
                              SPATIAL_TOLERANCE_RANGE[2])
    temporal_values = np.arange(TEMPORAL_TOLERANCE_RANGE[0], 
                               TEMPORAL_TOLERANCE_RANGE[1] + TEMPORAL_TOLERANCE_RANGE[2], 
                               TEMPORAL_TOLERANCE_RANGE[2])
    
    print(f"DT values: {DT_VALUES_MS}")
    print(f"Spatial tolerances: {spatial_values}")
    print(f"Temporal tolerances: {temporal_values}")
    print(f"Max spatial tolerance for caching: {MAX_SPATIAL_TOLERANCE}")
    print(f"Spatial query k: {SPATIAL_QUERY_K}")
    
    total_combinations = len(DT_VALUES_MS) * len(spatial_values) * len(temporal_values) * len(WINDOWS)
    print(f"Total combinations: {total_combinations}")
    
    # Initialize results storage
    results = []
    current_combination = 0
    
    # Test each dt value
    for dt_ms in DT_VALUES_MS:
        print(f"\nProcessing dt={dt_ms}ms...")
        dt_seconds = dt_ms * 1e-3
        
        # Test each window
        for window_idx, window in enumerate(WINDOWS):
            t0, t1 = window
            print(f"  Window {window_idx + 1}: {t0:.3f}s to {t1:.3f}s")
            
            # Load prediction data for this window and dt
            window_dir = os.path.join(WINDOW_PREDICTIONS_DIR, f"window_{window_idx + 1}_{t0:.3f}s_to_{t1:.3f}s")
            try:
                combined_events = try_load_combined_or_split(window_dir, dt_ms)
            except FileNotFoundError as e:
                print(f"    Warning: {e}; skipping window {window_idx + 1}")
                continue
            
            # Extract and filter events to ROI
            real_events = combined_events[combined_events[:, 4] == 0.0]
            pred_events = combined_events[combined_events[:, 4] == 1.0]
            
            if len(real_events) == 0 or len(pred_events) == 0:
                print(f"    Warning: No events in window {window_idx + 1}")
                continue
            
            # Pre-filter to ROI for efficiency
            cx, cy, r = DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS
            r_pred = r + MAX_SPATIAL_TOLERANCE + 2.0
            def _circle_mask_xy(arr, rad):
                return (arr[:, 0] - cx)**2 + (arr[:, 1] - cy)**2 <= (rad * 1.05)**2
            
            real_events_roi = real_events[_circle_mask_xy(real_events, r)]
            pred_events_roi = pred_events[_circle_mask_xy(pred_events, r_pred)]
            
            print(f"    ROI events: real={len(real_events_roi):,}, pred={len(pred_events_roi):,}")
            
            # BUILD CACHE ONCE per (dt, window)
            print(f"    Building spatial cache...")
            cache_start = time.time()
            cache = build_spatial_cache(
                real_events_roi, pred_events_roi, dt_seconds, 
                MAX_SPATIAL_TOLERANCE, k=SPATIAL_QUERY_K
            )
            cache_time = time.time() - cache_start
            print(f"    Cache built in {cache_time:.2f}s")
            
            if cache is None:
                print(f"    Warning: Failed to build cache for window {window_idx + 1}")
                continue
            
            # Test all tolerance combinations using the cache
            tolerance_combinations = len(spatial_values) * len(temporal_values)
            print(f"    Testing {tolerance_combinations} tolerance combinations...")
            
            combo_start = time.time()
            for spatial_tol in spatial_values:
                for temporal_tol in temporal_values:
                    current_combination += 1
                    
                    if current_combination % 100 == 0:
                        print(f"    Progress: {current_combination}/{total_combinations} ({current_combination/total_combinations*100:.1f}%)")
                    
                    # Fast cancellation using cache
                    residual_real, residual_pred, matched_pairs = run_cancellation_cached(
                        cache, temporal_tol, spatial_tol
                    )
                    
                    # Calculate ROI cancellation rate
                    roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
                        combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
                    )
                    
                    # Store results
                    results.append({
                        'dt_ms': dt_ms,
                        'window_idx': window_idx + 1,
                        'window_start': t0,
                        'window_end': t1,
                        'spatial_tolerance': spatial_tol,
                        'temporal_tolerance': temporal_tol,
                        'cancellation_rate': roi_cancellation_rate,
                        'total_roi_real': total_roi_real,
                        'total_roi_cancelled': total_roi_cancelled,
                        'total_matched_pairs': matched_pairs
                    })
            
            combo_time = time.time() - combo_start
            print(f"    Tolerance combinations completed in {combo_time:.2f}s")
            print(f"    Average time per combination: {combo_time/tolerance_combinations*1000:.1f}ms")
    
    return results

def save_interactive_3d_surface(X, Y, Z, best_dt, output_dir):
    """Save interactive 3D surface plots using PyVista and/or Mayavi"""
    
    # Create separate folder for interactive 3D files
    interactive_3d_dir = os.path.join(output_dir, "interactive_3d")
    os.makedirs(interactive_3d_dir, exist_ok=True)
    print(f"Creating interactive 3D files in: {interactive_3d_dir}")
    
    # Save data as numpy arrays for manual loading later
    data_path = os.path.join(interactive_3d_dir, "3d_surface_data.npz")
    np.savez(data_path, X=X, Y=Y, Z=Z, dt=best_dt)
    print(f"Saved 3D surface data: {data_path}")
    print("  You can load this data later with: data = np.load('3d_surface_data.npz')")
    
    # PyVista format (VTK-based, widely supported)
    if HAS_PYVISTA:
        try:
            # Create structured grid
            grid = pv.StructuredGrid(X, Y, Z)
            grid["Cancellation Rate"] = Z.ravel(order='F')
            
            # Save as VTK file (can be opened in ParaView, PyVista, etc.)
            vtk_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface.vtk")
            grid.save(vtk_path)
            print(f"Saved PyVista/VTK format: {vtk_path}")
            print("  Open with: pv.read('tolerance_3d_surface.vtk').plot()")
            
            # Save as PLY format (also widely supported)
            ply_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface.ply")
            grid.save(ply_path)
            print(f"Saved PLY format: {ply_path}")
            
            # Create and save an interactive HTML file
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, scalars="Cancellation Rate", cmap='viridis', 
                           show_scalar_bar=True, scalar_bar_args={'title': 'Cancellation Rate (%)'})
            plotter.add_axes()
            plotter.camera_position = 'iso'
            
            html_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface_interactive.html")
            plotter.export_html(html_path)
            print(f"Saved interactive HTML: {html_path}")
            print("  Open in web browser for interactive viewing")
            
        except Exception as e:
            print(f"Warning: PyVista save failed: {e}")
    else:
        print("PyVista not available. Install with: pip install pyvista")
    
    # Mayavi format
    if HAS_MAYAVI:
        try:
            # Create Mayavi figure
            mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
            surf = mlab.surf(X, Y, Z, colormap='viridis')
            mlab.axes(xlabel='Spatial Tolerance (pixels)', 
                     ylabel='Temporal Tolerance (ms)', 
                     zlabel='Cancellation Rate (%)')
            mlab.title(f'Cancellation Rate vs Tolerance Parameters (dt = {best_dt}ms)')
            mlab.colorbar(surf, title='Cancellation Rate (%)')
            
            # Save as Mayavi scene file
            mayavi_path = os.path.join(interactive_3d_dir, "tolerance_3d_surface.mv2")
            mlab.savefig(mayavi_path)
            print(f"Saved Mayavi format: {mayavi_path}")
            print("  Open with Mayavi2 application or mlab.load_engine()")
            
            mlab.close()
            
        except Exception as e:
            print(f"Warning: Mayavi save failed: {e}")
    else:
        print("Mayavi not available. Install with: pip install mayavi")
    
    # Create a simple Python script to reload and view the data
    viewer_script = f'''#!/usr/bin/env python3
"""
Interactive 3D Surface Viewer
Load and display the 3D tolerance surface plot interactively.

Usage:
    python view_3d_surface.py

Requirements:
    pip install pyvista  # or mayavi
"""

import numpy as np
import os

# Try PyVista first (recommended)
try:
    import pyvista as pv
    
    def view_with_pyvista():
        # Load the VTK file
        if os.path.exists("tolerance_3d_surface.vtk"):
            mesh = pv.read("tolerance_3d_surface.vtk")
            mesh.plot(scalars="Cancellation Rate", cmap='viridis', 
                     show_scalar_bar=True, 
                     scalar_bar_args={{'title': 'Cancellation Rate (%)'}},
                     window_size=[1024, 768])
        else:
            # Load from numpy data
            data = np.load("3d_surface_data.npz")
            X, Y, Z = data['X'], data['Y'], data['Z']
            dt = data['dt']
            
            grid = pv.StructuredGrid(X, Y, Z)
            grid["Cancellation Rate"] = Z.ravel(order='F')
            
            plotter = pv.Plotter(window_size=[1024, 768])
            plotter.add_mesh(grid, scalars="Cancellation Rate", cmap='viridis', 
                           show_scalar_bar=True, 
                           scalar_bar_args={{'title': 'Cancellation Rate (%)'}})
            plotter.add_axes()
            plotter.add_text(f'dt = {{dt}}ms', position='upper_left')
            plotter.show()
    
    if __name__ == "__main__":
        view_with_pyvista()

except ImportError:
    # Fallback to Mayavi
    try:
        from mayavi import mlab
        
        def view_with_mayavi():
            if os.path.exists("tolerance_3d_surface.mv2"):
                # Load Mayavi scene
                mlab.load_engine()
                # Note: Direct scene loading might need manual implementation
                print("Load the .mv2 file manually in Mayavi2 application")
            else:
                # Load from numpy data
                data = np.load("3d_surface_data.npz")
                X, Y, Z = data['X'], data['Y'], data['Z']
                dt = data['dt']
                
                mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
                surf = mlab.surf(X, Y, Z, colormap='viridis')
                mlab.axes(xlabel='Spatial Tolerance (pixels)', 
                         ylabel='Temporal Tolerance (ms)', 
                         zlabel='Cancellation Rate (%)')
                mlab.title(f'Cancellation Rate vs Tolerance Parameters (dt = {{dt}}ms)')
                mlab.colorbar(surf, title='Cancellation Rate (%)')
                mlab.show()
        
        if __name__ == "__main__":
            view_with_mayavi()
            
    except ImportError:
        print("Neither PyVista nor Mayavi available.")
        print("Install one of them:")
        print("  pip install pyvista  # Recommended")
        print("  pip install mayavi   # Alternative")
        
        # Fallback: show how to load data manually
        print("\\nYou can still load the data manually:")
        print("  data = np.load('3d_surface_data.npz')")
        print("  X, Y, Z = data['X'], data['Y'], data['Z']")
        print("  # Then use your preferred 3D plotting library")
'''
    
    script_path = os.path.join(interactive_3d_dir, "view_3d_surface.py")
    with open(script_path, 'w') as f:
        f.write(viewer_script)
    print(f"Created viewer script: {script_path}")
    print("  Run with: python view_3d_surface.py")
    
    # Also create a README file in the interactive_3d folder
    readme_content = f"""# Interactive 3D Surface Files

This folder contains interactive 3D visualization files for the tolerance analysis surface plot.

## Files:
- `3d_surface_data.npz` - Raw NumPy data (X, Y, Z coordinates and dt value)
- `tolerance_3d_surface.vtk` - VTK format (ParaView, PyVista compatible)
- `tolerance_3d_surface.ply` - PLY format (widely supported)
- `tolerance_3d_surface_interactive.html` - Interactive HTML (open in web browser)
- `tolerance_3d_surface.mv2` - Mayavi scene file
- `view_3d_surface.py` - Python script to view the surface

## How to View:

### Option 1: Python Script (Recommended)
```bash
cd interactive_3d
python view_3d_surface.py
```

### Option 2: PyVista (if installed)
```python
import pyvista as pv
mesh = pv.read('tolerance_3d_surface.vtk')
mesh.plot(scalars="Cancellation Rate", cmap='viridis')
```

### Option 3: Web Browser
Open `tolerance_3d_surface_interactive.html` in any web browser

### Option 4: ParaView
Open `tolerance_3d_surface.vtk` in ParaView application

## Surface Data:
- **dt value**: {best_dt}ms
- **X-axis**: Spatial Tolerance (pixels)
- **Y-axis**: Temporal Tolerance (ms)  
- **Z-axis**: Cancellation Rate (%)
- **Color**: Cancellation Rate (%)

## Dependencies:
```bash
pip install pyvista  # Recommended
# or
pip install mayavi  # Alternative
```
"""
    
    readme_path = os.path.join(interactive_3d_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Created README: {readme_path}")

def create_dt_spatial_temporal_correlation(results_df, output_dir):
    """
    Visualize correlation across dt, spatial tolerance, temporal tolerance, and cancellation rate.
    - 3D scatter: X=dt, Y=spatial, Z=temporal, color=cancellation rate
    - Heatmaps per dt: temporal vs spatial
    - Optional PyVista interactive point cloud if pyvista is installed
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 3D scatter (matplotlib) ----------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs = results_df['dt_ms'].values.astype(float)
    ys = results_df['spatial_tolerance'].values.astype(float)
    zs = results_df['temporal_tolerance'].values.astype(float)
    cs = results_df['cancellation_rate'].values.astype(float)

    sc = ax.scatter(xs, ys, zs, c=cs, s=30, alpha=0.85, cmap='viridis')
    cb = fig.colorbar(sc, pad=0.1)
    cb.set_label('Cancellation rate (%)')

    ax.set_xlabel('dt (ms)')
    ax.set_ylabel('Spatial tolerance (px)')
    ax.set_zlabel('Temporal tolerance (ms)')
    ax.set_title('dt × Spatial × Temporal → Cancellation rate (color)')

    out_path = os.path.join(output_dir, 'dt_spatial_temporal_3d_scatter.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    # Show interactive 4D plot in real-time
    if SHOW_4D_PLOTS_REALTIME:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    plt.close(fig)

    # ---------- Heatmaps per dt (small multiples) ----------
    dt_vals = sorted(results_df['dt_ms'].unique())
    n = len(dt_vals)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 3.8*rows), squeeze=False)

    for idx, dt in enumerate(dt_vals):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        sub = results_df[results_df['dt_ms'] == dt]
        # Average across windows if present
        sub = sub.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
        piv = sub.pivot_table(index='temporal_tolerance', columns='spatial_tolerance', values='cancellation_rate')
        sns.heatmap(piv.sort_index().sort_index(axis=1), cmap='viridis', ax=ax, cbar=True, 
                   cbar_kws={'shrink': 0.8}, fmt='.1f', annot=True if len(piv) <= 10 else False)
        ax.set_title(f'dt = {dt} ms')
        ax.set_xlabel('Spatial tolerance (px)')
        ax.set_ylabel('Temporal tolerance (ms)')

    # Hide empty subplots
    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axes[r][c].axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'dt_spatial_temporal_heatmaps.png')
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved: {out_path}")
    
    # Show interactive heatmaps in real-time
    if SHOW_4D_PLOTS_REALTIME:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display heatmaps: {e}")
    plt.close(fig)

    # ---------- Optional: interactive 3D with PyVista ----------
    if HAS_PYVISTA:
        try:
            import pyvista as pv
            cloud = pv.PolyData(np.column_stack([xs, ys, zs]))
            cloud['Cancellation rate (%)'] = cs
            p = pv.Plotter(off_screen=True)
            p.add_points(cloud, scalars='Cancellation rate (%)', cmap='viridis', point_size=8, render_points_as_spheres=True)
            p.add_axes()
            p.add_text('dt–Spatial–Temporal correlation', font_size=10)
            
            # Save interactive HTML
            html_path = os.path.join(output_dir, 'dt_spatial_temporal_pointcloud.html')
            p.export_html(html_path)
            print(f"Saved interactive HTML (PyVista): {html_path}")
            
            # Also save as VTK for later viewing
            vtk_path = os.path.join(output_dir, 'dt_spatial_temporal_pointcloud.vtk')
            cloud.save(vtk_path)
            print(f"Saved VTK point cloud: {vtk_path}")
            
        except Exception as e:
            print(f"Warning: PyVista 4D visualization failed: {e}")

def create_3d_surface_for_specific_dt(results_df, dt_value, output_dir, show_interactive=True):
    """Create 3D surface plot for a specific dt value"""
    dt_data = results_df[results_df['dt_ms'] == dt_value]
    
    if len(dt_data) == 0:
        print(f"No data found for dt={dt_value}ms")
        return
    
    # Average across windows
    combined_data = dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
    
    spatial_unique = sorted(combined_data['spatial_tolerance'].unique())
    temporal_unique = sorted(combined_data['temporal_tolerance'].unique())
    
    X, Y = np.meshgrid(spatial_unique, temporal_unique)
    Z = np.zeros_like(X)
    
    for i, temp in enumerate(temporal_unique):
        for j, spat in enumerate(spatial_unique):
            mask = (combined_data['spatial_tolerance'] == spat) & (combined_data['temporal_tolerance'] == temp)
            if mask.any():
                Z[i, j] = combined_data[mask]['cancellation_rate'].iloc[0]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Spatial Tolerance (pixels)')
    ax.set_ylabel('Temporal Tolerance (ms)')
    ax.set_zlabel('Cancellation Rate (%)')
    ax.set_title(f'Cancellation Rate vs Tolerance Parameters (dt = {dt_value}ms)')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Save plot
    out_path = os.path.join(output_dir, f"tolerance_3d_surface_dt_{dt_value}ms.png")
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    
    # Show interactive plot
    if show_interactive:
        try:
            plt.show()
        except Exception as e:
            print(f"Could not display plot: {e}")
    plt.close(fig)

def interactive_plot_viewer(results_df, output_dir):
    """Interactive menu system for viewing different dt plots"""
    available_dt_values = sorted(results_df['dt_ms'].unique())
    
    print(f"\nINTERACTIVE PLOT VIEWER")
    print(f"Available dt values: {available_dt_values}")
    
    while True:
        print(f"\nPLOT SELECTION MENU")
        print("Options:")
        for dt in available_dt_values:
            print(f"  {int(dt)}: 3D surface for dt = {int(dt)}ms")
        print("  all: 4D correlation plot")
        print("  best: Best dt surface plot")
        print("  quit: Exit")
        
        choice = input("\nChoice: ").strip().lower()
        
        if choice == 'quit':
            break
        elif choice == 'all':
            create_dt_spatial_temporal_correlation(results_df, output_dir)
        elif choice == 'best':
            create_best_dt_surface_plot(results_df, output_dir)
        elif choice in [str(int(dt)) for dt in available_dt_values]:
            dt_val = int(choice)
            create_3d_surface_for_specific_dt(results_df, dt_val, output_dir, show_interactive=True)
        else:
            print("Invalid choice. Try again.")

def create_best_dt_surface_plot(results_df, output_dir):
    """Create the original best dt surface plot"""
    best_dt = results_df.groupby('dt_ms')['cancellation_rate'].mean().idxmax()
    best_dt_data = results_df[results_df['dt_ms'] == best_dt]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Average across windows
    combined_data = best_dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
    
    spatial_unique = sorted(combined_data['spatial_tolerance'].unique())
    temporal_unique = sorted(combined_data['temporal_tolerance'].unique())
    
    X, Y = np.meshgrid(spatial_unique, temporal_unique)
    Z = np.zeros_like(X)
    
    for i, temp in enumerate(temporal_unique):
        for j, spat in enumerate(spatial_unique):
            mask = (combined_data['spatial_tolerance'] == spat) & (combined_data['temporal_tolerance'] == temp)
            if mask.any():
                Z[i, j] = combined_data[mask]['cancellation_rate'].iloc[0]
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Spatial Tolerance (pixels)')
    ax.set_ylabel('Temporal Tolerance (ms)')
    ax.set_zlabel('Cancellation Rate (%)')
    ax.set_title(f'Cancellation Rate vs Tolerance Parameters (BEST dt = {best_dt}ms)')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    plt.close(fig)

def create_comprehensive_plots(results_df, output_dir):
    """Create comprehensive visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. DT vs Cancellation Rate (averaged across tolerances)
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    
    # Average across all tolerance combinations for each dt
    dt_avg = results_df.groupby('dt_ms')['cancellation_rate'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    ax1.errorbar(dt_avg['dt_ms'], dt_avg['mean'], yerr=dt_avg['std'], 
                capsize=5, capthick=2, marker='o', markersize=8, linewidth=2)
    ax1.fill_between(dt_avg['dt_ms'], dt_avg['min'], dt_avg['max'], alpha=0.3)
    ax1.set_xlabel('dt (ms)')
    ax1.set_ylabel('Cancellation Rate (%)')
    ax1.set_title('Cancellation Rate vs dt (averaged across all tolerance combinations)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(['Mean ± Std', 'Min-Max Range'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dt_vs_cancellation_rate.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    
    # 2. Tolerance heatmaps for best dt values
    best_dt_values = results_df.groupby('dt_ms')['cancellation_rate'].mean().nlargest(3).index
    fig2, axes = plt.subplots(1, len(best_dt_values), figsize=(5*len(best_dt_values), 4))
    if len(best_dt_values) == 1:
        axes = [axes]
    
    for i, dt_val in enumerate(best_dt_values):
        ax = axes[i]
        
        # Average across all windows for this dt
        dt_data = results_df[results_df['dt_ms'] == dt_val]
        pivot_data = dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
        pivot_table = pivot_data.pivot_table(values='cancellation_rate', index='temporal_tolerance', columns='spatial_tolerance')
        
        sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='viridis', ax=ax)
        ax.set_title(f'dt = {dt_val}ms')
        ax.set_xlabel('Spatial Tolerance (pixels)')
        ax.set_ylabel('Temporal Tolerance (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tolerance_heatmaps_best_dt.png"), dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    
    # 3. 3D surface plot for best dt
    best_dt = results_df.groupby('dt_ms')['cancellation_rate'].mean().idxmax()
    best_dt_data = results_df[results_df['dt_ms'] == best_dt]
    
    fig3 = plt.figure(figsize=(12, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    
    # Average across windows
    combined_data = best_dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
    
    spatial_unique = sorted(combined_data['spatial_tolerance'].unique())
    temporal_unique = sorted(combined_data['temporal_tolerance'].unique())
    
    X, Y = np.meshgrid(spatial_unique, temporal_unique)
    Z = np.zeros_like(X)
    
    for i, temp in enumerate(temporal_unique):
        for j, spat in enumerate(spatial_unique):
            mask = (combined_data['spatial_tolerance'] == spat) & (combined_data['temporal_tolerance'] == temp)
            if mask.any():
                Z[i, j] = combined_data[mask]['cancellation_rate'].iloc[0]
    
    surf = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('Spatial Tolerance (pixels)')
    ax3.set_ylabel('Temporal Tolerance (ms)')
    ax3.set_zlabel('Cancellation Rate (%)')
    ax3.set_title(f'Cancellation Rate vs Tolerance Parameters (dt = {best_dt}ms)')
    
    fig3.colorbar(surf, shrink=0.5, aspect=5)
    out_path = os.path.join(output_dir, "tolerance_3d_surface_best_dt.png")
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved 3D surface: {out_path}")
    
    # Save interactive 3D formats
    if SAVE_INTERACTIVE_3D:
        print("\nSaving interactive 3D surface formats...")
        save_interactive_3d_surface(X, Y, Z, best_dt, output_dir)
    
    if INTERACTIVE_3D:
        print("Opening interactive 3D window (you can rotate and zoom)...")
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig3)
    
    print(f"Saved comprehensive plots to {output_dir}")

def find_optimal_parameters(results_df):
    """Find the optimal parameter combinations"""
    print("\n=== Optimal Parameter Analysis ===")
    
    # Find best tolerance combination for each dt value
    print("Best tolerance combinations for each dt value:")
    print("=" * 60)
    
    for dt_val in sorted(results_df['dt_ms'].unique()):
        dt_data = results_df[results_df['dt_ms'] == dt_val]
        
        # Find best combination for this dt (averaged across windows)
        dt_avg = dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean()
        best_tolerance = dt_avg.idxmax()
        best_rate = dt_avg[best_tolerance]
        
        print(f"dt = {int(dt_val):2d}ms: spatial={best_tolerance[0]:.1f}px, "
              f"temporal={best_tolerance[1]:.1f}ms, rate={best_rate:.2f}%")
        
        # Show top 3 tolerance combinations for this dt
        top3_dt = dt_avg.nlargest(3)
        print(f"  Top 3 for dt={int(dt_val)}ms:")
        for i, ((spat, temp), rate) in enumerate(top3_dt.items(), 1):
            print(f"    {i}. spatial={spat:.1f}px, temporal={temp:.1f}ms, rate={rate:.2f}%")
        print()
    
    # Find best overall combination (excluding dt=0)
    best_overall = results_df.loc[results_df['cancellation_rate'].idxmax()]
    
    print("Best overall combination:")
    print(f"  dt: {int(best_overall['dt_ms'])}ms")
    print(f"  Spatial tolerance: {best_overall['spatial_tolerance']:.1f} pixels")
    print(f"  Temporal tolerance: {best_overall['temporal_tolerance']:.1f} ms")
    print(f"  Cancellation rate: {best_overall['cancellation_rate']:.2f}%")
    print(f"  Window: {best_overall['window_start']:.3f}s to {best_overall['window_end']:.3f}s")
    
    # Find best dt (averaged across all tolerances)
    dt_avg = results_df.groupby('dt_ms')['cancellation_rate'].mean()
    best_dt = dt_avg.idxmax()
    print(f"\nBest dt (averaged across all tolerances): {int(best_dt)}ms ({dt_avg[best_dt]:.2f}%)")
    
    # Show dt performance ranking
    print(f"\nDT performance ranking (averaged across all tolerances):")
    dt_ranking = dt_avg.sort_values(ascending=False)
    for i, (dt_val, rate) in enumerate(dt_ranking.items(), 1):
        print(f"  {i}. dt={int(dt_val):2d}ms: {rate:.2f}%")
    
    # Top 10 combinations overall
    print(f"\nTop 10 combinations overall:")
    top10 = results_df.nlargest(10, 'cancellation_rate')
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        print(f"  {i:2d}. dt={int(row['dt_ms']):2d}ms, spatial={row['spatial_tolerance']:.1f}px, "
              f"temporal={row['temporal_tolerance']:.1f}ms, rate={row['cancellation_rate']:.2f}%")
    
    return best_overall

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("=== Comprehensive DT and Tolerance Analysis ===")
    print(f"DT values: {DT_VALUES_MS}")
    print(f"Spatial range: {SPATIAL_TOLERANCE_RANGE[0]} to {SPATIAL_TOLERANCE_RANGE[1]} (step: {SPATIAL_TOLERANCE_RANGE[2]})")
    print(f"Temporal range: {TEMPORAL_TOLERANCE_RANGE[0]} to {TEMPORAL_TOLERANCE_RANGE[1]} (step: {TEMPORAL_TOLERANCE_RANGE[2]})")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Save interactive 3D: {SAVE_INTERACTIVE_3D}")
    print(f"Show 4D plots in real-time: {SHOW_4D_PLOTS_REALTIME}")
    if SAVE_INTERACTIVE_3D:
        print(f"  PyVista available: {HAS_PYVISTA}")
        print(f"  Mayavi available: {HAS_MAYAVI}")
    
    # Run comprehensive analysis
    results = analyze_all_combinations()
    
    if not results:
        print("No results generated. Check if prediction data exists.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_filename = os.path.join(OUTPUT_DIR, "comprehensive_analysis_results.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"\nSaved results: {csv_filename}")
    
    # Create visualizations
    print("\nCreating comprehensive visualizations...")
    create_comprehensive_plots(results_df, OUTPUT_DIR)
    
    # Create individual dt surface plots (save files)
    print("Creating individual dt surface plots...")
    for dt_val in sorted(results_df['dt_ms'].unique()):
        create_3d_surface_for_specific_dt(results_df, dt_val, OUTPUT_DIR, show_interactive=False)
    
    # Find optimal parameters
    best_parameters = find_optimal_parameters(results_df)
    
    # Launch interactive plot viewer
    print("Launching Interactive Plot Viewer...")
    interactive_plot_viewer(results_df, OUTPUT_DIR)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total combinations tested: {len(results_df)}")
    print(f"Average cancellation rate: {results_df['cancellation_rate'].mean():.2f}%")
    print(f"Best cancellation rate: {results_df['cancellation_rate'].max():.2f}%")
    print(f"Worst cancellation rate: {results_df['cancellation_rate'].min():.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed_time:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
