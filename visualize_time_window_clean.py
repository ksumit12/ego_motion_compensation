#!/usr/bin/env python3
"""
Modular Event Cancellation and Visualization System
==================================================

This script provides fully modular functions for event cancellation and visualization.
Core functions can be imported and used independently in any script.

Core Modules:
1. CANCELLATION_CORE - Pure cancellation logic (no visualization)
2. ROI_ANALYSIS_CORE - ROI-based analysis functions  
3. VISUALIZATION_CORE - Pure visualization functions (no cancellation)
4. UTILS_CORE - Utility functions

Usage Examples:
    # Pure cancellation only
    from visualize_time_window_clean import cancel_events_time_aware, calculate_cancellation_stats
    unmatched_real, unmatched_pred, matches = cancel_events_time_aware(real_events, pred_events, dt, tol_ms, spatial_px)
    
    # ROI analysis only
    from visualize_time_window_clean import analyze_roi_cancellation
    roi_stats = analyze_roi_cancellation(real_events, residual_real, cx, cy, radius)
    
    # Pure visualization only  
    from visualize_time_window_clean import create_panel_figure, export_panel_images
    fig = create_panel_figure(combined_events, residual_real, residual_pred, window)
"""

import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better Linux compatibility
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree
import os
from typing import Tuple, Dict, Optional, List

# =============== Configuration ===============
COMBINED_PATH = "./combined_events_with_predictions.npy"
BIN_MS = 5.0
R_PIX = 2.0
POLARITY_MODE = "opposite"
IMG_W, IMG_H = 1280, 720
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264
WINDOWS = [(5.000, 5.010), (8.200, 8.210), (9.000, 9.010)]
OUTPUT_DIR = "./main_results"

# =============== UTILS_CORE ===============

def load_combined(path: str) -> np.ndarray:
    """
    CORE MODULAR FUNCTION: Load combined events data from file.
    
    This function can be used independently to load event data.
    
    Args:
        path: Path to .npy file containing combined events
        
    Returns:
        Array [x, y, polarity, timestamp, flag] sorted by timestamp
        - flag: 0.0 for real events, 1.0 for predicted events
    """
    arr = np.load(path, mmap_mode="r")
    if not np.all(arr[:-1, 3] <= arr[1:, 3]):
        arr = arr[np.argsort(arr[:, 3])]
    print(f"Loaded {len(arr):,} events (real={int(np.sum(arr[:,4]==0.0)):,}, pred={int(np.sum(arr[:,4]==1.0)):,})")
    return arr

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

# =============== CANCELLATION_CORE ===============

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
                            use_vectorized_processing: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
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
        
    OPTIMIZATION STRATEGIES:
        1. Vectorized spatial search per chunk (batch KDTree queries)
        2. Vectorized temporal gate application
        3. Vectorized polarity checking
        4. Memory-efficient chunking
        5. Pre-computed target times
        6. Early termination on empty candidates
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

def cancel_events_time_aware_async(real_events: np.ndarray, 
                                  predicted_events: np.ndarray, 
                                  dt_seconds: float, 
                                  temporal_tolerance_ms: float, 
                                  spatial_tolerance_pixels: float,
                                  polarity_mode: str = "opposite",
                                  chunk_size: int = 10000,
                                  max_workers: int = 4,
                                  verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    ASYNC VERSION: Event-by-event processing with asynchronous chunk processing.
    
    This version processes chunks in parallel while maintaining event-by-event logic
    within each chunk. Provides significant speedup for large datasets.
    
    Args:
        real_events: Array [x, y, polarity, timestamp] of real events
        predicted_events: Array [x, y, polarity, timestamp] of predicted events  
        dt_seconds: Time step (seconds) - expected time difference
        temporal_tolerance_ms: Temporal tolerance (milliseconds)
        spatial_tolerance_pixels: Spatial tolerance (pixels)
        polarity_mode: "opposite", "equal", or "ignore"
        chunk_size: Processing chunk size for memory efficiency (smaller for async)
        max_workers: Maximum number of parallel workers
        verbose: Whether to print progress updates
        
    Returns:
        Tuple of (unmatched_real_mask, unmatched_predicted_mask, total_matches)
    """
    import concurrent.futures
    from functools import partial
    
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
    
    # Process in smaller chunks for async processing
    chunk_size = min(chunk_size, num_real)
    
    if verbose:
        print(f"  Processing {num_real:,} real events asynchronously in chunks of {chunk_size:,}")
        print(f"  Using {max_workers} parallel workers")
    
    # Create chunk processing function
    process_chunk_func = partial(
        _process_chunk_vectorized,
        predicted_events=predicted_events,
        pred_tree=pred_tree,
        temporal_tolerance_s=temporal_tolerance_s,
        spatial_tolerance_pixels=spatial_tolerance_pixels,
        polarity_mode=polarity_mode,
        matched_real=matched_real,
        matched_predicted=matched_predicted
    )
    
    # Process chunks asynchronously
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for chunk_start in range(0, num_real, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_real)
            chunk_real = real_events[chunk_start:chunk_end]
            chunk_target_times = chunk_real[:, 3] + dt_seconds
            
            future = executor.submit(
                process_chunk_func,
                chunk_real, chunk_target_times, chunk_start
            )
            futures.append(future)
        
        # Collect results
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            chunk_matches = future.result()
            total_matches += chunk_matches
            
            if verbose:
                progress = ((i + 1) / len(futures)) * 100
                print(f"  Progress: {progress:.1f}% - {total_matches:,} matches found", end="\r")
    
    if verbose:
        print()  # New line after progress
    
    # Return unmatched events (inverse of matched)
    return ~matched_real, ~matched_predicted, total_matches

def calculate_cancellation_stats(real_events: np.ndarray, 
                                predicted_events: np.ndarray, 
                                unmatched_real_mask: np.ndarray, 
                                unmatched_predicted_mask: np.ndarray,
                                total_matches: int) -> Dict[str, float]:
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

# =============== ROI_ANALYSIS_CORE ===============

def analyze_roi_cancellation(real_events: np.ndarray, 
                           residual_real_events: np.ndarray,
                           center_x: float, 
                           center_y: float, 
                           radius: float,
                           scale: float = 1.05) -> Dict[str, float]:
    """
    CORE MODULAR FUNCTION: Analyze cancellation rates inside and outside ROI region.
    
    This function can be used independently to analyze ROI-based cancellation.
    Works with any dataset and ROI parameters.
    
    Args:
        real_events: Array [x, y, polarity, timestamp] of original real events
        residual_real_events: Array [x, y, polarity, timestamp] of unmatched real events
        center_x: ROI center X coordinate
        center_y: ROI center Y coordinate  
        radius: ROI radius
        scale: Scale factor for ROI boundary (default 1.05)
        
    Returns:
        Dictionary with ROI analysis results:
        - total_real_inside: Total real events inside ROI
        - total_real_outside: Total real events outside ROI
        - residual_real_inside: Unmatched real events inside ROI
        - residual_real_outside: Unmatched real events outside ROI
        - cancellation_rate_inside: Cancellation rate inside ROI (%)
        - cancellation_rate_outside: Cancellation rate outside ROI (%)
        - events_per_pixel_inside: Events per pixel inside ROI
        - events_per_pixel_outside: Events per pixel outside ROI
        - roi_area_pixels: Total pixels in ROI
        - outside_area_pixels: Total pixels outside ROI
    """
    # Create ROI mask
    def circle_mask(x, y, cx, cy, r, scale=1.05):
        return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

    # Analyze original real events
    inside_mask_original = circle_mask(real_events[:, 0], real_events[:, 1], center_x, center_y, radius, scale)
    outside_mask_original = ~inside_mask_original
    
    total_real_inside = np.sum(inside_mask_original)
    total_real_outside = np.sum(outside_mask_original)
    
    # Analyze residual real events
    if len(residual_real_events) > 0:
        inside_mask_residual = circle_mask(residual_real_events[:, 0], residual_real_events[:, 1], center_x, center_y, radius, scale)
        outside_mask_residual = ~inside_mask_residual
        
        residual_real_inside = np.sum(inside_mask_residual)
        residual_real_outside = np.sum(outside_mask_residual)
    else:
        residual_real_inside = 0
        residual_real_outside = 0
    
    # Calculate cancellation rates
    cancellation_rate_inside = ((total_real_inside - residual_real_inside) / total_real_inside * 100) if total_real_inside > 0 else 0
    cancellation_rate_outside = ((total_real_outside - residual_real_outside) / total_real_outside * 100) if total_real_outside > 0 else 0
    
    # Calculate pixel areas (approximate)
    roi_area_pixels = int(np.pi * (radius * scale) ** 2)
    # Assuming image size - this would need to be passed as parameter for exact calculation
    outside_area_pixels = 1280 * 720 - roi_area_pixels  # Default image size
    
    # Events per pixel
    events_per_pixel_inside = residual_real_inside / roi_area_pixels if roi_area_pixels > 0 else 0
    events_per_pixel_outside = residual_real_outside / outside_area_pixels if outside_area_pixels > 0 else 0
    
    return {
        'total_real_inside': total_real_inside,
        'total_real_outside': total_real_outside,
        'residual_real_inside': residual_real_inside,
        'residual_real_outside': residual_real_outside,
        'cancellation_rate_inside': cancellation_rate_inside,
        'cancellation_rate_outside': cancellation_rate_outside,
        'events_per_pixel_inside': events_per_pixel_inside,
        'events_per_pixel_outside': events_per_pixel_outside,
        'roi_area_pixels': roi_area_pixels,
        'outside_area_pixels': outside_area_pixels,
        'roi_center_x': center_x,
        'roi_center_y': center_y,
        'roi_radius': radius
    }

# =============== VISUALIZATION_CORE ===============

def create_panel_figure(combined_events: np.ndarray, 
                        residual_real_events: np.ndarray, 
                        residual_predicted_events: np.ndarray,
                        window: Tuple[float, float],
                        img_w: int = 1280, 
                        img_h: int = 720, 
                        use_gray: bool = False,
                        disc_center_x: Optional[float] = None,
                        disc_center_y: Optional[float] = None,
                        disc_radius: Optional[float] = None) -> plt.Figure:
    """
    CORE MODULAR FUNCTION: Create comprehensive analysis panel for a time window.
    
    This function can be used independently to visualize cancellation results.
    
    Args:
        combined_events: Array [x, y, polarity, timestamp, flag] of all events
        residual_real_events: Array of unmatched real events
        residual_predicted_events: Array of unmatched predicted events
        window: Tuple (start_time, end_time) for analysis window
        img_w: Image width (default 1280)
        img_h: Image height (default 720)
        use_gray: Whether to use gray colormap instead of seismic
        disc_center_x: Optional disc center X for overlay
        disc_center_y: Optional disc center Y for overlay
        disc_radius: Optional disc radius for overlay
        
    Returns:
        matplotlib Figure object with 3x3 panel layout
    """
    t0, t1 = window

    # Extract events for this window
    allm = (combined_events[:,3] >= t0) & (combined_events[:,3] < t1)
    w_all = combined_events[allm]
    w_real = w_all[w_all[:,4] == 0.0]
    w_pred = w_all[w_all[:,4] == 1.0]
    wr = residual_real_events[(residual_real_events[:,3] >= t0) & (residual_real_events[:,3] < t1)]
    wp = residual_predicted_events[(residual_predicted_events[:,3] >= t0) & (residual_predicted_events[:,3] < t1)]
    cancelled = len(w_real) - len(wr)

    # Build per-pixel images
    img_r, img_p, img_c, nr, npred = build_window_images(combined_events, window, img_w, img_h)
    img_r_n, img_p_n, img_c_n, max_abs = normalize_images_for_display(img_r, img_p, img_c)

    # Create figure
    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[1.1, 1.1, 1.0], figure=fig)

    # Row 1: Scatter plots
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    
    ax0.scatter(w_real[:,0], w_real[:,1], s=2, alpha=0.6, c="tab:blue")
    ax0.set_title(f"Real ({len(w_real):,})")
    
    ax1.scatter(w_pred[:,0], w_pred[:,1], s=2, alpha=0.6, c="tab:red")
    ax1.set_title(f"Predicted ({len(w_pred):,})")
    
    ax2.scatter(wr[:,0], wr[:,1], s=2, alpha=0.7, c="tab:blue", label=f"Real ({len(wr):,})")
    ax2.scatter(wp[:,0], wp[:,1], s=2, alpha=0.7, c="tab:red", label=f"Pred ({len(wp):,})")
    ax2.legend()
    
    for ax in (ax0, ax1, ax2):
        ax.set_xlim(0, img_w)
        ax.set_ylim(0, img_h)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x")
    ax0.set_ylabel("y")

    # Row 2: Per-pixel images
    cmap = "gray" if use_gray else "seismic"
    b0 = fig.add_subplot(gs[1, 0])
    b1 = fig.add_subplot(gs[1, 1])
    b2 = fig.add_subplot(gs[1, 2])
    
    im0 = b0.imshow(img_r_n, cmap=cmap, origin="upper", vmin=0, vmax=1)
    b0.set_title(f"Real (N={nr:,})")
    b0.set_xlabel("x")
    b0.set_ylabel("y")
    b0.grid(alpha=0.2)
    
    im1 = b1.imshow(img_p_n, cmap=cmap, origin="upper", vmin=0, vmax=1)
    b1.set_title(f"Pred (N={npred:,})")
    b1.set_xlabel("x")
    b1.set_ylabel("y")
    b1.grid(alpha=0.2)
    
    im2 = b2.imshow(img_c_n, cmap=cmap, origin="upper", vmin=0, vmax=1)
    b2.set_title("Combined")
    b2.set_xlabel("x")
    b2.set_ylabel("y")
    b2.grid(alpha=0.2)
    
    # Add disc overlay if provided
    if cmap == "seismic" and disc_center_x is not None and disc_center_y is not None and disc_radius is not None:
        disc_circle = plt.Circle((disc_center_x, disc_center_y), disc_radius, 
                               fill=True, color='yellow', alpha=0.15, linewidth=0)
        b2.add_patch(disc_circle)
        
        disc_outline = plt.Circle((disc_center_x, disc_center_y), disc_radius, 
                                fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
        b2.add_patch(disc_outline)
        
        b2.plot(disc_center_x, disc_center_y, 'yo', markersize=6, markeredgecolor='black', 
               markeredgewidth=1, label=f'Disc Center')
        b2.legend(loc='upper right', fontsize=7)
    
    for ax_im, im in zip((b0, b1, b2), (im0, im1, im2)):
        cb = fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.04)
        cb.set_label("signed count (Σ polarity)")
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels([f"-{max_abs}", "0", f"+{max_abs}"])

    # Row 3: Histograms
    c0 = fig.add_subplot(gs[2, 0])
    c1 = fig.add_subplot(gs[2, 1])
    c2 = fig.add_subplot(gs[2, 2])
    
    def plot_histogram(ax, data, title):
        ax.hist(data, bins=100, log=True, edgecolor="k", alpha=0.85)
        if data.size:
            med = float(np.median(data))
            p95 = float(np.percentile(data, 95))
            ax.axvline(med, color="tab:orange", ls="--", lw=1.2, label=f"median={med:.1f}")
            ax.axvline(p95, color="tab:green", ls="--", lw=1.0, label=f"95%={p95:.1f}")
        ax.set_xlabel("|signed count|")
        ax.set_ylabel("pixels (log)")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title(title)
    
    m_r = np.abs(img_r.ravel())
    m_p = np.abs(img_p.ravel())
    m_c = np.abs(img_c.ravel())
    nz_r = m_r[m_r > 0]
    nz_p = m_p[m_p > 0]
    nz_c = m_c[m_c > 0]

    plot_histogram(c0, nz_r, f"Real |count| (nonzero px={nz_r.size:,})")
    plot_histogram(c1, nz_p, f"Pred |count| (nonzero px={nz_p.size:,})")
    plot_histogram(c2, nz_c, f"Combined |count| (nonzero px={nz_c.size:,})")

    # Calculate statistics
    total_real_events = len(w_real)
    actual_cancellation_rate = (cancelled / total_real_events * 100) if total_real_events > 0 else 0
    
    fig.suptitle(
        f"Time window {t0:.3f}–{t1:.3f}s • cancel={cancelled:,} "
        f"({actual_cancellation_rate:.1f}%) • "
        f"spatial_tol=2.0px, temporal_tol=5.0ms • "
        f"{'gray' if use_gray else 'seismic'} images",
        y=0.995, fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def export_panel_images(combined_events: np.ndarray, 
                       residual_real_events: np.ndarray, 
                       residual_predicted_events: np.ndarray,
                       windows: List[Tuple[float, float]], 
                       output_dir: str, 
                       use_gray: bool = False,
                       disc_center_x: Optional[float] = None,
                       disc_center_y: Optional[float] = None,
                       disc_radius: Optional[float] = None) -> None:
    """
    CORE MODULAR FUNCTION: Export individual panel images for each time window.
    
    This function can be used independently to save visualization results.
    
    Args:
        combined_events: Array of all events
        residual_real_events: Array of unmatched real events
        residual_predicted_events: Array of unmatched predicted events
        windows: List of (start_time, end_time) tuples
        output_dir: Directory to save images
        use_gray: Whether to use gray colormap
        disc_center_x: Optional disc center X for overlay
        disc_center_y: Optional disc center Y for overlay
        disc_radius: Optional disc radius for overlay
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, w in enumerate(windows):
        fig = create_panel_figure(combined_events, residual_real_events, residual_predicted_events, w, 
                                 use_gray=use_gray, disc_center_x=disc_center_x, 
                                 disc_center_y=disc_center_y, disc_radius=disc_radius)
        suffix = "_gray" if use_gray else "_seismic"
        filename = f"ego_panel_{i+1}_{w[0]:.3f}s_to_{w[1]:.3f}s{suffix}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved panel image: {filename}")

# =============== LEGACY WRAPPER FUNCTIONS ===============

def run_cancellation(combined_events: np.ndarray, 
                    temporal_tolerance_ms: float, 
                    spatial_tolerance_pixels: float,
                    polarity_mode: str = "opposite",
                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    LEGACY WRAPPER: Run ego-motion cancellation using true temporal gate.
    
    This is a wrapper function that uses the modular core functions.
    Maintains backward compatibility with the original interface.
    
    Args:
        combined_events: Array [x, y, polarity, timestamp, flag] of all events
        temporal_tolerance_ms: Temporal tolerance (milliseconds)
        spatial_tolerance_pixels: Spatial tolerance (pixels)
        polarity_mode: "opposite", "equal", or "ignore"
        verbose: Whether to print progress updates
        
    Returns:
        Tuple of (residual_real_events, residual_predicted_events)
    """
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    
    total_real_events = len(real_events)
    total_predicted_events = len(pred_events)
    
    if verbose:
        print(f"Start cancellation: temporal_tol={temporal_tolerance_ms} ms, spatial_tol={spatial_tolerance_pixels} px")
        print(f"Events: {total_real_events:,} real, {total_predicted_events:,} predicted")
    
    start_time = time.time()
    
    # Estimate dt from data using modular function
    dt_seconds = estimate_dt_from_data(real_events, pred_events)
    
    if verbose:
        print(f"Estimated dt: {dt_seconds*1000:.1f}ms")
    
    # Run cancellation using modular function
    unmatched_real_mask, unmatched_predicted_mask, total_matched_pairs = cancel_events_time_aware(
        real_events, pred_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels,
        polarity_mode=polarity_mode, verbose=verbose
    )
    
    residual_real_events = real_events[unmatched_real_mask]
    residual_predicted_events = pred_events[unmatched_predicted_mask]
    
    elapsed_time = time.time() - start_time
    
    # Calculate rates using modular function
    stats = calculate_cancellation_stats(real_events, pred_events, unmatched_real_mask, unmatched_predicted_mask, total_matched_pairs)
    
    if verbose:
        print(f"Done in {elapsed_time:.1f}s")
        print(f"Real: {total_real_events:,} -> residual {len(residual_real_events):,} (cancelled {stats['real_cancelled']:,}, {stats['real_cancellation_rate']:.1f}%)")
        print(f"Pred: {total_predicted_events:,} -> residual {len(residual_predicted_events):,} (cancelled {stats['predicted_cancelled']:,}, {stats['predicted_cancellation_rate']:.1f}%)")
        print(f"Matched pairs: {total_matched_pairs:,}")
        print(f"Using TRUE temporal gate: |t_j-(t_i+dt)| <= {temporal_tolerance_ms}ms")
    
    return residual_real_events, residual_predicted_events

# =============== UTILITY FUNCTIONS ===============

def circle_mask(x: np.ndarray, y: np.ndarray, cx: float, cy: float, r: float, scale: float = 1.05) -> np.ndarray:
    """Create boolean mask for points inside scaled circle"""
    return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

def events_in_window(events: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Create boolean mask for events within time window"""
    return (events[:, 3] >= t0) & (events[:, 3] < t1)

def make_frame(H: int, W: int, xs: np.ndarray, ys: np.ndarray, ps: Optional[np.ndarray] = None) -> np.ndarray:
    """Create pixel-true frame from event coordinates and polarities"""
    frame = np.zeros((H, W), dtype=np.int32)
    if len(xs) == 0:
        return frame
    
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs_valid = xs[valid]
    ys_valid = ys[valid]
    
    if ps is not None:
        ps_valid = ps[valid]
        np.add.at(frame, (ys_valid, xs_valid), ps_valid)
    else:
        np.add.at(frame, (ys_valid, xs_valid), 1)
    
    return frame

def convert_polarity_to_signed(polarity_values: np.ndarray) -> np.ndarray:
    """Convert polarity from 0/1 to -1/+1 for visualization"""
    return np.where(polarity_values > 0.5, 1, -1).astype(np.int16)

def create_per_pixel_count_image(width: int, height: int, events: np.ndarray) -> np.ndarray:
    """Create per-pixel signed count image using bilinear interpolation"""
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    
    x = events[:, 0].astype(np.float32)
    y = events[:, 1].astype(np.float32)
    s = convert_polarity_to_signed(events[:, 2]).astype(np.float32)
    
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

def build_window_images(combined_events: np.ndarray, time_window: Tuple[float, float], image_width: int, image_height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Create per-pixel count images for real, predicted, and combined events"""
    start_time, end_time = time_window
    
    time_mask = (combined_events[:, 3] >= start_time) & (combined_events[:, 3] < end_time)
    window_events = combined_events[time_mask]
    
    real_events = window_events[window_events[:, 4] == 0.0][:, :3]
    predicted_events = window_events[window_events[:, 4] == 1.0][:, :3]
    all_events = window_events[:, :3]
    
    real_image = create_per_pixel_count_image(image_width, image_height, real_events)
    predicted_image = create_per_pixel_count_image(image_width, image_height, predicted_events)
    combined_image = create_per_pixel_count_image(image_width, image_height, all_events)
    
    return real_image, predicted_image, combined_image, len(real_events), len(predicted_events)

def normalize_images_for_display(real_image: np.ndarray, predicted_image: np.ndarray, combined_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Normalize three images to the same scale for fair comparison"""
    real_max = np.abs(real_image).max()
    predicted_max = np.abs(predicted_image).max()
    combined_max = np.abs(combined_image).max()
    overall_max = max(real_max, predicted_max, combined_max, 1)
    
    def normalize_single_image(image):
        normalized = (image / (2.0 * overall_max)) + 0.5
        return np.clip(normalized, 0.0, 1.0)
    
    normalized_real = normalize_single_image(real_image)
    normalized_predicted = normalize_single_image(predicted_image)
    normalized_combined = normalize_single_image(combined_image)
    
    return normalized_real, normalized_predicted, normalized_combined, int(overall_max)

# =============== MAIN EXECUTION ===============

def main():
    """
    Main execution function demonstrating modular usage.
    
    This function shows how to use the modular core functions independently.
    You can remove visualization parts to get pure cancellation-only functionality.
    """
    print("Loading combined events data...")
    combined = load_combined(COMBINED_PATH)
    
    # For testing, work with a manageable subset
    if len(combined) > 2_000_000:
        print(f"Large dataset detected ({len(combined):,} events). Using subset for testing...")
        
        # Find subset that overlaps with analysis windows
        target_start = 5.000
        target_end = 5.010
        
        time_mask = (combined[:, 3] >= target_start) & (combined[:, 3] < target_end + 0.1)
        target_events = combined[time_mask]
        
        if len(target_events) > 0:
            subset_size = min(2_000_000, len(target_events))
            combined = target_events[:subset_size]
            print(f"Using subset from target window: {len(combined):,} events")
        else:
            subset_size = 2_000_000
            combined = combined[:subset_size]
            print(f"Using first {subset_size:,} events (no target window found)")
        
        print(f"Real events: {int(np.sum(combined[:,4]==0.0)):,}")
        print(f"Pred events: {int(np.sum(combined[:,4]==1.0)):,}")
        print(f"Time range: {combined[0,3]:.3f}s to {combined[-1,3]:.3f}s")
        
        # Adjust analysis windows to match subset
        subset_start = combined[0,3]
        subset_end = combined[-1,3]
        subset_duration = subset_end - subset_start
        
        global WINDOWS
        WINDOWS = [
            (subset_start, subset_start + subset_duration/3),
            (subset_start + subset_duration/3, subset_start + 2*subset_duration/3),
            (subset_start + 2*subset_duration/3, subset_end)
        ]
        print(f"Adjusted analysis windows to match subset:")
        for i, w in enumerate(WINDOWS):
            print(f"  Window {i+1}: {w[0]:.3f}s to {w[1]:.3f}s")

    print("Running ego-motion cancellation...")
    print("Using TRUE temporal gate method (corrected)")
    
    # CORE MODULAR USAGE: Choose processing mode
    processing_mode = "vectorized"  # Options: "sequential", "vectorized", "async"
    
    if processing_mode == "async":
        print("Using ASYNC event-by-event processing...")
        # Split events for async processing
        real_events = combined[combined[:, 4] == 0.0]
        pred_events = combined[combined[:, 4] == 1.0]
        
        # Estimate dt
        dt_seconds = estimate_dt_from_data(real_events, pred_events)
        print(f"Estimated dt: {dt_seconds*1000:.1f}ms")
        
        # Run async cancellation
        unmatched_real_mask, unmatched_predicted_mask, total_matches = cancel_events_time_aware_async(
            real_events, pred_events, dt_seconds, BIN_MS, R_PIX,
            chunk_size=10000, max_workers=4, verbose=True
        )
        
        residual_real_events = real_events[unmatched_real_mask]
        residual_predicted_events = pred_events[unmatched_predicted_mask]
        
    elif processing_mode == "vectorized":
        print("Using VECTORIZED event-by-event processing...")
        residual_real_events, residual_predicted_events = run_cancellation(combined, BIN_MS, R_PIX)

    else:  # sequential
        print("Using SEQUENTIAL event-by-event processing...")
        # Split events for sequential processing
        real_events = combined[combined[:, 4] == 0.0]
        pred_events = combined[combined[:, 4] == 1.0]
        
        # Estimate dt
        dt_seconds = estimate_dt_from_data(real_events, pred_events)
        print(f"Estimated dt: {dt_seconds*1000:.1f}ms")
        
        # Run sequential cancellation
        unmatched_real_mask, unmatched_predicted_mask, total_matches = cancel_events_time_aware(
            real_events, pred_events, dt_seconds, BIN_MS, R_PIX,
            use_vectorized_processing=False, verbose=True
        )
        
        residual_real_events = real_events[unmatched_real_mask]
        residual_predicted_events = pred_events[unmatched_predicted_mask]

    # CORE MODULAR USAGE: ROI analysis
    print("\nAnalyzing ROI cancellation rates...")
    roi_stats = analyze_roi_cancellation(
        combined[combined[:, 4] == 0.0],  # Real events
        residual_real_events,
        DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS
    )
    
    print(f"[INSIDE ROI] real={roi_stats['total_real_inside']}, residual={roi_stats['residual_real_inside']}, "
          f"cancellation_rate={roi_stats['cancellation_rate_inside']:.2f}%, events_per_pixel={roi_stats['events_per_pixel_inside']:.4f}")
    print(f"[OUTSIDE ROI] real={roi_stats['total_real_outside']}, residual={roi_stats['residual_real_outside']}, "
          f"cancellation_rate={roi_stats['cancellation_rate_outside']:.2f}%, events_per_pixel={roi_stats['events_per_pixel_outside']:.4f}")

    # CORE MODULAR USAGE: Visualization (optional - can be removed for pure cancellation)
    print("Exporting analysis panels...")
    export_panel_images(combined, residual_real_events, residual_predicted_events, WINDOWS,
                        OUTPUT_DIR, use_gray=False, disc_center_x=DISC_CENTER_X, 
                        disc_center_y=DISC_CENTER_Y, disc_radius=DISC_RADIUS)
    export_panel_images(combined, residual_real_events, residual_predicted_events, WINDOWS,
                        OUTPUT_DIR, use_gray=True, disc_center_x=DISC_CENTER_X, 
                        disc_center_y=DISC_CENTER_Y, disc_radius=DISC_RADIUS)

    print(f"\nAnalysis complete!")
    print(f"Time windows analyzed: {len(WINDOWS)}")
    print(f"Cancellation parameters: {BIN_MS}ms temporal, {R_PIX}px spatial")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI analysis: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")

    # Return results for further analysis
    return {
        'combined_events': combined,
        'residual_real_events': residual_real_events,
        'residual_predicted_events': residual_predicted_events,
        'roi_stats': roi_stats,
        'windows': WINDOWS
    }

if __name__ == "__main__":
    main()
