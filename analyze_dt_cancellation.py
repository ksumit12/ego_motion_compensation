#!/usr/bin/env python3
"""
Analyze cancellation rates for different dt values within circular ROI.
This script loads the window predictions and calculates cancellation rates
for each dt value, focusing on the circular region of interest.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm

# =============== Configuration ===============
WINDOW_PREDICTIONS_DIR = "/media/sumit/New Volume1/window_predictions_5s_fine"

# Cancellation parameters (same as visualize_time_window.py)
BIN_MS = 5.0          # Temporal tolerance (ms)
R_PIX = 2.0           # Spatial tolerance (pixels)
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# Motion parameters for pixel displacement calculation (from actual data)
OMEGA_RAD_S = 3.612  # Actual mean angular velocity from tracker data (rad/s)
DISC_RADIUS_PX = 264  # ROI radius in pixels

# Disc center coordinates (from actual data)
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 250

# Time windows - Updated for actual data structure
WINDOWS = [
    (5.000, 10.000),  # Single 5-second window
]

# DT values to analyze - FINER RESOLUTION as Angus suggested
DT_RANGE_MS = (0, 3)  # Focus on 0-3ms range with fine resolution
DT_STEP_MS = 0.1      # Will be overridden with Angus's approach

# Output settings
OUTPUT_DIR = "./dt_analysis_results"
PLOT_DPI = 150

# Performance options for large datasets
USE_SUBSET_FOR_TESTING = True   # Process SUBSET for 5-second dataset
SUBSET_SIZE = 50000  # Process only 50k events per dt for fast analysis
USE_MEMORY_MAPPING = True  # Use memory mapping to avoid loading full arrays
CHUNK_SIZE = 10000  # Process events in smaller chunks for memory efficiency

def load_npy_or_npz(path, key=None, use_mmap=True, subset_size=None):
    """Load .npy or .npz with memory mapping and optional subsetting."""
    if path.endswith('.npz'):
        with np.load(path) as z:
            if key is not None and key in z:
                arr = z[key]
            else:
                # fallback to first array in archive
                first_key = list(z.files)[0]
                arr = z[first_key]
    else:
        if use_mmap:
            arr = np.load(path, mmap_mode='r')
        else:
            arr = np.load(path)
    
    # Apply subset if requested
    if subset_size is not None and len(arr) > subset_size:
        # Take a representative subset (first N events)
        arr = arr[:subset_size]
    
    return arr

def combine_real_pred(real_events, pred_events):
    """Combine 4-col real and pred into 5-col with flags and sort by time."""
    if len(real_events) == 0 and len(pred_events) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    real_flag = np.zeros((len(real_events), 1), dtype=np.float32)
    pred_flag = np.ones((len(pred_events), 1), dtype=np.float32)
    stacks = []
    if len(real_events) > 0:
        stacks.append(np.column_stack([real_events, real_flag]))
    if len(pred_events) > 0:
        stacks.append(np.column_stack([pred_events, pred_flag]))
    combined = np.vstack(stacks)
    return combined[np.argsort(combined[:, 3])]

def discover_window_dirs(base_dir):
    """Return list of (path, label) for window_* subdirs, sorted by name."""
    try:
        entries = sorted([p for p in Path(base_dir).iterdir() if p.is_dir() and p.name.startswith('window_')], key=lambda p: p.name)
        return [(str(p), p.name) for p in entries]
    except FileNotFoundError:
        return []

def try_load_combined_or_split(window_dir, dt_ms):
    """Try loading combined_events for given dt. If not present, load split real/pred."""
    subset_size = SUBSET_SIZE if USE_SUBSET_FOR_TESTING else None
    
    # Try combined .npy or .npz
    combined_candidates = [
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:04.1f}ms.npy"),
        os.path.join(window_dir, f"combined_events_dt_{dt_ms:04.1f}ms.npz"),
    ]
    for path in combined_candidates:
        if os.path.exists(path):
            arr = load_npy_or_npz(path, key='combined', use_mmap=USE_MEMORY_MAPPING, subset_size=subset_size)
            return arr
    # Try split
    real_candidates = [
        os.path.join(window_dir, 'real_events.npy'),
        os.path.join(window_dir, 'real_events.npz'),
    ]
    pred_candidates = [
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:04.1f}ms.npy"),
        os.path.join(window_dir, f"pred_events_dt_{dt_ms:04.1f}ms.npz"),
    ]
    real = None
    for path in real_candidates:
        if os.path.exists(path):
            real = load_npy_or_npz(path, key='real', use_mmap=USE_MEMORY_MAPPING, subset_size=subset_size)
            break
    if real is None:
        raise FileNotFoundError(f"Missing real_events in {window_dir}")
    pred = None
    for path in pred_candidates:
        if os.path.exists(path):
            pred = load_npy_or_npz(path, key='pred', use_mmap=USE_MEMORY_MAPPING, subset_size=subset_size)
            break
    if pred is None:
        raise FileNotFoundError(f"Missing predictions for dt={dt_ms}ms in {window_dir}")
    return combine_real_pred(real, pred)

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

def cancel_events_time_aware(real_events, predicted_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Match real and predicted events using TRUE temporal gate: |t_j-(t_i+Δt)|≤ε_t
    
    This implements the correct mathematical formulation from the thesis.
    HIGHLY OPTIMIZED version using spatial indexing for large datasets.
    """
    num_real = len(real_events)
    num_predicted = len(predicted_events)
    
    # If either list is empty, nothing can be matched
    if num_real == 0 or num_predicted == 0:
        return np.ones(num_real, bool), np.ones(num_predicted, bool), 0
    
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    matched_real = np.zeros(num_real, dtype=bool)
    matched_predicted = np.zeros(num_predicted, dtype=bool)
    total_matches = 0
    
    # Create spatial KDTree for predicted events (much faster spatial search)
    pred_tree = cKDTree(predicted_events[:, :2])  # Only x, y coordinates
    
    # Process real events in larger chunks for efficiency
    chunk_size = min(CHUNK_SIZE, num_real)  # Process in chunks of 1M events
    
    # Create progress bar for cancellation processing
    pbar = tqdm(total=num_real, desc="  Cancelling events", unit="events", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for chunk_start in range(0, num_real, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_real)
        chunk_real = real_events[chunk_start:chunk_end]
        
        # Vectorized processing of the entire chunk
        chunk_target_times = chunk_real[:, 3] + dt_seconds  # t_i + Δt for all events in chunk
        
        # For each real event in chunk, find spatial candidates first (much faster)
        # Query KDTree for spatial candidates within tolerance
        spatial_candidates_list = pred_tree.query_ball_point(chunk_real[:, :2], spatial_tolerance_pixels)
        
        # Process each event in the chunk
        for i, (real_event, target_time, spatial_candidates) in enumerate(zip(chunk_real, chunk_target_times, spatial_candidates_list)):
            real_idx = chunk_start + i
            if matched_real[real_idx]:
                pbar.update(1)
                continue
            
            if len(spatial_candidates) == 0:
                pbar.update(1)
                continue
            
            # Convert to numpy array and filter out already matched
            spatial_candidates = np.array(spatial_candidates)
            available_candidates = spatial_candidates[~matched_predicted[spatial_candidates]]
            
            if len(available_candidates) == 0:
                pbar.update(1)
                continue
            
            # Among spatial candidates, find temporal candidates
            candidate_times = predicted_events[available_candidates, 3]
            temporal_mask = np.abs(candidate_times - target_time) <= temporal_tolerance_s
            
            if not np.any(temporal_mask):
                pbar.update(1)
                continue
            
            # Get final candidates (both spatial and temporal)
            final_candidates = available_candidates[temporal_mask]
            candidate_events = predicted_events[final_candidates]
            
            # Vectorized polarity check
            real_polarity = real_event[2]
            pred_polarities = candidate_events[:, 2]
            
            if POLARITY_MODE == "ignore":
                polarity_matches = np.ones(len(candidate_events), dtype=bool)
            elif POLARITY_MODE == "equal":
                polarity_matches = (pred_polarities == real_polarity)
            else:  # "opposite"
                polarity_matches = (pred_polarities != real_polarity)
            
            if np.any(polarity_matches):
                # Find closest among valid candidates
                valid_candidates = final_candidates[polarity_matches]
                valid_events = candidate_events[polarity_matches]
                
                # Calculate distances to valid candidates
                distances = np.sqrt(np.sum((valid_events[:, :2] - real_event[:2])**2, axis=1))
                best_candidate = valid_candidates[np.argmin(distances)]
                
                # Make the match
                matched_real[real_idx] = True
                matched_predicted[best_candidate] = True
                total_matches += 1
            
            pbar.update(1)
    
    pbar.close()
    
    # Return unmatched events (inverse of matched)
    unmatched_real = ~matched_real
    unmatched_predicted = ~matched_predicted
    
    return unmatched_real, unmatched_predicted, total_matches


def run_cancellation_for_window(combined_events, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Run ego-motion cancellation using TRUE temporal gate: |t_j-(t_i+Δt)|≤ε_t
    
    This implements the correct mathematical formulation from the thesis.
    No binning - direct temporal relationship per event.
    """
    # Split events
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    
    total_real_events = len(real_events)
    total_predicted_events = len(pred_events)
    
    if total_real_events == 0 or total_predicted_events == 0:
        return np.zeros((0, 5), dtype=combined_events.dtype), np.zeros((0, 5), dtype=combined_events.dtype), 0
    
    # Use the correct temporal gate approach
    # We need dt_seconds - estimate from the data
    if len(real_events) > 0 and len(pred_events) > 0:
        # Estimate dt from the time difference between real and predicted events
        # This is a heuristic - in practice dt should be known from the prediction process
        sample_real_times = real_events[:min(1000, len(real_events)), 3]
        sample_pred_times = pred_events[:min(1000, len(pred_events)), 3]
        
        # Find the most common time difference (this should be dt)
        time_diffs = []
        for rt in sample_real_times:
            closest_pred_times = sample_pred_times[np.abs(sample_pred_times - rt) < 0.1]  # Within 100ms
            if len(closest_pred_times) > 0:
                closest_diff = np.min(closest_pred_times - rt)
                if closest_diff > 0:  # Predicted events should be later
                    time_diffs.append(closest_diff)
        
        if len(time_diffs) > 0:
            dt_seconds = np.median(time_diffs)  # Use median as robust estimate
        else:
            dt_seconds = 0.002  # Default 2ms if estimation fails
    else:
        dt_seconds = 0.002  # Default 2ms
    
    # Run time-aware cancellation
    unmatched_real_mask, unmatched_predicted_mask, total_matched_pairs = cancel_events_time_aware(
        real_events, pred_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels
    )
    
    # Get residual events
    residual_real_events = real_events[unmatched_real_mask]
    residual_predicted_events = pred_events[unmatched_predicted_mask]
    
    return residual_real_events, residual_predicted_events, total_matched_pairs

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

def analyze_window_dt_cancellation_dir(window_idx, window_dir, window_label, dt_values_ms):
    """Analyze cancellation rates for all dt values for a given window directory"""
    print(f"\nAnalyzing window {window_idx + 1}: {window_label}")
    
    if not os.path.exists(window_dir):
        print(f"  Warning: Directory {window_dir} not found, skipping...")
        return None
    
    results = []
    
    # Create progress bar for dt analysis
    dt_pbar = tqdm(dt_values_ms, desc=f"Window {window_idx + 1}", unit="dt")
    
    for dt_ms in dt_pbar:
        # Load combined or split (real+pred) for this dt
        dt_pbar.set_postfix_str(f"dt={dt_ms}ms")
        try:
            combined_events = try_load_combined_or_split(window_dir, dt_ms)
            if USE_SUBSET_FOR_TESTING and len(combined_events) > SUBSET_SIZE:
                print(f"\n    Using subset ({SUBSET_SIZE:,}/{len(combined_events):,}) events")
            else:
                print(f"\n    Loaded {len(combined_events):,} events")
        except FileNotFoundError as e:
            print(f"\n    Skip: {e}")
            continue
        
        # Run cancellation
        residual_real, residual_pred, matched_pairs = run_cancellation_for_window(
            combined_events, BIN_MS, R_PIX
        )
        
        # Calculate ROI cancellation rate
        roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
            combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
        )
        
        results.append({
            'dt_ms': dt_ms,
            'cancellation_rate': roi_cancellation_rate,
            'total_roi_real': total_roi_real,
            'total_roi_cancelled': total_roi_cancelled,
            'total_matched_pairs': matched_pairs
        })
        
        print(f"ROI cancel rate: {roi_cancellation_rate:.1f}% ({total_roi_cancelled}/{total_roi_real})")
    
    return results

def analyze_window_dt_cancellation(window_idx, window, dt_values_ms):
    """Analyze cancellation rates for all dt values for a specific window"""
    t0, t1 = window
    print(f"\nAnalyzing window {window_idx + 1}: {t0:.3f}s to {t1:.3f}s")
    
    # Load window predictions directory
    window_dir = os.path.join(WINDOW_PREDICTIONS_DIR, f"window_{window_idx + 1}_{t0:.3f}s_to_{t1:.3f}s")
    
    if not os.path.exists(window_dir):
        print(f"  Warning: Directory {window_dir} not found, skipping...")
        return None
    
    results = []
    
    for dt_ms in dt_values_ms:
        # Load combined events for this dt
        filename = f"combined_events_dt_{dt_ms:02d}ms.npy"
        filepath = os.path.join(window_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  Warning: File {filename} not found, skipping dt={dt_ms}ms")
            continue
        
        print(f"  Processing dt={dt_ms:2d}ms...", end=" ")
        
        # Load data
        combined_events = np.load(filepath)
        
        # Run cancellation
        residual_real, residual_pred, matched_pairs = run_cancellation_for_window(
            combined_events, BIN_MS, R_PIX
        )
        
        # Calculate ROI cancellation rate
        roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
            combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
        )
        
        results.append({
            'dt_ms': dt_ms,
            'cancellation_rate': roi_cancellation_rate,
            'total_roi_real': total_roi_real,
            'total_roi_cancelled': total_roi_cancelled,
            'total_matched_pairs': matched_pairs
        })
        
        print(f"ROI cancel rate: {roi_cancellation_rate:.1f}% ({total_roi_cancelled}/{total_roi_real})")
    
    return results

def create_cancellation_plots(all_results, output_dir):
    """Create cancellation plots with Angus's enhancements"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create single plot since we only have one window
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Calculate pixel displacement for overlay (using actual omega values)
    OMEGA_RAD_S = 3.612  # Actual mean angular velocity from tracker data
    MEAN_RADIUS_PX = 199  # Actual mean radius from real events
    
    for window_idx, window_results in enumerate(all_results):
        if window_results is None:
            continue
        
        # Extract data
        dt_values = [r['dt_ms'] for r in window_results]
        cancel_rates = [r['cancellation_rate'] for r in window_results]
        match_rates = [r['total_matched_pairs'] / r['total_roi_real'] * 100 if r['total_roi_real'] > 0 else 0 for r in window_results]
        
        # Plot cancellation rate
        ax.plot(dt_values, cancel_rates, 'b-', linewidth=2, marker='o', markersize=4, label='Cancellation Rate')
        
        # Angus's enhancements
        # 1. Shade ≥90% region lightly
        ax.axhspan(90, 100, alpha=0.1, color='green', label='≥90% Region')
        
        # 2. Find t90 and t80 thresholds
        t90_idx = np.where(np.array(cancel_rates) < 90)[0]
        t80_idx = np.where(np.array(cancel_rates) < 80)[0]
        
        t90_ms = dt_values[t90_idx[0]] if len(t90_idx) > 0 else None
        t80_ms = dt_values[t80_idx[0]] if len(t80_idx) > 0 else None
        
        # Add vertical lines at t90 and t80
        if t90_ms is not None:
            ax.axvline(x=t90_ms, color='orange', linestyle='--', alpha=0.7, label=f'First dt < 90%: {t90_ms:.1f}ms')
        if t80_ms is not None:
            ax.axvline(x=t80_ms, color='red', linestyle='--', alpha=0.7, label=f'First dt < 80%: {t80_ms:.1f}ms')
        
        # 3. Add match rate line (secondary y-axis) with better explanation
        ax2 = ax.twinx()
        ax2.plot(dt_values, match_rates, 'r--', alpha=0.6, linewidth=1, label='Match Rate (events finding any match)')
        ax2.set_ylabel('Match Rate (%)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Set plot limits and labels
        ax.set_xlim(0, 3)
        ax.set_ylim(70, 100)
        ax.set_xlabel('Prediction Horizon Δt (ms)', fontsize=14)
        ax.set_ylabel('Cancellation Rate (%)', fontsize=14)
        ax.set_title('Ego-Motion Cancellation: Early Drop-off Analysis\n(Fine Resolution: 0.1ms steps)', fontsize=16)
        
        # Add pixel displacement secondary x-axis
        ax3 = ax.twiny()
        ax3.set_xlim(0, 3)
        ax3.set_xlabel('Predicted Pixel Displacement (px)', fontsize=12)
        
        # Calculate pixel displacement for the secondary axis
        dt_range = np.linspace(0, 3, 100)
        pixel_range = [dt * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000 for dt in dt_range]
        ax3.plot(pixel_range, [100]*len(pixel_range), alpha=0)  # Invisible line for scaling
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax2.legend(loc='lower right')
        
        # Add text annotations with key findings
        if t90_ms is not None:
            t90_px = t90_ms * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000
            ax.text(0.02, 0.98, f'First dt < 90%: {t90_ms:.1f}ms ≈ {t90_px:.1f}px\n(Match Rate: events finding any nearby match)', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save CSV with additional columns
        pixel_displacement = [dt_ms * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000 for dt_ms in dt_values]
        results_df = pd.DataFrame({
            'dt_ms': dt_values,
            'cancellation_rate': cancel_rates,
            'match_rate': match_rates,
            'pixel_displacement': pixel_displacement
        })
        
        csv_path = os.path.join(output_dir, "dt_cancellation_data_fine.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Saved data: {csv_path}")
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "dt_cancellation_analysis_fine_resolution.png")
    plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved plot: {plot_path}")
    
    plt.close()
    
    return t90_ms, t80_ms

def create_flow_magnitude_image(output_dir):
    """Create flow magnitude map as Angus suggested with improved visualization"""
    OMEGA_RAD_S = 3.612  # Actual mean angular velocity from tracker data
    
    # Create image dimensions (720, 1280) as Angus suggested
    height, width = 720, 1280
    center_x, center_y = DISC_CENTER_X, DISC_CENTER_Y
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Calculate distance from center for each pixel
    distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create flow magnitude maps for different dt values
    dt_values = [1.0, 3.0]  # As Angus suggested
    
    # Calculate all pixel displacements to find global max
    all_displacements = []
    for dt_ms in dt_values:
        dt_seconds = dt_ms / 1000.0
        pixel_displacement = distances * OMEGA_RAD_S * dt_seconds
        all_displacements.append(pixel_displacement)
    
    # Use same colorbar scale for both images
    vmin = 0
    vmax = max(np.max(disp) for disp in all_displacements)
    
    fig, axes = plt.subplots(1, len(dt_values), figsize=(15, 6))
    if len(dt_values) == 1:
        axes = [axes]
    
    for i, dt_ms in enumerate(dt_values):
        dt_seconds = dt_ms / 1000.0
        
        # Calculate pixel displacement for each pixel
        pixel_displacement = distances * OMEGA_RAD_S * dt_seconds
        
        # Create the flow magnitude image with consistent colorbar
        im = axes[i].imshow(pixel_displacement, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Flow Magnitude Map (dt = {dt_ms}ms)', fontsize=14)
        axes[i].set_xlabel('X (pixels)', fontsize=12)
        axes[i].set_ylabel('Y (pixels)', fontsize=12)
        
        # Add ROI circle (dashed line)
        circle = plt.Circle((center_x, center_y), DISC_RADIUS, fill=False, 
                          linestyle='--', color='white', linewidth=2, alpha=0.8)
        axes[i].add_patch(circle)
        
        # Add center marker
        axes[i].plot(center_x, center_y, 'r+', markersize=10, markeredgewidth=2)
        axes[i].text(center_x + 20, center_y + 20, 'Center', color='red', fontsize=10)
    
    # Add single colorbar for both images, positioned between them
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.1)
    cbar.set_label('Pixel Displacement (px)', fontsize=12)
    
    # Adjust layout to make room for colorbar
    plt.subplots_adjust(right=0.85)
    
    # Save flow magnitude image
    flow_path = os.path.join(output_dir, "flow_magnitude_maps.png")
    plt.savefig(flow_path, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"Saved flow magnitude maps: {flow_path}")
    
    plt.close()

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("=== DT Cancellation Analysis - Angus's Fine Resolution ===")
    print(f"Window predictions directory: {WINDOW_PREDICTIONS_DIR}")
    print(f"Cancellation parameters: {BIN_MS}ms temporal, {R_PIX}px spatial")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"FOCUS: Fine resolution analysis (0.1ms steps for 0-2ms, 0.25ms for 2-3ms)")
    print(f"GOAL: Find t90, t80 thresholds and local prediction window")
    
    # Generate dt values with Angus's approach - FINER RESOLUTION
    # Primary zoom (0-2ms): 0.1ms steps
    # Context tail (2-3ms): 0.25ms steps
    dt_values_ms_primary = np.arange(0.0, 2.0 + 1e-9, 0.1)  # 0.0, 0.1, 0.2, ..., 2.0
    dt_values_ms_tail = np.arange(2.0, 3.0 + 1e-9, 0.25)     # 2.0, 2.25, 2.5, 2.75, 3.0
    
    # Combine and deduplicate
    dt_values_ms = np.concatenate([dt_values_ms_primary, dt_values_ms_tail])
    dt_values_ms = np.unique(dt_values_ms)  # Remove duplicates (2.0 will be duplicated)
    
    print(f"Analyzing {len(dt_values_ms)} dt values:")
    print(f"  Primary (0-2ms): {len(dt_values_ms_primary)} points at 0.1ms steps")
    print(f"  Tail (2-3ms): {len(dt_values_ms_tail)} points at 0.25ms steps")
    print(f"  Range: {dt_values_ms[0]:.1f}ms to {dt_values_ms[-1]:.1f}ms")
    
    # Determine window directories (auto-discover if available)
    discovered = discover_window_dirs(WINDOW_PREDICTIONS_DIR)
    all_results = []
    
    if discovered:
        print(f"Found {len(discovered)} window directories")
        if USE_SUBSET_FOR_TESTING:
            print(f"⚠️  LARGE DATASET DETECTED - Using subset mode ({SUBSET_SIZE:,} events per dt)")
            print(f"   Set USE_SUBSET_FOR_TESTING=False for full analysis")
        for window_idx, (window_dir, label) in enumerate(discovered):
            window_results = analyze_window_dt_cancellation_dir(window_idx, window_dir, label, dt_values_ms)
            all_results.append(window_results)
    else:
        print("No window directories found, using fallback window analysis")
        for window_idx, window in enumerate(WINDOWS):
            window_results = analyze_window_dt_cancellation(window_idx, window, dt_values_ms)
            all_results.append(window_results)
    
    # Create plots with Angus's enhancements
    print("\nCreating plots with Angus's enhancements...")
    t90_ms, t80_ms = create_cancellation_plots(all_results, OUTPUT_DIR)
    
    # Generate flow magnitude image as Angus suggested
    print("\nGenerating flow magnitude image...")
    create_flow_magnitude_image(OUTPUT_DIR)
    
    # Print summary - FOCUS ON EARLY DROP-OFF
    # Print summary with Angus's key findings
    print("\n=== Summary - Angus's Analysis Results ===")
    
    # Define constants for summary
    OMEGA_RAD_S = 3.612  # Actual mean angular velocity from tracker data
    MEAN_RADIUS_PX = 199  # Actual mean radius from real events
    
    if t90_ms is not None:
        t90_px = t90_ms * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000
        print(f"✓ t90 (cancellation < 90%): {t90_ms:.1f}ms ≈ {t90_px:.1f}px")
    else:
        print("✓ t90: Cancellation stays above 90% for all tested dt values")
    
    if t80_ms is not None:
        t80_px = t80_ms * OMEGA_RAD_S * MEAN_RADIUS_PX / 1000
        print(f"✓ t80 (cancellation < 80%): {t80_ms:.1f}ms ≈ {t80_px:.1f}px")
    else:
        print("✓ t80: Cancellation stays above 80% for all tested dt values")
    
    print(f"\nKey Findings:")
    print(f"• Fine resolution analysis: {len(dt_values_ms)} dt values (0.1ms steps for 0-2ms)")
    print(f"• Angular velocity: {OMEGA_RAD_S:.1f} rad/s")
    print(f"• Mean event radius: {MEAN_RADIUS_PX:.0f}px")
    if t90_ms:
        print(f"• Local prediction window: Cancellation remains above 90% for up to ≈ {t90_ms:.1f}ms")
        print(f"• This corresponds to ≈ {t90_px:.1f}px forward prediction")
        print(f"• Matches the few-pixel local range of biological retinal circuits")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed_time:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Fine resolution plot: dt_cancellation_analysis_fine_resolution.png")
    print(f"Flow magnitude maps: flow_magnitude_maps.png")

if __name__ == "__main__":
    main()