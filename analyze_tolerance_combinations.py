#!/usr/bin/env python3
"""
Analyze different combinations of spatial and temporal tolerances to find optimal cancellation parameters.
This script uses existing window prediction data to test various tolerance combinations efficiently.
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

# =============== Configuration ===============
WINDOW_PREDICTIONS_DIR = "./window_predictions"

# Tolerance ranges to test
SPATIAL_TOLERANCE_RANGE = (0.5, 5.0, 0.5)  # (min, max, step) in pixels
TEMPORAL_TOLERANCE_RANGE = (1.0, 10.0, 1.0)  # (min, max, step) in milliseconds

# Polarity mode
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# Disc center coordinates and radius (same as visualize_time_window.py)
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 250

# Time windows (same as in generate_window_predictions.py)
WINDOWS = [
    (5.000, 5.010),
    (8.200, 8.210),
    (9.000, 9.010),
]

# DT value to use for analysis (can be changed to test different dt values)
ANALYSIS_DT_MS = 3  # Use 3ms predictions for tolerance analysis

# Use improved cancellation method (fixes temporal binning problem)
USE_IMPROVED_CANCELLATION = False  # Keep original method - it's stable and works

# Output settings
OUTPUT_DIR = "./tolerance_analysis_results"
PLOT_DPI = 150

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

def run_cancellation_improved(combined_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Simple and fast improved cancellation with direct time matching.
    Uses vectorized operations for better performance.
    """
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    
    # Split events
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    
    if len(real_events) == 0 or len(pred_events) == 0:
        return real_events, pred_events, 0
    
    # Calculate target times for all real events
    target_times = real_events[:, 3] + dt_seconds
    
    # Use efficient time-based filtering
    total_matches = 0
    matched_real_indices = []
    matched_pred_indices = []
    
    # Build spatial tree for predicted events
    pred_tree = cKDTree(pred_events[:, :2])
    
    # For each real event, find predicted events in both temporal and spatial windows
    for i, real_event in enumerate(real_events):
        target_time = target_times[i]
        
        # Temporal filter: find predicted events within time tolerance
        time_mask = np.abs(pred_events[:, 3] - target_time) <= temporal_tolerance_s
        temporal_candidates = np.where(time_mask)[0]
        
        if len(temporal_candidates) == 0:
            continue
        
        # Spatial filter: find closest predicted event within spatial tolerance
        distances, indices = pred_tree.query(
            real_event[:2], k=len(temporal_candidates), 
            distance_upper_bound=spatial_tolerance_pixels
        )
        
        # Filter to only temporal candidates
        valid_candidates = []
        valid_distances = []
        
        for j, (dist, pred_idx) in enumerate(zip(distances, indices)):
            if pred_idx < len(pred_events) and pred_idx in temporal_candidates:
                # Check if already matched
                if pred_idx not in matched_pred_indices:
                    # Check polarity
                    if check_polarity_match(real_event[2], pred_events[pred_idx, 2]):
                        valid_candidates.append(pred_idx)
                        valid_distances.append(dist)
        
        # Find best match (closest spatially)
        if valid_candidates:
            best_idx = np.argmin(valid_distances)
            best_pred_idx = valid_candidates[best_idx]
            
            matched_real_indices.append(i)
            matched_pred_indices.append(best_pred_idx)
            total_matches += 1
    
    # Create masks for unmatched events
    matched_real_mask = np.zeros(len(real_events), dtype=bool)
    matched_pred_mask = np.zeros(len(pred_events), dtype=bool)
    
    if matched_real_indices:
        matched_real_mask[matched_real_indices] = True
    if matched_pred_indices:
        matched_pred_mask[matched_pred_indices] = True
    
    # Return unmatched events
    residual_real = real_events[~matched_real_mask]
    residual_pred = pred_events[~matched_pred_mask]
    
    return residual_real, residual_pred, total_matches

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

def run_cancellation_for_window(combined_events, temporal_bin_ms, spatial_tolerance_pixels):
    """Run ego-motion cancellation for a single window"""
    timestamps = combined_events[:, 3]
    time_bin_edges = time_edges(float(timestamps.min()), float(timestamps.max()), temporal_bin_ms)

    total_real_events = int(np.sum(combined_events[:, 4] == 0.0))
    total_predicted_events = int(np.sum(combined_events[:, 4] == 1.0))
    total_matched_pairs = 0
    unmatched_real_chunks, unmatched_predicted_chunks = [], []

    total_events = len(combined_events)
    current_index = 0
    
    for bin_index in range(len(time_bin_edges) - 1):
        bin_start_time, bin_end_time = time_bin_edges[bin_index], time_bin_edges[bin_index + 1]
        
        # Find events within this time bin
        bin_start_index = current_index
        while bin_start_index < total_events and combined_events[bin_start_index, 3] < bin_start_time:
            bin_start_index += 1
        
        bin_end_index = bin_start_index
        while bin_end_index < total_events and combined_events[bin_end_index, 3] < bin_end_time:
            bin_end_index += 1
        
        current_index = bin_start_index
        
        if bin_end_index <= bin_start_index:
            continue
            
        bin_events = combined_events[bin_start_index:bin_end_index]
        real_events_in_bin = bin_events[bin_events[:, 4] == 0.0]
        predicted_events_in_bin = bin_events[bin_events[:, 4] == 1.0]
        
        if len(real_events_in_bin) == 0 and len(predicted_events_in_bin) == 0:
            continue
            
        # Match events in this time bin
        unmatched_real_mask, unmatched_predicted_mask, num_matches = cancel_events_in_time_bin(
            real_events_in_bin, predicted_events_in_bin, spatial_tolerance_pixels
        )
        
        total_matched_pairs += num_matches
        
        # Collect unmatched events
        if unmatched_real_mask.any():
            unmatched_real_chunks.append(real_events_in_bin[unmatched_real_mask])
        if unmatched_predicted_mask.any():
            unmatched_predicted_chunks.append(predicted_events_in_bin[unmatched_predicted_mask])
    
    # Combine all unmatched events
    if unmatched_real_chunks:
        residual_real_events = np.vstack(unmatched_real_chunks)
    else:
        residual_real_events = np.zeros((0, 5), dtype=combined_events.dtype)
        
    if unmatched_predicted_chunks:
        residual_predicted_events = np.vstack(unmatched_predicted_chunks)
    else:
        residual_predicted_events = np.zeros((0, 5), dtype=combined_events.dtype)
    
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

def analyze_tolerance_combinations():
    """Analyze all combinations of spatial and temporal tolerances"""
    print("=== Tolerance Combination Analysis ===")
    
    # Generate tolerance values
    spatial_values = np.arange(SPATIAL_TOLERANCE_RANGE[0], 
                              SPATIAL_TOLERANCE_RANGE[1] + SPATIAL_TOLERANCE_RANGE[2], 
                              SPATIAL_TOLERANCE_RANGE[2])
    temporal_values = np.arange(TEMPORAL_TOLERANCE_RANGE[0], 
                               TEMPORAL_TOLERANCE_RANGE[1] + TEMPORAL_TOLERANCE_RANGE[2], 
                               TEMPORAL_TOLERANCE_RANGE[2])
    
    print(f"Spatial tolerances: {spatial_values}")
    print(f"Temporal tolerances: {temporal_values}")
    print(f"Total combinations: {len(spatial_values) * len(temporal_values)}")
    
    # Initialize results storage
    results = []
    
    # Test each window
    for window_idx, window in enumerate(WINDOWS):
        t0, t1 = window
        print(f"\nAnalyzing window {window_idx + 1}: {t0:.3f}s to {t1:.3f}s")
        
        # Load prediction data for this window and dt
        window_dir = os.path.join(WINDOW_PREDICTIONS_DIR, f"window_{window_idx + 1}_{t0:.3f}s_to_{t1:.3f}s")
        filename = f"combined_events_dt_{ANALYSIS_DT_MS:02d}ms.npy"
        filepath = os.path.join(window_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"  Warning: File {filename} not found, skipping window {window_idx + 1}")
            continue
        
        # Load combined events
        combined_events = np.load(filepath)
        print(f"  Loaded {len(combined_events):,} events")
        
        # Test all tolerance combinations
        for spatial_tol in spatial_values:
            for temporal_tol in temporal_values:
                # Run cancellation with these tolerances
                if USE_IMPROVED_CANCELLATION:
                    # Use improved direct time matching
                    dt_seconds = ANALYSIS_DT_MS / 1000.0
                    residual_real, residual_pred, matched_pairs = run_cancellation_improved(
                        combined_events, dt_seconds, temporal_tol, spatial_tol
                    )
                else:
                    # Use original binning method
                    residual_real, residual_pred, matched_pairs = run_cancellation_for_window(
                        combined_events, temporal_tol, spatial_tol
                    )
                
                # Calculate ROI cancellation rate
                roi_cancellation_rate, total_roi_real, total_roi_cancelled = calculate_roi_cancellation_rate(
                    combined_events, residual_real, (DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS
                )
                
                # Store results
                results.append({
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
    
    return results

def create_tolerance_heatmaps(results_df, output_dir):
    """Create heatmap visualizations for tolerance combinations"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots for each window
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot each window separately
    for window_idx in range(len(WINDOWS)):
        ax = axes[window_idx]
        
        # Filter data for this window
        window_data = results_df[results_df['window_idx'] == window_idx + 1]
        
        if len(window_data) == 0:
            ax.text(0.5, 0.5, f'No data for Window {window_idx + 1}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create pivot table for heatmap
        pivot_data = window_data.pivot_table(
            values='cancellation_rate', 
            index='temporal_tolerance', 
            columns='spatial_tolerance', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis', 
                   ax=ax, cbar_kws={'label': 'Cancellation Rate (%)'})
        ax.set_title(f'Window {window_idx + 1}: {WINDOWS[window_idx][0]:.3f}s to {WINDOWS[window_idx][1]:.3f}s')
        ax.set_xlabel('Spatial Tolerance (pixels)')
        ax.set_ylabel('Temporal Tolerance (ms)')
    
    # Combined heatmap (average across all windows)
    ax_combined = axes[3]
    combined_data = results_df.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
    pivot_combined = combined_data.pivot_table(
        values='cancellation_rate', 
        index='temporal_tolerance', 
        columns='spatial_tolerance', 
        aggfunc='mean'
    )
    
    sns.heatmap(pivot_combined, annot=True, fmt='.1f', cmap='viridis', 
               ax=ax_combined, cbar_kws={'label': 'Avg Cancellation Rate (%)'})
    ax_combined.set_title('Combined: Average Across All Windows')
    ax_combined.set_xlabel('Spatial Tolerance (pixels)')
    ax_combined.set_ylabel('Temporal Tolerance (ms)')
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = os.path.join(output_dir, "tolerance_combination_heatmaps.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved heatmap: {plot_filename}")
    
    return fig

def create_3d_surface_plot(results_df, output_dir):
    """Create 3D surface plot showing cancellation rates"""
    from mpl_toolkits.mplot3d import Axes3D
    
    # Average across all windows
    combined_data = results_df.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data for 3D plotting
    spatial_unique = sorted(combined_data['spatial_tolerance'].unique())
    temporal_unique = sorted(combined_data['temporal_tolerance'].unique())
    
    X, Y = np.meshgrid(spatial_unique, temporal_unique)
    Z = np.zeros_like(X)
    
    for i, temp in enumerate(temporal_unique):
        for j, spat in enumerate(spatial_unique):
            mask = (combined_data['spatial_tolerance'] == spat) & (combined_data['temporal_tolerance'] == temp)
            if mask.any():
                Z[i, j] = combined_data[mask]['cancellation_rate'].iloc[0]
    
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Spatial Tolerance (pixels)')
    ax.set_ylabel('Temporal Tolerance (ms)')
    ax.set_zlabel('Cancellation Rate (%)')
    ax.set_title('Cancellation Rate vs Tolerance Parameters (3D Surface)')
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Save plot
    plot_filename = os.path.join(output_dir, "tolerance_3d_surface.png")
    plt.savefig(plot_filename, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved 3D surface: {plot_filename}")
    
    return fig

def find_optimal_tolerances(results_df):
    """Find the optimal tolerance combinations"""
    print("\n=== Optimal Tolerance Analysis ===")
    
    # Find best combination overall (average across windows)
    combined_data = results_df.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
    best_overall = combined_data.loc[combined_data['cancellation_rate'].idxmax()]
    
    print(f"Best overall combination:")
    print(f"  Spatial tolerance: {best_overall['spatial_tolerance']:.1f} pixels")
    print(f"  Temporal tolerance: {best_overall['temporal_tolerance']:.1f} ms")
    print(f"  Average cancellation rate: {best_overall['cancellation_rate']:.2f}%")
    
    # Find best for each window
    print(f"\nBest combination per window:")
    for window_idx in range(len(WINDOWS)):
        window_data = results_df[results_df['window_idx'] == window_idx + 1]
        if len(window_data) > 0:
            best_window = window_data.loc[window_data['cancellation_rate'].idxmax()]
            print(f"  Window {window_idx + 1}: spatial={best_window['spatial_tolerance']:.1f}px, "
                  f"temporal={best_window['temporal_tolerance']:.1f}ms, "
                  f"rate={best_window['cancellation_rate']:.2f}%")
    
    # Find top 5 combinations
    print(f"\nTop 5 combinations (overall):")
    top5 = combined_data.nlargest(5, 'cancellation_rate')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. spatial={row['spatial_tolerance']:.1f}px, "
              f"temporal={row['temporal_tolerance']:.1f}ms, "
              f"rate={row['cancellation_rate']:.2f}%")
    
    return best_overall

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("=== Tolerance Combination Analysis ===")
    print(f"Using dt={ANALYSIS_DT_MS}ms predictions")
    print(f"Spatial range: {SPATIAL_TOLERANCE_RANGE[0]} to {SPATIAL_TOLERANCE_RANGE[1]} (step: {SPATIAL_TOLERANCE_RANGE[2]})")
    print(f"Temporal range: {TEMPORAL_TOLERANCE_RANGE[0]} to {TEMPORAL_TOLERANCE_RANGE[1]} (step: {TEMPORAL_TOLERANCE_RANGE[2]})")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Run analysis
    results = analyze_tolerance_combinations()
    
    if not results:
        print("No results generated. Check if prediction data exists.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_filename = os.path.join(OUTPUT_DIR, "tolerance_analysis_results.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"\nSaved results: {csv_filename}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig1 = create_tolerance_heatmaps(results_df, OUTPUT_DIR)
    fig2 = create_3d_surface_plot(results_df, OUTPUT_DIR)
    
    # Find optimal tolerances
    best_tolerances = find_optimal_tolerances(results_df)
    
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
