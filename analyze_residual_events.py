#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def analyze_residual_events(combined_events, resid_real, resid_pred, time_window=(5.0, 5.005)):
    """Analyze why certain events weren't cancelled"""
    
    t0, t1 = time_window
    print(f"Analyzing residuals in window: {t0}s to {t1}s")
    
    # Get events in time window
    mask_all = (combined_events[:, 3] >= t0) & (combined_events[:, 3] < t1)
    window_events = combined_events[mask_all]
    
    real_events = window_events[window_events[:, 4] == 0.0]
    pred_events = window_events[window_events[:, 4] == 1.0]
    
    # Get residuals in time window
    r_mask = (resid_real[:, 3] >= t0) & (resid_real[:, 3] < t1)
    p_mask = (resid_pred[:, 3] >= t0) & (resid_pred[:, 3] < t1)
    wr = resid_real[r_mask]
    wp = resid_pred[p_mask]
    
    print(f"Real events in window: {len(real_events):,}")
    print(f"Pred events in window: {len(pred_events):,}")
    print(f"Real residuals: {len(wr):,}")
    print(f"Pred residuals: {len(wp):,}")
    
    # Analyze why real events weren't cancelled
    if len(wr) > 0 and len(pred_events) > 0:
        print("\n=== Analyzing Real Event Residuals ===")
        analyze_why_not_cancelled(wr, pred_events, "Real")
    
    # Analyze why predicted events weren't cancelled
    if len(wp) > 0 and len(real_events) > 0:
        print("\n=== Analyzing Predicted Event Residuals ===")
        analyze_why_not_cancelled(wp, real_events, "Predicted")
    
    # Visualize residual patterns
    visualize_residual_patterns(wr, wp, real_events, pred_events)

def analyze_why_not_cancelled(residual_events, potential_matches, event_type):
    """Analyze why certain events couldn't find matches"""
    
    if len(residual_events) == 0 or len(potential_matches) == 0:
        return
    
    # Find nearest neighbors for each residual event
    tree = cKDTree(potential_matches[:, :2])
    distances, indices = tree.query(residual_events[:, :2], k=1)
    
    # Analyze distance distribution
    print(f"{event_type} events - Distance to nearest potential match:")
    print(f"  Min distance: {distances.min():.2f} pixels")
    print(f"  Max distance: {distances.max():.2f} pixels")
    print(f"  Mean distance: {distances.mean():.2f} pixels")
    print(f"  Median distance: {np.median(distances):.2f} pixels")
    
    # Count events within different tolerance ranges
    tolerance_ranges = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    for tol in tolerance_ranges:
        within_tol = np.sum(distances <= tol)
        percentage = (within_tol / len(distances)) * 100
        print(f"  Within {tol:2.1f}px: {within_tol:4d} events ({percentage:5.1f}%)")
    
    # Find events that are very close but still not matched
    very_close = distances <= 2.0
    if np.any(very_close):
        print(f"  ⚠️  {np.sum(very_close)} events within 2px but not matched!")
        print(f"     This might indicate a bug in matching logic")
    
    # Find events that are very far (likely hand/noise)
    very_far = distances > 20.0
    if np.any(very_far):
        print(f"  ✅ {np.sum(very_far)} events >20px away (likely hand/noise)")

def visualize_residual_patterns(wr, wp, real_events, pred_events):
    """Visualize the patterns of residual events"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All events in window
    axes[0,0].scatter(real_events[:, 0], real_events[:, 1], s=1, alpha=0.6, c='blue', label=f'Real ({len(real_events):,})')
    axes[0,0].scatter(pred_events[:, 0], pred_events[:, 1], s=1, alpha=0.6, c='red', label=f'Predicted ({len(pred_events):,})')
    axes[0,0].set_title('All Events in Time Window')
    axes[0,0].legend()
    
    # Plot 2: Residual events only
    axes[0,1].scatter(wr[:, 0], wr[:, 1], s=2, alpha=0.8, c='blue', label=f'Real Residual ({len(wr):,})')
    axes[0,1].scatter(wp[:, 0], wp[:, 1], s=2, alpha=0.8, c='red', label=f'Pred Residual ({len(wp):,})')
    axes[0,1].set_title('Residual Events (Not Cancelled)')
    axes[0,1].legend()
    
    # Plot 3: Cancelled events (original - residual)
    cancelled_real = real_events[~np.isin(real_events[:, :2], wr[:, :2]).all(axis=1)]
    cancelled_pred = pred_events[~np.isin(pred_events[:, :2], wp[:, :2]).all(axis=1)]
    
    axes[1,0].scatter(cancelled_real[:, 0], cancelled_real[:, 1], s=1, alpha=0.6, c='green', label=f'Cancelled Real ({len(cancelled_real):,})')
    axes[1,0].scatter(cancelled_pred[:, 0], cancelled_pred[:, 1], s=1, alpha=0.6, c='orange', label=f'Cancelled Pred ({len(cancelled_pred):,})')
    axes[1,0].set_title('Cancelled Events (Successfully Matched)')
    axes[1,0].legend()
    
    # Plot 4: Density comparison
    axes[1,1].hist2d(wr[:, 0], wr[:, 1], bins=50, alpha=0.7, cmap='Blues', label='Real Residuals')
    axes[1,1].hist2d(wp[:, 0], wp[:, 1], bins=50, alpha=0.7, cmap='Reds', label='Pred Residuals')
    axes[1,1].set_title('Residual Event Density')
    
    # Format all plots
    for ax in axes.flat:
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 720)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (pixels)')
        if ax in [axes[0,0], axes[1,0]]:
            ax.set_ylabel('Y (pixels)')
    
    plt.tight_layout()
    plt.savefig('residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Successfully cancelled: {len(cancelled_real):,} real + {len(cancelled_pred):,} predicted events")
    print(f"Remaining residuals: {len(wr):,} real + {len(wp):,} predicted events")
    
    if len(real_events) > 0:
        cancellation_rate = (len(cancelled_real) / len(real_events)) * 100
        print(f"Cancellation rate: {cancellation_rate:.1f}%")
    
    print(f"\nResidual events are likely:")
    print(f"- Hand events (non-disc motion)")
    print(f"- Events at disc boundaries")
    print(f"- Temporal misalignments")
    print(f"- Spatial outliers")

if __name__ == "__main__":
    # Load your data
    combined_events = np.load("./combined_events_with_predictions.npy")
    
    # You'll need to run cancellation first to get residuals
    print("This script analyzes residual events after cancellation.")
    print("Run visualize_event_comparison.py first to generate residuals, then:")
    print("resid_real = np.load('./residual_real.npy')  # if you saved them")
    print("resid_pred = np.load('./residual_pred.npy')  # if you saved them")
    
    # For now, let's analyze the current window
    print("\nAnalyzing current time window...")
    analyze_residual_events(combined_events, None, None)













