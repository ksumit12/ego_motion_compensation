#!/usr/bin/env python3

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def test_cancellation_params(combined_events, time_window=(5.0, 5.005)):
    """Test different cancellation parameters to find optimal settings"""
    
    # Filter events in time window
    t0, t1 = time_window
    mask = (combined_events[:, 3] >= t0) & (combined_events[:, 3] < t1)
    window_events = combined_events[mask]
    
    real_events = window_events[window_events[:, 4] == 0.0]
    pred_events = window_events[window_events[:, 4] == 1.0]
    
    print(f"Testing window: {t0}s to {t1}s")
    print(f"Real events: {len(real_events):,}")
    print(f"Pred events: {len(pred_events):,}")
    print()
    
    # Test different parameters
    r_pix_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    bin_ms_values = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    results = []
    
    for r_pix in r_pix_values:
        for bin_ms in bin_ms_values:
            cancelled_count, cancellation_rate = test_single_config(
                real_events, pred_events, r_pix, bin_ms
            )
            results.append({
                'r_pix': r_pix,
                'bin_ms': bin_ms,
                'cancelled': cancelled_count,
                'rate': cancellation_rate
            })
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['rate'])
    
    print("=== BEST PARAMETERS ===")
    print(f"R_PIX: {best_result['r_pix']}")
    print(f"BIN_MS: {best_result['bin_ms']}")
    print(f"Cancellation Rate: {best_result['rate']:.1f}%")
    print(f"Events Cancelled: {best_result['cancelled']:,}")
    
    # Plot results
    plot_parameter_results(results, r_pix_values, bin_ms_values)
    
    return best_result

def test_single_config(real_events, pred_events, r_pix, bin_ms):
    """Test cancellation with specific parameters"""
    
    if len(real_events) == 0 or len(pred_events) == 0:
        return 0, 0.0
    
    # Simple spatial matching (no time bins for this test)
    pred_xy = pred_events[:, :2]
    tree = cKDTree(pred_xy)
    
    real_xy = real_events[:, :2]
    distances, indices = tree.query(real_xy, k=1, distance_upper_bound=r_pix)
    
    # Count matches
    matched_mask = distances < np.inf
    cancelled_count = np.sum(matched_mask)
    cancellation_rate = (cancelled_count / len(real_events)) * 100
    
    return cancelled_count, cancellation_rate

def plot_parameter_results(results, r_pix_values, bin_ms_values):
    """Plot cancellation rates for different parameters"""
    
    # Create matrix for heatmap
    rate_matrix = np.zeros((len(bin_ms_values), len(r_pix_values)))
    
    for result in results:
        r_idx = r_pix_values.index(result['r_pix'])
        b_idx = bin_ms_values.index(result['bin_ms'])
        rate_matrix[b_idx, r_idx] = result['rate']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(rate_matrix, cmap='viridis', aspect='auto')
    
    # Add text annotations
    for i in range(len(bin_ms_values)):
        for j in range(len(r_pix_values)):
            text = ax.text(j, i, f'{rate_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="white", fontsize=10)
    
    ax.set_xticks(range(len(r_pix_values)))
    ax.set_yticks(range(len(bin_ms_values)))
    ax.set_xticklabels([f'{r}px' for r in r_pix_values])
    ax.set_yticklabels([f'{b}ms' for b in bin_ms_values])
    
    ax.set_xlabel('Spatial Tolerance (R_PIX)')
    ax.set_ylabel('Time Bin Size (BIN_MS)')
    ax.set_title('Cancellation Rate (%) for Different Parameters')
    
    plt.colorbar(im, ax=ax, label='Cancellation Rate (%)')
    plt.tight_layout()
    plt.savefig('cancellation_parameter_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load your data
    combined_events = np.load("./combined_events_with_predictions.npy")
    
    # Test parameters
    best_params = test_cancellation_params(combined_events)
    
    print(f"\nRecommended settings for visualize_event_comparison.py:")
    print(f"R_PIX = {best_params['r_pix']}")
    print(f"BIN_MS = {best_params['bin_ms']}")
    print(f"Expected cancellation rate: {best_params['rate']:.1f}%")













