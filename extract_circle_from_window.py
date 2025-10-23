#!/usr/bin/env python3
"""
Circle extraction script for event data.
Takes a small time window (1-2 seconds) and plots all points to extract circle parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# Set headless backend only if no display is available
try:
    if not os.environ.get('DISPLAY'):
        import matplotlib
        matplotlib.use("Agg")
except:
    pass

# =============== Configuration ===============
COMBINED_PATH = "./combined_events_with_predictions.npy"
OUTPUT_DIR = "./main_results"

# Time window to analyze (1-2 seconds)
TIME_WINDOW = (5.0, 7.0)  # Adjust as needed

# Image settings
IMG_W, IMG_H = 1280, 720

# Circle fitting parameters
MAX_ITERATIONS = 1000
TOLERANCE = 1e-6

# =============== Data Loading ===============
def load_combined(path):
    """Load combined events data"""
    arr = np.load(path, mmap_mode="r")
    if not np.all(arr[:-1, 3] <= arr[1:, 3]):
        arr = arr[np.argsort(arr[:, 3])]
    print(f"Loaded {len(arr):,} events "
          f"(real={int(np.sum(arr[:,4]==0.0)):,}, pred={int(np.sum(arr[:,4]==1.0)):,})")
    return arr

def extract_time_window(combined_events, start_time, end_time):
    """Extract events within specified time window"""
    time_mask = (combined_events[:, 3] >= start_time) & (combined_events[:, 3] < end_time)
    window_events = combined_events[time_mask]
    
    # Separate real and predicted events
    real_events = window_events[window_events[:, 4] == 0.0]
    predicted_events = window_events[window_events[:, 4] == 1.0]
    
    print(f"Time window {start_time:.3f}-{end_time:.3f}s:")
    print(f"  Real events: {len(real_events):,}")
    print(f"  Predicted events: {len(predicted_events):,}")
    print(f"  Total events: {len(window_events):,}")
    
    return real_events, predicted_events, window_events

# =============== Circle Fitting ===============
def circle_residuals(params, points):
    """Calculate residuals for circle fitting using least squares"""
    xc, yc, r = params
    x, y = points[:, 0], points[:, 1]
    
    # Distance from each point to circle center
    distances = np.sqrt((x - xc)**2 + (y - yc)**2)
    
    # Residual is difference between actual distance and radius
    residuals = distances - r
    
    return residuals

def fit_circle_least_squares(points):
    """Fit circle to points using least squares method"""
    if len(points) < 3:
        return None, None, None, None
    
    # Initial guess: center at centroid, radius as mean distance from centroid
    x_center = np.mean(points[:, 0])
    y_center = np.mean(points[:, 1])
    initial_radius = np.mean(np.sqrt((points[:, 0] - x_center)**2 + (points[:, 1] - y_center)**2))
    
    initial_params = [x_center, y_center, initial_radius]
    
    try:
        # Minimize sum of squared residuals
        result = minimize(
            lambda params: np.sum(circle_residuals(params, points)**2),
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': MAX_ITERATIONS, 'ftol': TOLERANCE}
        )
        
        if result.success:
            xc, yc, r = result.x
            residuals = circle_residuals(result.x, points)
            rmse = np.sqrt(np.mean(residuals**2))
            return xc, yc, r, rmse
        else:
            print(f"Circle fitting failed: {result.message}")
            return None, None, None, None
            
    except Exception as e:
        print(f"Circle fitting error: {e}")
        return None, None, None, None

# =============== Visualization ===============
def plot_events_with_circle(real_events, predicted_events, all_events, 
                           real_circle=None, pred_circle=None, 
                           output_path=None):
    """Plot events and fitted circles"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Event Analysis - Time Window {TIME_WINDOW[0]:.1f}s to {TIME_WINDOW[1]:.1f}s', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Real events only
    ax1 = axes[0, 0]
    ax1.scatter(real_events[:, 0], real_events[:, 1], s=1, alpha=0.6, c='blue', label=f'Real ({len(real_events):,})')
    ax1.set_xlim(0, IMG_W)
    ax1.set_ylim(0, IMG_H)
    ax1.invert_yaxis()
    ax1.set_title('Real Events')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add real circle if fitted
    if real_circle is not None:
        xc, yc, r, rmse = real_circle
        circle = plt.Circle((xc, yc), r, fill=False, color='red', linewidth=2, linestyle='--')
        ax1.add_patch(circle)
        ax1.plot(xc, yc, 'ro', markersize=8, label=f'Center: ({xc:.1f}, {yc:.1f})')
        ax1.text(0.02, 0.98, f'Radius: {r:.1f}px\nRMSE: {rmse:.2f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax1.legend()
    
    # Plot 2: Predicted events only
    ax2 = axes[0, 1]
    ax2.scatter(predicted_events[:, 0], predicted_events[:, 1], s=1, alpha=0.6, c='red', label=f'Predicted ({len(predicted_events):,})')
    ax2.set_xlim(0, IMG_W)
    ax2.set_ylim(0, IMG_H)
    ax2.invert_yaxis()
    ax2.set_title('Predicted Events')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add predicted circle if fitted
    if pred_circle is not None:
        xc, yc, r, rmse = pred_circle
        circle = plt.Circle((xc, yc), r, fill=False, color='blue', linewidth=2, linestyle='--')
        ax2.add_patch(circle)
        ax2.plot(xc, yc, 'bo', markersize=8, label=f'Center: ({xc:.1f}, {yc:.1f})')
        ax2.text(0.02, 0.98, f'Radius: {r:.1f}px\nRMSE: {rmse:.2f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.legend()
    
    # Plot 3: Combined events
    ax3 = axes[1, 0]
    ax3.scatter(real_events[:, 0], real_events[:, 1], s=1, alpha=0.6, c='blue', label=f'Real ({len(real_events):,})')
    ax3.scatter(predicted_events[:, 0], predicted_events[:, 1], s=1, alpha=0.6, c='red', label=f'Predicted ({len(predicted_events):,})')
    ax3.set_xlim(0, IMG_W)
    ax3.set_ylim(0, IMG_H)
    ax3.invert_yaxis()
    ax3.set_title('Combined Events')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add both circles if fitted
    if real_circle is not None:
        xc, yc, r, rmse = real_circle
        circle = plt.Circle((xc, yc), r, fill=False, color='blue', linewidth=2, linestyle='--', alpha=0.7)
        ax3.add_patch(circle)
        ax3.plot(xc, yc, 'bo', markersize=6, alpha=0.7)
    
    if pred_circle is not None:
        xc, yc, r, rmse = pred_circle
        circle = plt.Circle((xc, yc), r, fill=False, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax3.add_patch(circle)
        ax3.plot(xc, yc, 'ro', markersize=6, alpha=0.7)
    
    # Plot 4: Event density heatmap
    ax4 = axes[1, 1]
    if len(all_events) > 0:
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(all_events[:, 0], all_events[:, 1], 
                                            bins=[50, 50], range=[[0, IMG_W], [0, IMG_H]])
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]  # Note: y-axis inverted
        im = ax4.imshow(hist.T, extent=extent, cmap='hot', origin='upper', aspect='auto')
        ax4.set_title('Event Density Heatmap')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax4, label='Event Count')
    else:
        ax4.text(0.5, 0.5, 'No events in time window', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Event Density Heatmap')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    
    return fig

# =============== Main Analysis ===============
def analyze_circle_extraction():
    """Main function to analyze circle extraction"""
    print("Loading combined events data...")
    combined = load_combined(COMBINED_PATH)
    
    print(f"\nExtracting time window {TIME_WINDOW[0]:.1f}s to {TIME_WINDOW[1]:.1f}s...")
    real_events, predicted_events, all_events = extract_time_window(combined, TIME_WINDOW[0], TIME_WINDOW[1])
    
    if len(all_events) == 0:
        print("No events found in specified time window!")
        return
    
    # Fit circles to real and predicted events separately
    print("\nFitting circles...")
    
    real_circle = None
    pred_circle = None
    
    if len(real_events) >= 3:
        print("Fitting circle to real events...")
        real_circle = fit_circle_least_squares(real_events[:, :2])  # Only x, y coordinates
        if real_circle[0] is not None:
            xc, yc, r, rmse = real_circle
            print(f"  Real circle: center=({xc:.1f}, {yc:.1f}), radius={r:.1f}px, RMSE={rmse:.2f}")
        else:
            print("  Failed to fit circle to real events")
    else:
        print("  Not enough real events for circle fitting (need ≥3)")
    
    if len(predicted_events) >= 3:
        print("Fitting circle to predicted events...")
        pred_circle = fit_circle_least_squares(predicted_events[:, :2])  # Only x, y coordinates
        if pred_circle[0] is not None:
            xc, yc, r, rmse = pred_circle
            print(f"  Predicted circle: center=({xc:.1f}, {yc:.1f}), radius={r:.1f}px, RMSE={rmse:.2f}")
        else:
            print("  Failed to fit circle to predicted events")
    else:
        print("  Not enough predicted events for circle fitting (need ≥3)")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate plots
    print(f"\nGenerating plots...")
    output_path = os.path.join(OUTPUT_DIR, f"circle_extraction_{TIME_WINDOW[0]:.1f}s_to_{TIME_WINDOW[1]:.1f}s.png")
    fig = plot_events_with_circle(real_events, predicted_events, all_events, 
                                 real_circle, pred_circle, output_path)
    
    # Save circle parameters to file
    params_file = os.path.join(OUTPUT_DIR, f"circle_parameters_{TIME_WINDOW[0]:.1f}s_to_{TIME_WINDOW[1]:.1f}s.txt")
    with open(params_file, 'w') as f:
        f.write(f"Circle Extraction Results - Time Window {TIME_WINDOW[0]:.1f}s to {TIME_WINDOW[1]:.1f}s\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total events in window: {len(all_events):,}\n")
        f.write(f"Real events: {len(real_events):,}\n")
        f.write(f"Predicted events: {len(predicted_events):,}\n\n")
        
        if real_circle and real_circle[0] is not None:
            xc, yc, r, rmse = real_circle
            f.write(f"Real Events Circle:\n")
            f.write(f"  Center: ({xc:.3f}, {yc:.3f}) pixels\n")
            f.write(f"  Radius: {r:.3f} pixels\n")
            f.write(f"  RMSE: {rmse:.3f} pixels\n")
        else:
            f.write("Real Events Circle: Failed to fit\n")
        
        f.write("\n")
        
        if pred_circle and pred_circle[0] is not None:
            xc, yc, r, rmse = pred_circle
            f.write(f"Predicted Events Circle:\n")
            f.write(f"  Center: ({xc:.3f}, {yc:.3f}) pixels\n")
            f.write(f"  Radius: {r:.3f} pixels\n")
            f.write(f"  RMSE: {rmse:.3f} pixels\n")
        else:
            f.write("Predicted Events Circle: Failed to fit\n")
    
    print(f"Saved circle parameters: {params_file}")
    
    # Show plot
    plt.show()
    
    print(f"\nAnalysis complete! Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    analyze_circle_extraction()

