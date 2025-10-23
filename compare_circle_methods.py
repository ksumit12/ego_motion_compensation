#!/usr/bin/env python3
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
TRACKER_FOLDER = "AEB_tracker"
OUTPUT_PLOT_FOLDER = "results_plots"
DATASET_PREFIX = "perlin_1280hz_hand_outframe_"

# Common parameters
WINDOW_S = 0.50
STEP_S = 0.05
MIN_POINTS = 300
R_MIN, R_MAX = 50.0, 500.0
MIN_ARC_DEG = 90.0
INLIER_TOL = 5.0
MAX_RMS = 3.0

# RANSAC specific
MIN_INLIERS = 200
MAX_RMS_RANSAC = 2.5
rng = np.random.default_rng(0)

def fix_time(t):
    """Convert timestamps to seconds if needed"""
    t = np.asarray(t, float)
    if not len(t): return t
    m = float(t.max())
    if m > 1e12: t *= 1e-9
    elif m > 1e6: t *= 1e-6
    elif m > 6e4 and np.median(np.diff(np.sort(t))) > 1.0: t *= 1e-3
    return t

def read_positions(path):
    """Read tracker CSV file and return sorted t, x, y arrays"""
    rows = []
    for line in open(path):
        parts = line.strip().split(",")
        if len(parts) < 2: continue
        try: 
            ts = float(parts[0])
            xy = parts[1].split()
            if len(xy) < 2: continue
            x, y = float(xy[0]), float(xy[1])
            rows.append((ts, x, y))
        except: continue
    
    if not rows: return np.array([]), np.array([]), np.array([])
    a = np.asarray(rows, float)
    t, x, y = fix_time(a[:,0]), a[:,1], a[:,2]
    o = np.argsort(t)
    return t[o], x[o], y[o]

# ============ LEAST SQUARES METHOD ============
def least_square_fit(x, y):
    """Fit circle using least squares method"""
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x*x + y*y)
    a, bcoef, c = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -a/2.0, -bcoef/2.0
    r = np.sqrt(max(cx*cx + cy*cy - c, 0.0))
    return float(cx), float(cy), float(r)

def fit_circle_least_square_robust(x, y):
    """Fit circle with outlier removal using least squares method"""
    if len(x) < 3: return None
    cx, cy, r = least_square_fit(x, y)
    inliers = np.abs(np.hypot(x - cx, y - cy) - r) <= INLIER_TOL
    if inliers.sum() < 3: return None
    
    cx2, cy2, r2 = least_square_fit(x[inliers], y[inliers])
    rms = float(np.sqrt(np.mean((np.hypot(x[inliers]-cx2, y[inliers]-cy2) - r2)**2)))
    return dict(cx=cx2, cy=cy2, r=r2, rms=rms, inliers=inliers)

# ============ RANSAC METHOD ============
def fit_circle_3points(x0, y0, x1, y1, x2, y2):
    """Fit circle through 3 points"""
    A = np.array([[2*(x1-x0), 2*(y1-y0)],
                  [2*(x2-x0), 2*(y2-y0)]], float)
    b = np.array([x1*x1+y1*y1-x0*x0-y0*y0,
                  x2*x2+y2*y2-x0*x0-y0*y0], float)
    if abs(np.linalg.det(A)) < 1e-12: return None
    cx, cy = np.linalg.solve(A, b)
    r = np.hypot(x0-cx, y0-cy)
    return cx, cy, r

def ransac_circle(x, y):
    """RANSAC circle fitting"""
    n = len(x)
    if n < 3: return None
    best = None
    best_inliers = -1
    
    for _ in range(200):
        i0, i1, i2 = rng.choice(n, 3, replace=False)
        res = fit_circle_3points(x[i0], y[i0], x[i1], y[i1], x[i2], y[i2])
        if res is None: continue
        
        cx, cy, r = res
        if not (R_MIN <= r <= R_MAX): continue
        
        distances = np.hypot(x - cx, y - cy)
        residuals = np.abs(distances - r)
        inliers = residuals <= INLIER_TOL
        num_inliers = int(inliers.sum())
        
        if num_inliers > best_inliers:
            rms = np.sqrt(np.mean(residuals[inliers]**2)) if num_inliers >= 3 else np.inf
            best_inliers = num_inliers
            best = dict(cx=cx, cy=cy, r=r, rms=rms, inliers=inliers, num_inliers=num_inliers)
    
    return best

def calc_omega(t, x, y, cx, cy):
    """Calculate angular velocity from trajectory"""
    theta = np.unwrap(np.arctan2(y - cy, x - cx))
    arc_deg = float(np.degrees(theta.max() - theta.min()))
    if len(t) < 5: return None, arc_deg
    omega = np.polyfit(t - t[0], theta, 1)[0]
    return float(omega), arc_deg

def extract_windows_least_square(t, x, y):
    """Extract windows using least squares method"""
    results = []
    left = t[0]
    
    while left + WINDOW_S <= t[-1]:
        right = left + WINDOW_S
        i0 = np.searchsorted(t, left, side="left")
        i1 = np.searchsorted(t, right, side="right")
        left += STEP_S
        
        if i1 - i0 < MIN_POINTS: continue
        tw, xw, yw = t[i0:i1], x[i0:i1], y[i0:i1]
        
        fit = fit_circle_least_square_robust(xw, yw)
        if fit is None: continue
        
        cx, cy, r, rms = fit["cx"], fit["cy"], fit["r"], fit["rms"]
        if not (R_MIN <= r <= R_MAX) or rms > MAX_RMS: continue
        
        omega, arc = calc_omega(tw, xw, yw, cx, cy)
        if omega is None or not np.isfinite(omega) or arc < MIN_ARC_DEG: continue
        
        results.append({
            'center_x': float(cx), 'center_y': float(cy), 'radius': float(r),
            'omega': float(omega), 'rms': float(rms), 'method': 'least_square'
        })
    
    return pd.DataFrame(results)

def extract_windows_ransac(t, x, y):
    """Extract windows using RANSAC method"""
    results = []
    left = t[0]
    
    while left + WINDOW_S <= t[-1]:
        right = left + WINDOW_S
        i0 = np.searchsorted(t, left, side="left")
        i1 = np.searchsorted(t, right, side="right")
        left += STEP_S
        
        if i1 - i0 < MIN_POINTS: continue
        tw, xw, yw = t[i0:i1], x[i0:i1], y[i0:i1]
        
        fit = ransac_circle(xw, yw)
        if fit is None: continue
        
        cx, cy, r, rms, num_inliers = fit["cx"], fit["cy"], fit["r"], fit["rms"], fit["num_inliers"]
        if not (R_MIN <= r <= R_MAX) or rms > MAX_RMS_RANSAC or num_inliers < MIN_INLIERS: continue
        
        omega, arc = calc_omega(tw, xw, yw, cx, cy)
        if omega is None or not np.isfinite(omega) or arc < MIN_ARC_DEG: continue
        
        results.append({
            'center_x': float(cx), 'center_y': float(cy), 'radius': float(r),
            'omega': float(omega), 'rms': float(rms), 'method': 'ransac'
        })
    
    return pd.DataFrame(results)

def plot_individual_comparison(t, x, y, df_least_square, df_ransac, file_name, out_png):
    """Create comparison plot for individual file"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    methods = [('least_square', df_least_square, 'blue'), ('ransac', df_ransac, 'red')]
    
    for i, (method, df, color) in enumerate(methods):
        ax = axes[i]
        
        if df.empty:
            ax.text(0.5, 0.5, f'No valid {method} results', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{method.upper()} - No Results')
            continue
        
        # Plot trajectory
        ax.plot(x, y, lw=0.6, alpha=0.3, color='gray', label="trajectory")
        
        # Get median center and radius
        cxm, cym = df["center_x"].median(), df["center_y"].median()
        rm = df["radius"].median()
        
        # Plot center
        ax.plot(cxm, cym, 'o', color=color, ms=12, mew=3, markerfacecolor='white',
                label=f"center ({cxm:.1f},{cym:.1f})")
        
        # Plot circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x, circle_y = cxm + rm * np.cos(theta), cym + rm * np.sin(theta)
        ax.plot(circle_x, circle_y, "-", color=color, lw=3, alpha=0.8,
                label=f"circle (r={rm:.1f})")
        
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'{method.upper()} Method - {file_name}')
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)

def plot_combined_comparison(all_data_least_square, all_data_ransac, dataset_name, out_png):
    """Create combined comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    methods = [('LEAST SQUARES', all_data_least_square, axes[0]), ('RANSAC', all_data_ransac, axes[1])]
    
    for method_name, all_data, ax in methods:
        if not all_data:
            ax.text(0.5, 0.5, f'No {method_name} results', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            continue
        
        for i, (file_data, file_name) in enumerate(all_data):
            if file_data['df'].empty: continue
            
            color = colors[i % len(colors)]
            df = file_data['df']
            cxm, cym, rm = df["center_x"].median(), df["center_y"].median(), df["radius"].median()
            
            # Center point
            ax.plot(cxm, cym, 'o', color=color, ms=10, mew=2, markerfacecolor='white',
                    label=f"{file_name} center ({cxm:.1f},{cym:.1f})")
            
            # Circle outline
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x, circle_y = cxm + rm * np.cos(theta), cym + rm * np.sin(theta)
            ax.plot(circle_x, circle_y, "-", color=color, lw=3, alpha=0.9,
                    label=f"{file_name} circle (r={rm:.1f})")
        
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(f"{method_name} Method - {dataset_name}")
        ax.set_xlabel("x (pixels)")
        ax.set_ylabel("y (pixels)")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Combined comparison plot saved: {out_png}")

def main():
    """Main processing function"""
    os.makedirs(OUTPUT_PLOT_FOLDER, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(TRACKER_FOLDER, f"{DATASET_PREFIX}*.csv")))
    if not files:
        print("No files found")
        return
    
    all_data_least_square, all_data_ransac = [], []
    
    for file_path in files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"-> {base_name}")
        
        t, x, y = read_positions(file_path)
        if len(t) < MIN_POINTS:
            print("   too few points")
            continue
        
        # Extract using both methods
        df_least_square = extract_windows_least_square(t, x, y)
        df_ransac = extract_windows_ransac(t, x, y)
        
        print(f"   Least Squares: {len(df_least_square)} windows, RANSAC: {len(df_ransac)} windows")
        
        # Store for combined plots
        all_data_least_square.append(({'df': df_least_square, 'x': x, 'y': y}, base_name))
        all_data_ransac.append(({'df': df_ransac, 'x': x, 'y': y}, base_name))
        
        # Create individual comparison plot
        individual_plot_path = os.path.join(OUTPUT_PLOT_FOLDER, f"{base_name}_comparison.png")
        plot_individual_comparison(t, x, y, df_least_square, df_ransac, base_name, individual_plot_path)
        print(f"   Individual comparison saved: {individual_plot_path}")
    
    # Create combined comparison plot
    combined_plot_path = os.path.join(OUTPUT_PLOT_FOLDER, f"{DATASET_PREFIX}method_comparison.png")
    plot_combined_comparison(all_data_least_square, all_data_ransac, DATASET_PREFIX.rstrip("_"), combined_plot_path)
    
    print(f"\nComparison complete!")
    print(f"Individual plots: {len(files)} files processed")
    print(f"Combined plot: {combined_plot_path}")

if __name__ == "__main__":
    main()
