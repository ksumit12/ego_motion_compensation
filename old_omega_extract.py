#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRACKER_FILE = "AEB_tracker/perlin_1280hz_hand_outframe_2.csv"
OUTPUT_CSV = "center_omega_results.csv"
OUTPUT_PLOT = "center_omega_plot.png"

WINDOW_MS = 500.0
STEP_MS = 10.0
MIN_POINTS = 300
R_MIN, R_MAX = 50.0, 500.0
INLIER_TOL_PX = 5.0
N_TRIALS = 200
MIN_INLIERS = 200
MAX_INLIER_RMS_PX = 2.5
MIN_ARC_DEG = 80.0

def fix_time(t):
    t = np.asarray(t, dtype=np.float64)
    tmax = float(np.max(t))
    if tmax > 1e12:
        t *= 1e-9
    elif tmax > 1e6:
        t *= 1e-6
    elif tmax > 6e4:
        dt = np.median(np.diff(np.sort(t)))
        if dt > 1.0:
            t *= 1e-3
    return t

def read_csv(path):
    try:
        df = pd.read_csv(path)
        t = df.iloc[:, 0].to_numpy(np.float64)
        x = df.iloc[:, 1].to_numpy(np.float64)
        y = df.iloc[:, 2].to_numpy(np.float64)
    except Exception:
        parsed = []
        with open(path, "r") as f:
            for line in f:
                vals = []
                for token in line.strip().split(","):
                    for s in token.split():
                        try:
                            vals.append(float(s))
                        except:
                            pass
                if len(vals) >= 3:
                    parsed.append(vals[:3])
        
        if not parsed:
            raise ValueError("Could not parse any numeric triplets from file.")
        
        arr = np.asarray(parsed, dtype=np.float64)
        t, x, y = arr[:, 0], arr[:, 1], arr[:, 2]
    
    t = fix_time(t)
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    print(f"Loaded {len(t)} points: {t[0]:.3f}s to {t[-1]:.3f}s")
    return t, x, y

def fit_circle(x0, y0, x1, y1, x2, y2):
    A = np.array([[2*(x1-x0), 2*(y1-y0)], 
                  [2*(x2-x0), 2*(y2-y0)]], dtype=np.float64)
    b = np.array([x1*x1 + y1*y1 - x0*x0 - y0*y0, 
                  x2*x2 + y2*y2 - x0*x0 - y0*y0], dtype=np.float64)
    
    if abs(np.linalg.det(A)) < 1e-12:
        return None
    
    cx, cy = np.linalg.solve(A, b)
    r = np.hypot(x0 - cx, y0 - cy)
    return cx, cy, r

def ransac_circle(x, y):
    n = len(x)
    if n < 3:
        return None
    
    best = None
    best_inliers = -1
    best_rms = np.inf
    
    for _ in range(N_TRIALS):
        i0, i1, i2 = np.random.choice(n, 3, replace=False)
        res = fit_circle(x[i0], y[i0], x[i1], y[i1], x[i2], y[i2])
        if res is None:
            continue
        
        cx, cy, r = res
        if not (R_MIN <= r <= R_MAX):
            continue
        
        d = np.hypot(x - cx, y - cy)
        resid = np.abs(d - r)
        inliers_mask = resid <= INLIER_TOL_PX
        nin = int(np.count_nonzero(inliers_mask))
        
        if nin > best_inliers:
            best_inliers = nin
            rms = float(np.sqrt(np.mean(resid[inliers_mask]**2))) if nin >= 3 else np.inf
            best_rms = rms
            best = (cx, cy, r, nin, rms, inliers_mask)
    
    return best

def get_omega(tw, xw, yw, cx, cy):
    theta = np.unwrap(np.arctan2(yw - cy, xw - cx))
    arc_deg = float(np.degrees(theta.max() - theta.min()))
    
    if arc_deg < MIN_ARC_DEG or len(tw) < 5:
        return None, arc_deg
    
    t0 = tw[0]
    t_rel = tw - t0
    slope = np.polyfit(t_rel, theta, 1)[0]
    return float(slope), arc_deg

def extract_center_omega(t, x, y):
    w = WINDOW_MS * 1e-3
    s = STEP_MS * 1e-3
    n = len(t)
    results = []
    
    start = end = 0
    left = t[0]
    t_end = t[-1]
    
    while left <= (t_end - w):
        right = left + w
        
        while end < n and t[end] < right:
            end += 1
        while start < end and t[start] < left:
            start += 1
        
        npts = end - start
        if npts >= MIN_POINTS:
            xw, yw, tw = x[start:end], y[start:end], t[start:end]
            rans = ransac_circle(xw, yw)
            
            if rans is not None:
                cx, cy, r, nin, rms, mask = rans
                if nin >= MIN_INLIERS and rms <= MAX_INLIER_RMS_PX:
                    om, arc_deg = get_omega(tw, xw, yw, cx, cy)
                    if om is not None and np.isfinite(om):
                        results.append({
                            "timestamp": 0.5*(left + right),
                            "center_x": float(cx),
                            "center_y": float(cy),
                            "radius": float(r),
                            "omega_rad_s": float(om),
                            "num_points": int(npts),
                            "inliers": int(nin),
                            "inlier_rms": float(rms),
                            "arc_deg": float(arc_deg),
                        })
        
        left += s
    
    return results

def plot_results(t, x, y, results):
    if not results:
        print("No results to plot.")
        return
    
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].scatter(x, y, s=1, alpha=0.25)
    axes[0, 0].set_aspect('equal')
    axes[0, 0].set_title("Raw blob positions")
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(df["timestamp"], df["center_x"], label="center_x")
    axes[0, 1].plot(df["timestamp"], df["center_y"], label="center_y")
    axes[0, 1].legend()
    axes[0, 1].set_title("Center over time")
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].plot(df["timestamp"], df["omega_rad_s"], lw=1.2, label="omega")
    med = np.median(df["omega_rad_s"])
    axes[1, 0].axhline(med, ls="--", color="r", label=f"median {med:.3f} rad/s")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_title("Angular velocity over time")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].hist(df["omega_rad_s"], bins=24, edgecolor="k", alpha=0.85)
    axes[1, 1].axvline(np.median(df["omega_rad_s"]), ls="--", color="r")
    axes[1, 1].set_title("Omega distribution")
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=140, bbox_inches="tight")
    print(f"Saved plot: {OUTPUT_PLOT}")
    plt.close(fig)

def main():
    print("Loading tracker data...")
    t, x, y = read_csv(TRACKER_FILE)
    
    print(f"Window: {WINDOW_MS:.0f} ms, step: {STEP_MS:.0f} ms")
    print("Extracting center & omega...")
    results = extract_center_omega(t, x, y)
    
    if not results:
        print("No valid windows produced results.")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    
    med_cx = df["center_x"].median()
    med_cy = df["center_y"].median()
    med_om = df["omega_rad_s"].median()
    print(f"Center median: ({med_cx:.2f}, {med_cy:.2f}) px")
    print(f"Omega median: {med_om:.6f} rad/s")
    
    plot_results(t, x, y, results)
    print("Done.")

if __name__ == "__main__":
    main()