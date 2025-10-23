#!/usr/bin/env python3
"""
Extract (center_x, center_y, omega) from tracker positions with adaptive windows.

Improvements:
- Uses ONLY RANSAC inliers to estimate theta(t) and slope = omega.
- Exposes fit quality (theta_r2, inlier_rms, arc_deg, counts).
- Computes omega_dot (numerical gradient over the output timestamps) for later use.

Outputs:
- CSV with columns:
  timestamp, window_left, window_right, window_len_s,
  center_x, center_y, radius,
  omega_rad_s, omega_dot_rad_s2,
  num_points, inliers, inlier_rms, arc_deg, theta_r2
- PNG plot for quick QA.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- User paths ----------------
TRACKER_FILE = "AEB_tracker/perlin_1280hz_hand_outframe_2.csv"  # (time,x,y) columns
OUTPUT_CSV   = "center_omega_results.csv"
OUTPUT_PLOT  = "center_omega_plot.png"

# ---------------- Params ----------------
WINDOW_MS = 500.0     # nominal window (used as a reference)
STEP_MS   = 10.0      # stride between windows
MIN_POINTS = 300
R_MIN, R_MAX = 50.0, 500.0
INLIER_TOL_PX = 5.0
N_TRIALS = 200
MIN_INLIERS = 200
MAX_INLIER_RMS_PX = 2.5
MIN_ARC_DEG = 80.0

# Adaptive windowing:
START_FRAC   = 0.4    # start with 40% of WINDOW_MS
GROW_FACTOR  = 1.20   # grow by 20% when constraints aren’t met
MAX_W_MULT   = 3.0    # cap at 3x the nominal window

rng = np.random.default_rng(0)

# ---------------- IO ----------------
def fix_time(t):
    t = np.asarray(t, dtype=np.float64)
    tmax = float(np.max(t))
    if tmax > 1e12:
        t *= 1e-9   # ns -> s
    elif tmax > 1e6:
        t *= 1e-6   # us -> s
    elif tmax > 6e4:
        dt = np.median(np.diff(np.sort(t)))
        if dt > 1.0:
            t *= 1e-3  # ms -> s 
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

# ---------------- Geometry ----------------
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

    for _ in range(N_TRIALS):
        i0, i1, i2 = rng.choice(n, 3, replace=False)
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
            # compute RMS on inliers for quality
            if nin >= 3:
                rms = float(np.sqrt(np.mean(resid[inliers_mask]**2)))
            else:
                rms = np.inf
            best_inliers = nin
            best = (cx, cy, r, nin, rms, inliers_mask)

    return best  # (cx, cy, r, nin, rms, inliers_mask) or None

def line_fit_slope_r2(t_rel, y):
    """
    Simple linear regression slope + R^2 (no intercept removal beyond centering).
    """
    t_rel = np.asarray(t_rel, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(t_rel) < 2:
        return None, None
    # slope via polyfit + residuals
    p, residuals, _, _, _ = np.polyfit(t_rel, y, 1, full=True)
    slope = float(p[0])
    if residuals.size > 0:
        ss_res = float(residuals[0])
    else:
        # perfectly linear (or too few points)
        yhat = np.polyval(p, t_rel)
        ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return slope, r2

def get_omega_from_inliers(tw, xw, yw, cx, cy, inlier_mask):
    """
    Compute theta(t) ONLY on inliers; return slope=omega and arc coverage.
    """
    if inlier_mask is None or np.count_nonzero(inlier_mask) < 5:
        return None, None, None
    tw_i = tw[inlier_mask]
    xw_i = xw[inlier_mask]
    yw_i = yw[inlier_mask]

    theta = np.unwrap(np.arctan2(yw_i - cy, xw_i - cx))
    arc_deg = float(np.degrees(theta.max() - theta.min()))
    if arc_deg < MIN_ARC_DEG or len(tw_i) < 5:
        return None, arc_deg, None

    t_rel = tw_i - tw_i[0]
    slope, r2 = line_fit_slope_r2(t_rel, theta)
    if slope is None or not np.isfinite(slope):
        return None, arc_deg, None
    return float(slope), arc_deg, float(r2)

# ---------------- Adaptive extraction ----------------
def extract_center_omega(t, x, y):
    base_w = WINDOW_MS * 1e-3
    s = STEP_MS * 1e-3
    t_end = t[-1]

    results = []
    left = t[0]

    while left <= (t_end - base_w):
        w = max(base_w * START_FRAC, 0.05)           # start small
        w_max = base_w * MAX_W_MULT
        best_row = None

        while (left + w) <= t_end and w <= w_max:
            right = left + w
            start = np.searchsorted(t, left, side="left")
            end   = np.searchsorted(t, right, side="right")
            npts = end - start
            if npts < MIN_POINTS:
                w *= GROW_FACTOR
                continue

            tw = t[start:end]
            xw = x[start:end]
            yw = y[start:end]

            rans = ransac_circle(xw, yw)
            if rans is None:
                w *= GROW_FACTOR
                continue

            cx, cy, r, nin, rms, mask = rans
            if nin < MIN_INLIERS or rms > MAX_INLIER_RMS_PX:
                w *= GROW_FACTOR
                continue

            om, arc_deg, r2 = get_omega_from_inliers(tw, xw, yw, cx, cy, mask)
            if om is None or not np.isfinite(om) or arc_deg < MIN_ARC_DEG:
                w *= GROW_FACTOR
                continue

            best_row = {
                "timestamp": 0.5*(left + right),
                "window_left": float(left),
                "window_right": float(right),
                "window_len_s": float(w),
                "center_x": float(cx),
                "center_y": float(cy),
                "radius": float(r),
                "omega_rad_s": float(om),
                "num_points": int(npts),
                "inliers": int(nin),
                "inlier_rms": float(rms),
                "arc_deg": float(arc_deg),
                "theta_r2": float(r2 if r2 is not None else np.nan),
            }
            break  # stop growing once constraints satisfied

        if best_row is not None:
            results.append(best_row)
        # advance by fixed step regardless
        left += s

    return results

# ---------------- Plot ----------------
def plot_results(t, x, y, df):
    if df is None or df.empty:
        print("No results to plot.")
        return

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

# ---------------- Main ----------------
def main():
    print("Loading tracker data...")
    t, x, y = read_csv(TRACKER_FILE)
    print(f"Nominal window: {WINDOW_MS:.0f} ms (start {START_FRAC*100:.0f}%), step: {STEP_MS:.0f} ms")

    print("Extracting center & omega with adaptive window (inlier-only slope)...")
    results = extract_center_omega(t, x, y)
    if not results:
        print("No valid windows produced results.")
        return

    df = pd.DataFrame(results).sort_values("timestamp").reset_index(drop=True)

    # Compute omega_dot from the extracted omega series over (non-uniform) time
    om = df["omega_rad_s"].to_numpy(np.float64)
    tt = df["timestamp"].to_numpy(np.float64)
    # Guard if timestamps are constant (shouldn't be in practice)
    if np.allclose(tt, tt[0]):
        omdot = np.zeros_like(om)
    else:
        omdot = np.gradient(om, tt)
    df["omega_dot_rad_s2"] = omdot

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")

    med_cx = df["center_x"].median()
    med_cy = df["center_y"].median()
    med_om = df["omega_rad_s"].median()
    med_w  = df["window_len_s"].median()
    print(f"Center median: ({med_cx:.2f}, {med_cy:.2f}) px | "
          f"Omega median: {med_om:.6f} rad/s | "
          f"Median window: {med_w*1e3:.0f} ms")

    plot_results(t, x, y, df)
    print("Done.")

if __name__ == "__main__":
    main()
