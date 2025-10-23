#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

def ema(x, alpha):
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="center_omega_results.csv")
    ap.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing factor (0..1, higher=smoother)")
    ap.add_argument("--dt_ms", type=float, default=2.0, help="uniform resample step in ms")
    ap.add_argument("--rms_max", type=float, default=2.5, help="max inlier_rms to keep")
    ap.add_argument("--arc_min", type=float, default=70.0, help="min arc_deg to keep")
    ap.add_argument("--clip_lo", type=float, default=1.0, help="omega lower clip percentile")
    ap.add_argument("--clip_hi", type=float, default=99.0, help="omega upper clip percentile")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # basic quality filter
    m = (
        np.isfinite(df["omega_rad_s"]) &
        np.isfinite(df["center_x"]) &
        np.isfinite(df["center_y"]) &
        (df["inlier_rms"] <= args.rms_max) &
        (df["arc_deg"] >= args.arc_min)
    )
    df = df.loc[m].sort_values("timestamp").reset_index(drop=True)
    if len(df) < 10:
        raise SystemExit("Not enough good rows after filtering.")

    # percentile clip on omega to remove spikes
    lo = np.percentile(df["omega_rad_s"], args.clip_lo)
    hi = np.percentile(df["omega_rad_s"], args.clip_hi)
    omg = df["omega_rad_s"].clip(lo, hi).to_numpy()
    cx  = df["center_x"].to_numpy()
    cy  = df["center_y"].to_numpy()
    tt  = df["timestamp"].to_numpy()

    # smooth (EMA)
    omg_s = ema(omg, args.alpha)
    cx_s  = ema(cx,  args.alpha*0.7)   # slightly less smoothing
    cy_s  = ema(cy,  args.alpha*0.7)

    # resample to uniform grid
    t0, t1 = tt[0], tt[-1]
    dt = args.dt_ms * 1e-3
    tu = np.arange(t0, t1 + 0.5*dt, dt)

    # linear interpolation to the uniform grid
    cx_u  = np.interp(tu, tt, cx_s)
    cy_u  = np.interp(tu, tt, cy_s)
    omg_u = np.interp(tu, tt, omg_s)

    out = pd.DataFrame({
        "timestamp": tu,
        "center_x": cx_u,
        "center_y": cy_u,
        "omega_rad_s": omg_u
    })
    out.to_csv("center_omega_smoothed.csv", index=False)
    print(f"Saved center_omega_smoothed.csv with {len(out)} rows "
          f"(dt ≈ {args.dt_ms:.1f} ms, EMA α={args.alpha})")

if __name__ == "__main__":
    main()
