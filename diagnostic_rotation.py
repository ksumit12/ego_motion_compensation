#!/usr/bin/env python3
"""
Plot omega(t) using theta_dot from the tracker CSV.

Tracker CSV column order (no header):
[ts, px, py, vx, vy, theta, theta_dot, l1, l2, d1, d2]

Usage:
  python plot_theta_dot_as_omega.py --tracker AEB_tracker/perlin_1280hz_hand_outframe_2.csv
  # If theta_dot is in deg/s, add --deg
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fix_time_units(t):
    """Convert timestamps to seconds if they look like ns/us/ms."""
    t = np.asarray(t, dtype=np.float64)
    tmax = float(np.nanmax(t))
    if tmax > 1e12:   # ns
        t *= 1e-9
    elif tmax > 1e6:  # us
        t *= 1e-6
    elif tmax > 6e4:  # likely ms
        dt = np.median(np.diff(np.sort(t)))
        if dt > 1.0:
            t *= 1e-3
    return t

def load_tracker_theta_dot(path, in_degrees=False):
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 7:
        raise ValueError(f"Expected >= 7 columns, got {df.shape[1]}")
    df.columns = ["ts","px","py","vx","vy","theta","theta_dot","l1","l2","d1","d2"][:df.shape[1]]
    t = fix_time_units(df["ts"].to_numpy(np.float64))
    w = df["theta_dot"].to_numpy(np.float64)
    if in_degrees:
        w = np.deg2rad(w)  # deg/s -> rad/s
    return t, w

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", required=True, help="Path to tracker CSV (11-col, no header)")
    ap.add_argument("--deg", action="store_true", help="Interpret theta_dot as deg/s (convert to rad/s)")
    args = ap.parse_args()

    t, omega = load_tracker_theta_dot(args.tracker, in_degrees=args.deg)

    print(f"Samples: {len(t)} | time: {t.min():.3f}s → {t.max():.3f}s")
    print(f"omega stats (rad/s): mean={omega.mean():.3f}, median={np.median(omega):.3f}, "
          f"min={omega.min():.3f}, max={omega.max():.3f}")

    plt.figure(figsize=(10,5))
    plt.plot(t, omega, lw=1)
    plt.title("Angular Velocity ω(t) from tracker theta_dot")
    plt.xlabel("Time (s)")
    plt.ylabel("ω (rad/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
