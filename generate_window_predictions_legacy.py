#!/usr/bin/env python3
"""
Generate predictions for specific time windows with different dt values.
This script creates predictions only for the time windows used in visualization,
testing dt values from 1ms to 20ms.
"""

import time
import argparse
import numpy as np
import pandas as pd
import os
from pathlib import Path

# =============== Configuration ===============
REAL_EVENTS_FILE = "/home/sumit/anu_research/recording/new_data/perlin_1280hz_hand_outframe.csv"
TRACKER_CSV_FILE = "/home/sumit/anu_research/ego_motion/results_csv/perlin_1280hz_hand_outframe_combined.csv"

# Legacy version: Use small time windows for all dt values
# This matches what analyze_dt_and_tolerance.py expects
WINDOW_START_S = None  # Disable long window mode
WINDOW_DURATION_S = None  # Disable long window mode

# Legacy short windows (10ms each) - used by default
WINDOWS = [
    (5.000, 5.010),
    (8.200, 8.210),
    (9.000, 9.010),
]

# DT values to test (in milliseconds) - test all dt values for small windows
DT_RANGE_MS = (0, 20)
DT_STEP_MS = 1

# Output directory (using local path for small files)
OUTPUT_DIR = "./window_predictions"
OMEGA_SOURCE = "circlefit"  # "circlefit" | "theta_dot"
OMEGA_BIAS = 0.0

# Saving options
# Only save per-dt predictions and a single copy of real events per window.
# Disable combined to avoid huge writes and I/O errors on external drives.
SAVE_COMBINED = False
USE_COMPRESSED = False        # You can set to True to reduce file size

# =============== Motion Model ===============
def apply_rotation(x, y, cx, cy, omega, dt):
    """Rotate (x,y) about (cx,cy) by θ = ω·dt."""
    theta = omega * dt
    c = np.cos(theta)
    s = np.sin(theta)
    dx = x - cx
    dy = y - cy
    x_new = cx + c*dx - s*dy
    y_new = cy + s*dx + c*dy
    return x_new.astype(np.float32), y_new.astype(np.float32)

# =============== Data Loading ===============
def load_event_data_fast(path):
    """Load event data from CSV file"""
    df = pd.read_csv(path, names=["x","y","p","t"],
                     dtype={"x":np.float32,"y":np.float32,"p":np.float32,"t":np.float64})
    df["t"] = df["t"] * 1e-6  # μs -> s
    ev = df[["x","y","p","t"]].to_numpy(np.float32)

    if not np.all(ev[:-1,3] <= ev[1:,3]):
        ev = ev[np.argsort(ev[:,3])]

    # Map {-1,+1} to {0,1} if present
    uniq = np.unique(ev[:,2])
    if uniq.shape[0] == 2 and uniq.min() == -1.0 and uniq.max() == 1.0:
        ev[:,2] = (ev[:,2] + 1.0) * 0.5

    return ev

def load_tracker_series(path, source="circlefit"):
    """Load tracker data"""
    df = pd.read_csv(path)
    need = {"timestamp","center_x","center_y"}
    if not need.issubset(df.columns):
        raise ValueError("tracker CSV must have columns: timestamp, center_x, center_y")

    df = df.sort_values("timestamp")
    t_s = df["timestamp"].to_numpy(np.float64)
    cx_s = df["center_x"].to_numpy(np.float64)
    cy_s = df["center_y"].to_numpy(np.float64)

    if source == "theta_dot":
        col = "theta_dot_rad_s" if "theta_dot_rad_s" in df.columns else "omega_rad_s"
    else:
        col = "omega_circlefit_rad_s" if "omega_circlefit_rad_s" in df.columns else "omega_rad_s"
    if col not in df.columns:
        raise ValueError("No suitable omega column in tracker CSV.")
    om_s = df[col].to_numpy(np.float64)

    return t_s, cx_s, cy_s, om_s

# =============== Utility Functions ===============
def interp1(tq, tx, vx):
    """1D linear interpolation with edge hold."""
    return np.interp(tq, tx, vx, left=vx[0], right=vx[-1])

def extract_window_events(events, window):
    """Extract events within a specific time window"""
    t0, t1 = window
    mask = (events[:, 3] >= t0) & (events[:, 3] < t1)
    return events[mask]

# =============== Prediction Functions ===============
def predict_events_for_window(real_events, t_s, cx_s, cy_s, om_s, dt, omega_bias):
    """Generate predictions for events in a time window"""
    if len(real_events) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    x = real_events[:, 0].astype(np.float64)
    y = real_events[:, 1].astype(np.float64)
    p = real_events[:, 2].astype(np.float32)
    tt = real_events[:, 3].astype(np.float64)

    # Interpolate tracker data at event times
    cx = interp1(tt, t_s, cx_s)
    cy = interp1(tt, t_s, cy_s)
    om = interp1(tt, t_s, om_s)
    
    if omega_bias != 0.0:
        om = om + omega_bias

    # Apply rotation
    px, py = apply_rotation(x, y, cx, cy, om, dt)
    pt = (tt + dt).astype(np.float32)
    pp = (1.0 - p).astype(np.float32)  # flip polarity
    
    return np.column_stack([px, py, pp, pt])

def combine_window_events(real_events, pred_events):
    """Combine real and predicted events with flags"""
    if len(real_events) == 0 and len(pred_events) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    
    real_flag = np.zeros((len(real_events), 1), dtype=np.float32)
    pred_flag = np.ones((len(pred_events), 1), dtype=np.float32)
    
    if len(real_events) == 0:
        return np.column_stack([pred_events, pred_flag])
    elif len(pred_events) == 0:
        return np.column_stack([real_events, real_flag])
    else:
        combined = np.vstack([
            np.column_stack([real_events, real_flag]),
            np.column_stack([pred_events, pred_flag])
        ])
        return combined[np.argsort(combined[:, 3])]

# =============== Main Processing ===============
def process_window_predictions(output_dir: str, window_start_s: float = None, window_duration_s: float = None,
                               save_combined: bool = False, use_compressed: bool = False):
    """Generate predictions for configured time window(s) and dt values.
    If window_start_s is provided, generates a single long window of given duration.
    Otherwise, uses legacy short WINDOWS list.
    """
    print("Loading data...")
    real_events = load_event_data_fast(REAL_EVENTS_FILE)
    t_s, cx_s, cy_s, om_s = load_tracker_series(TRACKER_CSV_FILE, source=OMEGA_SOURCE)

    print(f"Loaded {len(real_events):,} real events")
    print(f"Tracker data: {len(t_s):,} time points")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Determine windows
    if window_start_s is not None and window_duration_s is not None:
        candidate_windows = [(float(window_start_s), float(window_start_s) + float(window_duration_s))]
    else:
        candidate_windows = WINDOWS

    # Generate dt values
    dt_values_ms = np.arange(DT_RANGE_MS[0], DT_RANGE_MS[1] + DT_STEP_MS, DT_STEP_MS)
    print(f"Testing {len(dt_values_ms)} dt values: {dt_values_ms[0]}ms to {dt_values_ms[-1]}ms")

    total_combinations = len(candidate_windows) * len(dt_values_ms)
    current_combination = 0

    for window_idx, window in enumerate(candidate_windows):
        t0, t1 = window
        print(f"\nProcessing window {window_idx + 1}/{len(candidate_windows)}: {t0:.3f}s to {t1:.3f}s")

        # Extract events for this window
        window_events = extract_window_events(real_events, window)
        print(f"  Found {len(window_events):,} real events in window")

        if len(window_events) == 0:
            print("  No events in window, skipping...")
            continue

        # Create window-specific output directory
        window_dir = os.path.join(output_dir, f"window_{window_idx + 1}_{t0:.3f}s_to_{t1:.3f}s")
        os.makedirs(window_dir, exist_ok=True)

        # Save real events once for this window (for analysis reuse)
        real_fname = os.path.join(window_dir, "real_events.npy" if not use_compressed else "real_events.npz")
        if use_compressed:
            np.savez_compressed(real_fname, real=window_events)
        else:
            np.save(real_fname, window_events)

        # Storage estimate per-array
        n_real = len(window_events)
        bytes_per_real_row = 4 * 4  # 4 float32 columns
        real_bytes = n_real * bytes_per_real_row

        for dt_ms in dt_values_ms:
            current_combination += 1
            dt_seconds = dt_ms / 1000.0

            print(f"  Processing dt={dt_ms:2d}ms ({current_combination:2d}/{total_combinations})", end=" ... ")

            # Generate predictions
            pred_events = predict_events_for_window(window_events, t_s, cx_s, cy_s, om_s,
                                                    dt_seconds, OMEGA_BIAS)

            # Save predictions per-dt
            pred_fname = os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms." + ("npz" if use_compressed else "npy"))
            if use_compressed:
                np.savez_compressed(pred_fname, pred=pred_events)
            else:
                np.save(pred_fname, pred_events)

            # Optionally, also save combined for compatibility
            if save_combined:
                combined_events = combine_window_events(window_events, pred_events)
                comb_fname = os.path.join(window_dir, f"combined_events_dt_{dt_ms:02d}ms." + ("npz" if use_compressed else "npy"))
                if use_compressed:
                    np.savez_compressed(comb_fname, combined=combined_events)
                else:
                    np.save(comb_fname, combined_events)

            print(f"saved {len(pred_events):,} predictions")

        # Report storage estimates for this window
        bytes_per_pred_row = 4 * 4  # 4 float32 columns
        pred_bytes_per_dt = n_real * bytes_per_pred_row
        total_pred_bytes_20 = pred_bytes_per_dt * len(dt_values_ms)
        total_bytes = real_bytes + total_pred_bytes_20
        def human(b):
            for unit in ["B","KB","MB","GB","TB"]:
                if b < 1024:
                    return f"{b:.2f} {unit}"
                b /= 1024
            return f"{b:.2f} PB"
        print(f"  Storage estimate (uncompressed): real {human(real_bytes)}, predictions per dt {human(pred_bytes_per_dt)}, all dts {human(total_pred_bytes_20)}; total {human(total_bytes)}")

    print(f"\nCompleted! All predictions saved to: {output_dir}")
    print(f"Total combinations processed: {current_combination}")

def main():
    """Main execution function"""
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Generate window predictions for dt sweep")
    parser.add_argument("--window-start", type=float, default=WINDOW_START_S, help="Start time (s) for long window; set to -1 to use legacy windows")
    parser.add_argument("--window-duration", type=float, default=WINDOW_DURATION_S, help="Duration (s) for long window")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory for predictions")
    parser.add_argument("--compressed", action="store_true", help="Save arrays as .npz compressed files")
    parser.add_argument("--save-combined", action="store_true", help="Also save combined real+pred files per dt")
    args = parser.parse_args()

    # Force legacy mode for small windows
    use_legacy = True

    print("=== Window Prediction Generator (Legacy Mode) ===")
    print(f"Using legacy WINDOWS list: {len(WINDOWS)} windows")
    print("Windows:", WINDOWS)
    ws = None
    wd = None

    print(f"DT range: {DT_RANGE_MS[0]}ms to {DT_RANGE_MS[1]}ms (step: {DT_STEP_MS}ms)")
    print(f"Omega source: {OMEGA_SOURCE}")
    print(f"Omega bias: {OMEGA_BIAS}")
    print(f"Output directory: {args.output_dir}")
    print(f"Compressed saving: {'ON' if args.compressed else 'OFF'}; Save combined: {'ON' if args.save_combined else 'OFF'}")

    process_window_predictions(output_dir=args.output_dir,
                               window_start_s=None if use_legacy else ws,
                               window_duration_s=None if use_legacy else wd,
                               save_combined=args.save_combined or SAVE_COMBINED,
                               use_compressed=args.compressed or USE_COMPRESSED)

    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.1f}s")

if __name__ == "__main__":
    main()
