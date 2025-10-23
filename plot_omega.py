#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _coerce_numeric(series):
    """Convert to numeric, stripping stray commas if needed."""
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def load_omega_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV and return a DataFrame with columns: time, omega, omega_dot.
    Tries (1) normal CSV with headers, (2) whitespace-delimited without headers.
    """
    # --- First attempt: normal CSV with headers ---
    try:
        df = pd.read_csv(csv_path)
        lower_cols = {c.lower(): c for c in df.columns}
        if {"time", "omega", "omega_dot"}.issubset(lower_cols.keys()):
            out = pd.DataFrame({
                "time": _coerce_numeric(df[lower_cols["time"]]),
                "omega": _coerce_numeric(df[lower_cols["omega"]]),
                "omega_dot": _coerce_numeric(df[lower_cols["omega_dot"]]),
            })
            return out.dropna()
    except Exception:
        pass

    # --- Second attempt: whitespace-delimited, no headers ---
    # Your file looks like this case: 11 columns; time in col 0 (with comma),
    # omega in col 5, omega_dot in col 6.
    df2 = pd.read_csv(csv_path, delim_whitespace=True, header=None, engine="python")

    # Heuristic mapping (works for the sample you shared)
    # If there are enough columns, assume:
    # 0: time, 5: omega, 6: omega_dot
    if df2.shape[1] >= 7:
        time = _coerce_numeric(df2.iloc[:, 0])
        omega = _coerce_numeric(df2.iloc[:, 5])
        omega_dot = _coerce_numeric(df2.iloc[:, 6])
        out = pd.DataFrame({"time": time, "omega": omega, "omega_dot": omega_dot})
        return out.dropna()

    raise ValueError(
        "Could not find time/omega/omega_dot columns. "
        "Please ensure the CSV has headers (time,omega,omega_dot) or "
        "is whitespace-delimited with time in col0, omega in col5, omega_dot in col6."
    )


def main():
    ap = argparse.ArgumentParser(description="Plot omega and omega_dot over time from a local CSV.")
    ap.add_argument("csv", type=Path, help="Path to CSV file")
    ap.add_argument("--title-prefix", type=str, default="", help="Optional prefix for plot titles")
    ap.add_argument("--show", action="store_true", help="Show interactive windows")
    args = ap.parse_args()

    df = load_omega_csv(args.csv)
    df = df.sort_values("time")

    # Basic sanity filter to drop obvious junk (optional)
    # e.g., remove infs and super-NaNs already handled; you can also clip extremes if needed
    for c in ("omega", "omega_dot"):
        df = df[np.isfinite(df[c])]

    stem = args.csv.with_suffix("").name
    out_dir = args.csv.parent

    # --- Plot omega ---
    plt.figure()
    plt.plot(df["time"].values, df["omega"].values)
    plt.xlabel("Time (s)")
    plt.ylabel("Omega (rad/s)")
    title = (args.title_prefix + " " if args.title_prefix else "") + "Angular Velocity (ω) vs Time"
    plt.title(title)
    plt.grid(True, linewidth=0.3)
    omega_png = out_dir / f"{stem}_omega.png"
    plt.savefig(omega_png, dpi=200, bbox_inches="tight")

    # --- Plot omega_dot ---
    plt.figure()
    plt.plot(df["time"].values, df["omega_dot"].values)
    plt.xlabel("Time (s)")
    plt.ylabel("Omega dot (rad/s²)")
    title2 = (args.title_prefix + " " if args.title_prefix else "") + "Angular Acceleration (ω̇) vs Time"
    plt.title(title2)
    plt.grid(True, linewidth=0.3)
    omega_dot_png = out_dir / f"{stem}_omega_dot.png"
    plt.savefig(omega_dot_png, dpi=200, bbox_inches="tight")

    if args.show:
        plt.show()

    print(f"Saved: {omega_png}")
    print(f"Saved: {omega_dot_png}")


if __name__ == "__main__":
    main()
