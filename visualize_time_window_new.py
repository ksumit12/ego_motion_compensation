#!/usr/bin/env python3
"""
Event cancellation visualization script (rewritten & extended).
Analyzes and visualizes ego-motion cancellation results across different time windows.

Key upgrades:
- True temporal-window cancellation (|t - (t_i+Δt)| <= ε_t), no hard bin boundary misses.
- Optional phase-aligned, overlapped bins (if bin-mode is retained).
- Mutual nearest-neighbor (MNN) matching to reduce spurious pairs.
- Adaptive spatial tolerance as a function of radius & timing (optional).
- Polarity modes preserved: "opposite" | "equal" | "ignore".
"""

import os
import time
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import matplotlib.gridspec as gridspec

# --------------------------------- Headless backend ---------------------------------
try:
    if not os.environ.get('DISPLAY'):
        matplotlib.use("Agg")
except Exception:
    pass

# =============================== Configuration (top) ================================
COMBINED_PATH = "./combined_events_with_predictions.npy"

# --- Matching core switches ---
CANCEL_MODE = "window"   # "window" (recommended) | "bin"
MUTUAL_NN   = True       # enforce mutual nearest neighbors (reduces spurious matches)
ADAPTIVE_SPATIAL_TOL = True  # tolerance grows with radius & timing uncertainty
SAFE_OVERLAP_BINS = True     # if CANCEL_MODE == "bin": use 50% overlapped bins to reduce edge misses
PHASE_ALIGN_BINS  = True     # align bin centers with (t + DT_SECONDS)

# --- Motion / timing parameters (keep in sync with predictor) ---
DT_SECONDS = 0.002       # Δt used when generating predicted events (s)
EPS_T      = 0.003       # temporal tolerance (s) for "window" mode; if "bin", BIN_MS ~= 2*EPS_T*1e3
BIN_MS     = 5.0         # bin width (ms) if using "bin" mode (effective temporal tol ~ BIN_MS/2)

# --- Spatial tolerance params ---
R_PIX      = 2.0         # base spatial tolerance (px), used if ADAPTIVE_SPATIAL_TOL=False
SIGMA_T    = 0.00008     # timing jitter proxy (s), ~80 µs as a starting point
DELTA_C    = 0.3         # miscenter allowance (pixels)
K_ADAPT    = 2.0         # safety factor for adaptive gate

# --- Polarity handling ---
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# --- Rasterization (unchanged from your original) ---
USE_BILINEAR_INTERP = True  # True -> bilinear splatting; False -> nearest neighbor
IMG_W, IMG_H = 1280, 720

# --- Plotting ---
HIST_BINS = 100

# --- Disc ROI (for overlays & optional analysis) ---
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS   = 250  # px

# --- Time windows to analyze ---
WINDOWS = [
    (5.000, 5.010),
    (8.200, 8.210),
    (9.000, 9.010),
]

# --- Outputs ---
OUTPUT_DIR = "./main_results"
OUTPUT_FILES = {
    'scatter': "cancellation_visualization.png",
    'images': "per_pixel_images.png",
    'images_gray': "per_pixel_images_gray.png",
    'histogram': "per_pixel_hist.png",
    'surface': "per_pixel_surfaces_signed.png",
    'tolerance': "tolerance_analysis.png"
}

# =============================== Utility / IO helpers ===============================
def load_combined(path):
    arr = np.load(path, mmap_mode="r")
    if not np.all(arr[:-1, 3] <= arr[1:, 3]):
        arr = arr[np.argsort(arr[:, 3])]
    print(f"Loaded {len(arr):,} events "
          f"(real={int(np.sum(arr[:,4]==0.0)):,}, pred={int(np.sum(arr[:,4]==1.0)):,})")
    return arr

def time_edges(tmin, tmax, bin_ms):
    w = bin_ms * 1e-3
    n = int(np.ceil((tmax - tmin) / w)) + 1
    return tmin + np.arange(n+1) * w

def circle_mask(x, y, cx, cy, r, scale=1.05):
    return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

def events_in_window(events, t0, t1):
    return (events[:, 3] >= t0) & (events[:, 3] < t1)

# ================================ Polarity predicates ===============================
def check_polarity_match(real_p, pred_p):
    if POLARITY_MODE == "ignore":
        return True
    if POLARITY_MODE == "equal":
        return real_p == pred_p
    return real_p != pred_p  # "opposite"

def _signed_p(p):
    return np.where(p > 0.5, 1, -1).astype(np.float32)

# =============================== Rasterization (images) =============================
def _raster_bilinear(width, height, events):
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    x = events[:, 0].astype(np.float32)
    y = events[:, 1].astype(np.float32)
    s = _signed_p(events[:, 2])
    x0 = np.floor(x).astype(np.int32); y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    dx = x - x0; dy = y - y0
    w00 = (1.0 - dx) * (1.0 - dy); w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy;         w11 = dx * dy
    img = np.zeros((height, width), dtype=np.float32)
    m00 = (x0 >= 0) & (x0 < width) & (y0 >= 0) & (y0 < height)
    m10 = (x1 >= 0) & (x1 < width) & (y0 >= 0) & (y0 < height)
    m01 = (x0 >= 0) & (x0 < width) & (y1 >= 0) & (y1 < height)
    m11 = (x1 >= 0) & (x1 < width) & (y1 >= 0) & (y1 < height)
    np.add.at(img, (y0[m00], x0[m00]), s[m00] * w00[m00])
    np.add.at(img, (y0[m10], x1[m10]), s[m10] * w10[m10])
    np.add.at(img, (y1[m01], x0[m01]), s[m01] * w01[m01])
    np.add.at(img, (y1[m11], x1[m11]), s[m11] * w11[m11])
    return img

def _raster_nearest(width, height, events):
    if len(events) == 0:
        return np.zeros((height, width), dtype=np.float32)
    x = events[:, 0].astype(np.int32)
    y = events[:, 1].astype(np.int32)
    s = _signed_p(events[:, 2])
    m = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    img = np.zeros((height, width), dtype=np.float32)
    np.add.at(img, (y[m], x[m]), s[m])
    return img

def create_per_pixel_count_image(width, height, events):
    return _raster_bilinear(width, height, events) if USE_BILINEAR_INTERP else _raster_nearest(width, height, events)

def normalize_images_for_display(real_image, predicted_image, combined_image):
    real_max = np.abs(real_image).max()
    predicted_max = np.abs(predicted_image).max()
    combined_max = np.abs(combined_image).max()
    overall_max = max(real_max, predicted_max, combined_max, 1)
    def norm(im): return np.clip((im / (2.0 * overall_max)) + 0.5, 0.0, 1.0)
    return norm(real_image), norm(predicted_image), norm(combined_image), int(overall_max)

def build_window_images(combined_events, time_window, image_width, image_height):
    t0, t1 = time_window
    m = (combined_events[:, 3] >= t0) & (combined_events[:, 3] < t1)
    win = combined_events[m]
    real_ev = win[win[:, 4] == 0.0][:, :3]
    pred_ev = win[win[:, 4] == 1.0][:, :3]
    all_ev  = win[:, :3]
    img_r = create_per_pixel_count_image(image_width, image_height, real_ev)
    img_p = create_per_pixel_count_image(image_width, image_height, pred_ev)
    img_c = create_per_pixel_count_image(image_width, image_height, all_ev)
    return img_r, img_p, img_c, len(real_ev), len(pred_ev)

# ============================== Spatial tolerance models ============================
def adaptive_r_pix_for_points(xy, cx, cy, omega_abs, sigma_t=SIGMA_T, dc=DELTA_C, k=K_ADAPT):
    """
    Per-event spatial tolerance based on radius r and timing uncertainty:
        r_pix ≈ k * (r * |omega| * sigma_t + dc)
    Returns an array of per-event tolerances (float32).
    """
    if not ADAPTIVE_SPATIAL_TOL:
        return np.full((xy.shape[0],), float(R_PIX), dtype=np.float32)
    dx = xy[:, 0] - cx; dy = xy[:, 1] - cy
    r  = np.hypot(dx, dy)
    gate = k * (r * omega_abs * sigma_t + dc)
    gate = np.clip(gate, R_PIX * 0.6, R_PIX * 4.0)  # keep within sane bounds
    return gate.astype(np.float32)

# ============================== Matching primitives =================================
def _mnn_filter(real_xy, pred_xy, pairs_r2p, pairs_p2r):
    """
    Mutual nearest neighbor filtering:
    - pairs_r2p: (ri -> pi) indices or -1
    - pairs_p2r: (pi -> ri) indices or -1
    Keep only (ri, pi) where pi == pairs_r2p[ri] and ri == pairs_p2r[pi].
    Returns a boolean mask over real indices: True if mutually matched.
    """
    keep = np.zeros(len(real_xy), dtype=bool)
    valid_r = np.where(pairs_r2p >= 0)[0]
    for ri in valid_r:
        pi = pairs_r2p[ri]
        if pi >= 0 and pairs_p2r[pi] == ri:
            keep[ri] = True
    return keep

def _query_with_variable_radius(tree, q_xy, r_per_query):
    """
    Query a KD-tree with per-query distance thresholds.
    We use a two-stage approach:
        1) Fixed upper bound = max(r_per_query) to get neighbors superset.
        2) Filter by actual per-query radii.
    Returns:
        idx_best (int32 array): index of nearest neighbor within that query's radius, or -1.
        dist_best (float32 array): distance of that nearest neighbor (inf if none).
    """
    R = float(np.max(r_per_query)) if len(r_per_query) else 0.0
    if R <= 0.0 or len(q_xy) == 0:
        return np.full((len(q_xy),), -1, dtype=np.int32), np.full((len(q_xy),), np.inf, dtype=np.float32)
    # Query with large cap (k=1 is enough since we only need nearest)
    dist, idx = tree.query(q_xy, k=1, distance_upper_bound=R)
    # Filter by actual per-query radius
    good = dist <= r_per_query
    idx_best = np.where(good, idx.astype(np.int32), -1)
    dist_best = np.where(good, dist.astype(np.float32), np.inf)
    return idx_best, dist_best

# ============================== Cancellation: window mode ===========================
def cancel_with_temporal_window(combined, dt_seconds, eps_t_seconds, cx=DISC_CENTER_X, cy=DISC_CENTER_Y,
                                omega_abs_hint=2*math.pi,  # fallback 1 rev/s
                                mutual_nn=MUTUAL_NN):
    """
    Exact temporal gate: for each predicted event at t*, only consider real events with |t - t*| <= eps_t.
    Spatial gate: per-event, optionally adaptive to radius and timing uncertainty.
    MNN: mutual nearest-neighbor to reduce spurious pairs.
    Returns residual_real, residual_pred, total_matches
    """
    real = combined[combined[:, 4] == 0.0]
    pred = combined[combined[:, 4] == 1.0]
    if len(real) == 0 or len(pred) == 0:
        return real, pred, 0

    # Both time-sorted already (load_combined ensures that)
    # Build index pointers for temporal slicing
    j0 = 0
    Np = len(pred); Nr = len(real)
    used_real = np.zeros(Nr, dtype=bool)
    used_pred = np.zeros(Np, dtype=bool)

    # Precompute per-predicted-event spatial tolerance (adaptive)
    r_pix_pred = adaptive_r_pix_for_points(pred[:, :2], cx, cy, omega_abs_hint)

    # One pass over predicted events; do batch KDTree in small windows to amortize cost
    total_matches = 0
    block_size = 20000  # trade-off for performance
    for i0 in range(0, Np, block_size):
        i1 = min(i0 + block_size, Np)
        pred_block = pred[i0:i1]
        r_pix_block = r_pix_pred[i0:i1]

        # For each predicted in block, find real window: (t* - eps_t, t* + eps_t)
        # Compute aggregate [tmin, tmax] to slice once
        tstars = pred_block[:, 3]
        tmin = float(np.min(tstars) - eps_t_seconds)
        tmax = float(np.max(tstars) + eps_t_seconds)

        # Advance j0 to first real >= tmin
        while j0 < Nr and real[j0, 3] < tmin:
            j0 += 1
        j1 = j0
        while j1 < Nr and real[j1, 3] <= tmax:
            j1 += 1
        real_win = real[j0:j1]

        if len(real_win) == 0:
            continue

        # Build spatial tree over real_win
        tree_r = cKDTree(real_win[:, :2])

        # For every predicted in block, restrict further by time and do spatial match with per-query radius
        # Stage 1: time gating
        # We must skip already used events
        # Build arrays for variable radius query: we query using all preds, then mask by time & polarity
        idx_r_for_pred, dist_r_for_pred = _query_with_variable_radius(tree_r, pred_block[:, :2], r_pix_block)

        # Additional screening: explicit time + polarity + used flags + mutual NN
        candidate_pairs = []  # (i_pred_global, i_real_global, dist)
        for k in range(len(pred_block)):
            if used_pred[i0 + k]:
                continue
            ri_local = idx_r_for_pred[k]
            if ri_local < 0:
                continue

            # Time gate
            if not (abs(real_win[ri_local, 3] - (pred_block[k, 3])) <= eps_t_seconds):
                continue
            # Polarity gate
            if not check_polarity_match(real_win[ri_local, 2], pred_block[k, 2]):
                continue
            # Used flags
            gi = j0 + ri_local  # global real index
            if used_real[gi]:
                continue
            candidate_pairs.append((i0 + k, gi, float(dist_r_for_pred[k])))

        if not candidate_pairs:
            continue

        # Optional MNN: build reverse map pred<-real
        if mutual_nn:
            # Build kd-tree over predicted block for reverse search
            tree_p = cKDTree(pred_block[:, :2])
            # For each real in the window, find nearest predicted (with its own radius)
            # We need per-real radius too -> mirror adaptive with real XY
            r_pix_real = adaptive_r_pix_for_points(real_win[:, :2], cx, cy, omega_abs_hint)
            idx_p_for_real, _ = _query_with_variable_radius(tree_p, real_win[:, :2], r_pix_real)

            # Turn into global indices: pred global = i0 + idx_local ; real global = j0 + i
            # For filtering we need arrays indexed by (pred-block idx) and (real-window idx)
            pairs_r2p = -np.ones(len(real_win), dtype=np.int32)
            valid_r_loc = np.where(idx_p_for_real >= 0)[0]
            for rloc in valid_r_loc:
                pairs_r2p[rloc] = int(idx_p_for_real[rloc])

            pairs_p2r = -np.ones(len(pred_block), dtype=np.int32)
            # We need reverse map efficiently: do a 2nd query on real_win -> nearest pred per pred_block point
            # But we already have idx_r_for_pred from pred->real. Let's compute p2r via a tree on real_win:
            tree_r_for_p = cKDTree(real_win[:, :2])
            idx_r_for_pb, _ = _query_with_variable_radius(tree_r_for_p, pred_block[:, :2], r_pix_block)
            valid_p_loc = np.where(idx_r_for_pb >= 0)[0]
            for ploc in valid_p_loc:
                pairs_p2r[ploc] = int(idx_r_for_pb[ploc])

            # Now do mutual check on candidate_pairs
            filtered = []
            for (gp, gr, d) in candidate_pairs:
                ploc = gp - i0
                rloc = gr - j0
                if ploc < 0 or rloc < 0:
                    continue
                # Mutual condition: pred->real picks rloc, and real->pred picks ploc
                if pairs_p2r[ploc] == rloc and pairs_r2p[rloc] == ploc:
                    filtered.append((gp, gr, d))
            candidate_pairs = filtered

        # Greedy assign by distance
        candidate_pairs.sort(key=lambda z: z[2])
        for gp, gr, d in candidate_pairs:
            if used_pred[gp] or used_real[gr]:
                continue
            used_pred[gp] = True
            used_real[gr] = True
            total_matches += 1

    # Build residual arrays
    resid_real = real[~used_real]
    resid_pred = pred[~used_pred]
    return resid_real, resid_pred, total_matches

# ============================== Cancellation: bin mode ==============================
def _phase_aligned_edges(tmin, tmax, bin_ms, dt_seconds, overlap=False):
    """
    Build phase-aligned edges so that bin centers align with (t + Δt).
    Optionally create a 2nd stream of bins offset by BIN/2 to emulate 50% overlap.
    Returns: list of arrays of edges (one or two).
    """
    w = bin_ms * 1e-3
    if PHASE_ALIGN_BINS:
        # Align so predicted timestamp t+Δt sits near bin center
        phase = (dt_seconds % w) / 2.0
    else:
        phase = 0.0
    e0 = time_edges(tmin + phase, tmax + phase, bin_ms)

    if overlap:
        # 50% overlap: second stream shifted by w/2
        e1 = e0 + (w / 2.0)
        return [e0, e1]
    return [e0]

def _cancel_in_single_bin(real_bin, pred_bin, spatial_tol_pixels):
    """
    Old bin-based NN + greedy matching (kept for compatibility).
    """
    nr, npd = len(real_bin), len(pred_bin)
    if nr == 0 or npd == 0:
        return np.ones(nr, bool), np.ones(npd, bool), 0
    tree = cKDTree(pred_bin[:, :2])
    dist, idx = tree.query(real_bin[:, :2], k=1, distance_upper_bound=spatial_tol_pixels)
    real_cand = np.where(idx < npd)[0]
    if real_cand.size == 0:
        return np.ones(nr, bool), np.ones(npd, bool), 0
    cand = []
    for r_i in real_cand:
        p_i = int(idx[r_i])
        if check_polarity_match(real_bin[r_i, 2], pred_bin[p_i, 2]):
            cand.append((float(dist[r_i]), r_i, p_i))
    if not cand:
        return np.ones(nr, bool), np.ones(npd, bool), 0
    cand.sort(key=lambda z: z[0])
    used_pred = set()
    mr = np.zeros(nr, bool); mp = np.zeros(npd, bool)
    for _, r_i, p_i in cand:
        if p_i in used_pred:
            continue
        used_pred.add(p_i); mr[r_i] = True; mp[p_i] = True
    return ~mr, ~mp, int(mr.sum())

def cancel_with_bins(combined, bin_ms, spatial_tolerance_pixels,
                     dt_seconds=DT_SECONDS, overlap=SAFE_OVERLAP_BINS):
    """
    Bin-based cancellation (kept for backward compatibility).
    Improvements:
      - Phase-aligned binning to Δt (reduces boundary misses).
      - Optional 50% overlapped bins (significantly reduces edge artifacts).
    """
    t = combined[:, 3]
    edges_streams = _phase_aligned_edges(float(t.min()), float(t.max()), bin_ms, dt_seconds, overlap=overlap)

    total_real = int(np.sum(combined[:, 4] == 0.0))
    total_pred = int(np.sum(combined[:, 4] == 1.0))
    used_real = np.zeros(len(combined), dtype=bool)
    used_pred = np.zeros(len(combined), dtype=bool)

    total_matches = 0
    for edges in edges_streams:
        N = len(combined); left_ix = 0
        for b in range(len(edges) - 1):
            left, right = edges[b], edges[b+1]
            i0 = left_ix
            while i0 < N and combined[i0,3] < left:  i0 += 1
            i1 = i0
            while i1 < N and combined[i1,3] < right: i1 += 1
            left_ix = i0
            if i1 <= i0:
                continue
            view = combined[i0:i1]
            # mask out items already matched in earlier stream/bins
            m_unused_real = (view[:,4] == 0.0)
            m_unused_pred = (view[:,4] == 1.0)
            # but drop already-used indices
            gidx = np.arange(i0, i1)
            m_unused_real &= ~used_real[gidx]
            m_unused_pred &= ~used_pred[gidx]
            real_bin = view[m_unused_real]
            pred_bin = view[m_unused_pred]
            if len(real_bin) == 0 or len(pred_bin) == 0:
                continue
            ur, up, m = _cancel_in_single_bin(real_bin, pred_bin, spatial_tolerance_pixels)
            total_matches += m
            # mark used in global space
            # rebuild global indices for real/pred in this bin
            idx_real_local = np.where(m_unused_real)[0]
            idx_pred_local = np.where(m_unused_pred)[0]
            used_real[gidx[idx_real_local[~ur]]] = True
            used_pred[gidx[idx_pred_local[~up]]] = True
    resid_real = combined[(combined[:,4]==0.0) & (~used_real)]
    resid_pred = combined[(combined[:,4]==1.0) & (~used_pred)]
    return resid_real, resid_pred, total_matches

# ============================== “Fixed” hybrid (compatibility) ======================
def run_cancellation_fixed(combined_events, dt_seconds, temporal_tolerance_ms, spatial_tolerance_pixels):
    """
    Your original "fixed" method improved: expanded bins to include dt, then time-aware matching.
    Kept for compatibility, but "window" mode above is preferred and more exact.
    """
    temporal_tolerance_s = temporal_tolerance_ms * 1e-3
    real_events = combined_events[combined_events[:, 4] == 0.0]
    pred_events = combined_events[combined_events[:, 4] == 1.0]
    if len(real_events) == 0 or len(pred_events) == 0:
        return real_events, pred_events, 0

    effective_bin_ms = temporal_tolerance_ms + (dt_seconds * 1000.0)
    timestamps = combined_events[:, 3]
    edges = time_edges(float(timestamps.min()), float(timestamps.max()), effective_bin_ms)

    total_matched_pairs = 0
    unmatched_real_chunks, unmatched_pred_chunks = [], []
    print(f"[fixed] dt={dt_seconds*1000:.1f}ms, eps_t={temporal_tolerance_ms}ms, eff_bin={effective_bin_ms:.1f}ms")

    for b in range(len(edges) - 1):
        t0, t1 = edges[b], edges[b+1]
        m = (combined_events[:,3] >= t0) & (combined_events[:,3] < t1)
        bin_events = combined_events[m]
        if len(bin_events) == 0:
            continue

        bin_real = bin_events[bin_events[:,4]==0.0]
        bin_pred = bin_events[bin_events[:,4]==1.0]
        if len(bin_real) == 0 or len(bin_pred) == 0:
            if len(bin_real): unmatched_real_chunks.append(bin_real)
            if len(bin_pred): unmatched_pred_chunks.append(bin_pred)
            continue

        matched_r = np.zeros(len(bin_real), bool)
        matched_p = np.zeros(len(bin_pred), bool)

        for i, re in enumerate(bin_real):
            if matched_r[i]: continue
            tstar = re[3] + dt_seconds
            time_diffs = np.abs(bin_pred[:,3] - tstar)
            cand_idx = np.where((time_diffs <= temporal_tolerance_s) & (~matched_p))[0]
            if cand_idx.size == 0: continue
            # spatial + polarity
            rp = re[:2]; best = None; best_d = float('inf')
            for j in cand_idx:
                pe = bin_pred[j]
                if not check_polarity_match(re[2], pe[2]): continue
                d = float(np.linalg.norm(rp - pe[:2]))
                if d <= spatial_tolerance_pixels and d < best_d:
                    best_d = d; best = j
            if best is not None:
                matched_r[i] = True
                matched_p[best] = True
                total_matched_pairs += 1

        if not matched_r.all():
            unmatched_real_chunks.append(bin_real[~matched_r])
        if not matched_p.all():
            unmatched_pred_chunks.append(bin_pred[~matched_p])

    resid_real = np.vstack(unmatched_real_chunks) if unmatched_real_chunks else np.zeros((0,5), dtype=combined_events.dtype)
    resid_pred = np.vstack(unmatched_pred_chunks) if unmatched_pred_chunks else np.zeros((0,5), dtype=combined_events.dtype)
    print(f"[fixed] matches={total_matched_pairs:,}")
    return resid_real, resid_pred, total_matched_pairs

# =============================== High-level runners =================================
def run_cancellation_window(combined):
    print("Cancellation mode: exact temporal window")
    resid_real, resid_pred, m = cancel_with_temporal_window(
        combined,
        dt_seconds=DT_SECONDS,
        eps_t_seconds=EPS_T,
        cx=DISC_CENTER_X, cy=DISC_CENTER_Y,
        omega_abs_hint=2*math.pi,  # 1 rev/s default hint; adjust if you know |ω|
        mutual_nn=MUTUAL_NN
    )
    total_real = int(np.sum(combined[:,4]==0.0))
    total_pred = int(np.sum(combined[:,4]==1.0))
    rc = (total_real - len(resid_real)) / max(total_real,1) * 100.0
    pc = (total_pred - len(resid_pred)) / max(total_pred,1) * 100.0
    print(f"Real: {total_real:,} -> residual {len(resid_real):,} (cancelled {total_real-len(resid_real):,}, {rc:.1f}%)")
    print(f"Pred: {total_pred:,} -> residual {len(resid_pred):,} (cancelled {total_pred-len(resid_pred):,}, {pc:.1f}%)")
    print(f"Matched pairs: {m:,}")
    print(f"Δt={DT_SECONDS*1e3:.2f} ms, ε_t={EPS_T*1e3:.1f} ms, spatial_tol={'adaptive' if ADAPTIVE_SPATIAL_TOL else R_PIX} px, polarity={POLARITY_MODE}")
    return resid_real, resid_pred

def run_cancellation_bin(combined):
    print("Cancellation mode: bin-based (phase-aligned; overlap={} )".format(SAFE_OVERLAP_BINS))
    resid_real, resid_pred, m = cancel_with_bins(
        combined,
        bin_ms=BIN_MS,
        spatial_tolerance_pixels=R_PIX,
        dt_seconds=DT_SECONDS,
        overlap=SAFE_OVERLAP_BINS
    )
    total_real = int(np.sum(combined[:,4]==0.0))
    total_pred = int(np.sum(combined[:,4]==1.0))
    rc = (total_real - len(resid_real)) / max(total_real,1) * 100.0
    pc = (total_pred - len(resid_pred)) / max(total_pred,1) * 100.0
    print(f"Real: {total_real:,} -> residual {len(resid_real):,} (cancelled {total_real-len(resid_real):,}, {rc:.1f}%)")
    print(f"Pred: {total_pred:,} -> residual {len(resid_pred):,} (cancelled {total_pred-len(resid_pred):,}, {pc:.1f}%)")
    print(f"Matched pairs: {m:,}")
    print(f"BIN={BIN_MS:.1f} ms (~ε_t≈{0.5*BIN_MS:.1f} ms), spatial_tol={R_PIX}px, polarity={POLARITY_MODE}")
    return resid_real, resid_pred

# ============================= Visualization / Panels ===============================
def make_frame(H, W, xs, ys, ps=None):
    frame = np.zeros((H, W), dtype=np.int32)
    if len(xs) == 0:
        return frame
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs_valid = xs[valid]; ys_valid = ys[valid]
    if ps is not None:
        np.add.at(frame, (ys_valid, xs_valid), ps[valid])
    else:
        np.add.at(frame, (ys_valid, xs_valid), 1)
    return frame

def convert_polarity_to_signed(p):
    return np.where(p > 0.5, 1, -1).astype(np.int16)

def make_panel_figure(combined, resid_real, resid_pred, window, img_w=IMG_W, img_h=IMG_H, use_gray=False):
    t0, t1 = window
    allm = (combined[:,3] >= t0) & (combined[:,3] < t1)
    w_all = combined[allm]
    w_real = w_all[w_all[:,4] == 0.0]
    w_pred = w_all[w_all[:,4] == 1.0]
    wr = resid_real[(resid_real[:,3] >= t0) & (resid_real[:,3] < t1)]
    wp = resid_pred[(resid_pred[:,3] >= t0) & (resid_pred[:,3] < t1)]
    cancelled = len(w_real) - len(wr)

    img_r, img_p, img_c, nr, npred = build_window_images(combined, window, img_w, img_h)
    img_r_n, img_p_n, img_c_n, max_abs = normalize_images_for_display(img_r, img_p, img_c)

    m_r = np.abs(img_r.ravel()); m_p = np.abs(img_p.ravel()); m_c = np.abs(img_c.ravel())
    nz_r = m_r[m_r > 0]; nz_p = m_p[m_p > 0]; nz_c = m_c[m_c > 0]

    fig = plt.figure(figsize=(22, 16))
    gs = gridspec.GridSpec(nrows=3, ncols=3, height_ratios=[1.1, 1.1, 1.0], figure=fig)

    ax0 = fig.add_subplot(gs[0, 0]); ax1 = fig.add_subplot(gs[0, 1]); ax2 = fig.add_subplot(gs[0, 2])
    ax0.scatter(w_real[:,0], w_real[:,1], s=2, alpha=0.6, c="tab:blue"); ax0.set_title(f"Real ({len(w_real):,})")
    ax1.scatter(w_pred[:,0], w_pred[:,1], s=2, alpha=0.6, c="tab:red");  ax1.set_title(f"Predicted ({len(w_pred):,})")
    ax2.scatter(wr[:,0], wr[:,1], s=2, alpha=0.7, c="tab:blue", label=f"Real ({len(wr):,})")
    ax2.scatter(wp[:,0], wp[:,1], s=2, alpha=0.7, c="tab:red",  label=f"Pred ({len(wp):,})")
    ax2.legend()
    for ax in (ax0, ax1, ax2):
        ax.set_xlim(0, img_w); ax.set_ylim(0, img_h); ax.invert_yaxis(); ax.grid(True, alpha=0.3); ax.set_xlabel("x")
    ax0.set_ylabel("y")

    cmap = "gray" if use_gray else "seismic"
    b0 = fig.add_subplot(gs[1, 0]); b1 = fig.add_subplot(gs[1, 1]); b2 = fig.add_subplot(gs[1, 2])
    im0 = b0.imshow(img_r_n, cmap=cmap, origin="upper", vmin=0, vmax=1); b0.set_title(f"Real (N={nr:,})")
    im1 = b1.imshow(img_p_n, cmap=cmap, origin="upper", vmin=0, vmax=1); b1.set_title(f"Pred (N={npred:,})")
    im2 = b2.imshow(img_c_n, cmap=cmap, origin="upper", vmin=0, vmax=1); b2.set_title("Combined")
    for ax in (b0,b1,b2):
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.grid(alpha=0.2)

    if cmap == "seismic":
        disc_circle = plt.Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, fill=True, color='yellow', alpha=0.15, linewidth=0)
        b2.add_patch(disc_circle)
        disc_outline = plt.Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
        b2.add_patch(disc_outline)
        b2.plot(DISC_CENTER_X, DISC_CENTER_Y, 'yo', markersize=6, markeredgecolor='black', markeredgewidth=1, label='Disc Center')
        b2.legend(loc='upper right', fontsize=7)

    for ax_im, im in zip((b0, b1, b2), (im0, im1, im2)):
        cb = fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.04)
        cb.set_label("signed count (Σ polarity)")
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels([f"-{max_abs}", "0", f"+{max_abs}"])

    c0 = fig.add_subplot(gs[2, 0]); c1 = fig.add_subplot(gs[2, 1]); c2 = fig.add_subplot(gs[2, 2])
    def plot_histogram(ax, data, title):
        ax.hist(data, bins=HIST_BINS, log=True, edgecolor="k", alpha=0.85)
        if data.size:
            med = float(np.median(data)); p95 = float(np.percentile(data, 95))
            ax.axvline(med, color="tab:orange", ls="--", lw=1.2, label=f"median={med:.1f}")
            ax.axvline(p95, color="tab:green",  ls="--", lw=1.0, label=f"95%={p95:.1f}")
        ax.set_xlabel("|signed count|"); ax.set_ylabel("pixels (log)"); ax.grid(alpha=0.3); ax.legend(); ax.set_title(title)
    m_r = np.abs(img_r.ravel()); m_p = np.abs(img_p.ravel()); m_c = np.abs(img_c.ravel())
    nz_r = m_r[m_r > 0]; nz_p = m_p[m_p > 0]; nz_c = m_c[m_c > 0]
    plot_histogram(c0, nz_r, f"Real |count| (nonzero px={nz_r.size:,})")
    plot_histogram(c1, nz_p, f"Pred |count| (nonzero px={nz_p.size:,})")
    plot_histogram(c2, nz_c, f"Combined |count| (nonzero px={nz_c.size:,})")

    total_real = len(w_real); actual_cr = (cancelled / total_real * 100) if total_real > 0 else 0
    expected_cr = 95 * (1 - np.exp(-BIN_MS / 2.0))  # kept for continuity (informal reference)
    interp_method = "bilinear" if USE_BILINEAR_INTERP else "nearest"
    fig.suptitle(
        f"Time window {t0:.3f}–{t1:.3f}s • cancel={cancelled:,} "
        f"({actual_cr:.1f}% actual vs {expected_cr:.1f}% expected) • "
        f"spatial_tol={'adaptive' if ADAPTIVE_SPATIAL_TOL else R_PIX}px, ε_t={EPS_T*1e3:.1f}ms • "
        f"{'gray' if use_gray else 'seismic'} images • {interp_method} raster",
        y=0.995, fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig

def export_windows_panel_images(combined, resid_real, resid_pred, windows, output_dir, use_gray=False):
    os.makedirs(output_dir, exist_ok=True)
    for i, w in enumerate(windows):
        fig = make_panel_figure(combined, resid_real, resid_pred, w, use_gray=use_gray)
        suffix = "_gray" if use_gray else "_seismic"
        filename = f"ego_panel_{i+1}_{w[0]:.3f}s_to_{w[1]:.3f}s{suffix}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved panel image: {filename}")

# =============================== ROI analysis (as before) ===========================
def create_roi_analysis_figure(combined_events, residual_real_events, residual_predicted_events,
                               window, cx, cy, radius, img_w, img_h):
    from matplotlib.colors import TwoSlopeNorm
    t0, t1 = window
    time_mask = (combined_events[:, 3] >= t0) & (combined_events[:, 3] < t1)
    window_events = combined_events[time_mask]
    real_events = window_events[window_events[:, 4] == 0.0]
    pred_events = window_events[window_events[:, 4] == 1.0]

    resid_real_window = residual_real_events[(residual_real_events[:,3] >= t0) & (residual_real_events[:,3] < t1)]
    resid_pred_window = residual_predicted_events[(residual_predicted_events[:,3] >= t0) & (residual_predicted_events[:,3] < t1)]

    sel_r_t = events_in_window(real_events, t0, t1)
    sel_p_t = events_in_window(pred_events, t0, t1)

    inside_r_full = circle_mask(real_events[:,0], real_events[:,1], cx, cy, radius, scale=1.05)
    inside_p_full = circle_mask(pred_events[:,0], pred_events[:,1], cx, cy, radius, scale=1.05)

    inside_r = inside_r_full[sel_r_t]
    outside_r = ~inside_r

    # Residual membership (approximate by exact tuple equality within tolerance)
    residual_full = np.zeros(len(real_events), dtype=bool)
    if len(resid_real_window) > 0:
        # map rows for quick lookup (coarse tolerance)
        # Build KD-tree on residual real for spatial+time proximity to mark as residual
        tree_rr = cKDTree(resid_real_window[:, :2])
        dist, idx = tree_rr.query(real_events[:, :2], k=1, distance_upper_bound=1.0)
        # also require similar time (within 0.25 ms)
        tclose = np.abs(resid_real_window[np.clip(idx,0,len(resid_real_window)-1),3] - real_events[:,3]) <= 0.00025
        residual_full = (idx < len(resid_real_window)) & tclose

    residual_t = residual_full[sel_r_t]
    resid_in = residual_t & inside_r
    resid_out = residual_t & outside_r

    matched_real_full = ~residual_full
    matched_r_t = matched_real_full[sel_r_t]
    match_in = matched_r_t & inside_r
    match_out = matched_r_t & outside_r

    yy, xx = np.mgrid[0:img_h, 0:img_w]
    pix_in_mask = circle_mask(xx, yy, cx, cy, radius, scale=1.05)
    in_area = int(pix_in_mask.sum())
    out_area = img_w * img_h - in_area

    count_real_inside = int(np.sum(inside_r))
    count_residual_inside = int(np.sum(resid_in))
    count_real_outside = int(np.sum(outside_r))
    count_residual_outside = int(np.sum(resid_out))

    cr_inside = (count_real_inside - count_residual_inside) / count_real_inside * 100 if count_real_inside > 0 else 0
    cr_outside = (count_real_outside - count_residual_outside) / count_real_outside * 100 if count_real_outside > 0 else 0

    e_in  = (count_residual_inside / in_area) if in_area > 0 else 0
    e_out = (count_residual_outside / out_area) if out_area > 0 else 0

    print(f"[INSIDE ROI] real={count_real_inside}, residual={count_residual_inside}, cancellation_rate={cr_inside:.2f}%, events_per_pixel={e_in:.4f}")
    print(f"[OUTSIDE ROI] real={count_real_outside}, residual={count_residual_outside}, cancellation_rate={cr_outside:.2f}%, events_per_pixel={e_out:.4f}")

    xr = real_events[sel_r_t, 0].astype(int)
    yr = real_events[sel_r_t, 1].astype(int)
    pr = real_events[sel_r_t, 2].astype(int)
    pr_signed = convert_polarity_to_signed(pr)

    F_real_in  = make_frame(img_h, img_w, xr[inside_r],   yr[inside_r],   pr_signed[inside_r])
    F_resid_in = make_frame(img_h, img_w, xr[resid_in],   yr[resid_in],   pr_signed[resid_in])
    F_pair_in  = make_frame(img_h, img_w, xr[match_in],   yr[match_in],   None)

    F_real_out  = make_frame(img_h, img_w, xr[outside_r],  yr[outside_r],  pr_signed[outside_r])
    F_resid_out = make_frame(img_h, img_w, xr[resid_out],  yr[resid_out],  pr_signed[resid_out])
    F_pair_out  = make_frame(img_h, img_w, xr[match_out],  yr[match_out],  None)

    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = gridspec.GridSpec(3, 3, figure=fig)

    ax_main = fig.add_subplot(gs[0, :])
    img_r, img_p, img_c, nr, npred = build_window_images(combined_events, window, img_w, img_h)
    img_r_n, img_p_n, img_c_n, max_abs = normalize_images_for_display(img_r, img_p, img_c)

    im2 = ax_main.imshow(img_c_n, cmap="seismic", origin="upper", vmin=0, vmax=1)
    ax_main.set_title("Combined Seismic Overlay")
    ax_main.set_xlabel("x"); ax_main.set_ylabel("y"); ax_main.grid(alpha=0.2)

    disc_circle = plt.Circle((cx, cy), radius, fill=True, color='yellow', alpha=0.15, linewidth=0)
    ax_main.add_patch(disc_circle)
    disc_outline = plt.Circle((cx, cy), radius, fill=False, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
    ax_main.add_patch(disc_outline)
    ax_main.plot(cx, cy, 'yo', markersize=6, markeredgecolor='black', markeredgewidth=1, label='Disc Center')
    ax_main.legend(loc='upper right', fontsize=7)

    cb = fig.colorbar(im2, ax=ax_main, fraction=0.046, pad=0.04)
    cb.set_label("signed count (Σ polarity)")
    cb.set_ticks([0.0, 0.5, 1.0]); cb.set_ticklabels([f"-{max_abs}", "0", f"+{max_abs}"])

    ax_i1 = fig.add_subplot(gs[1, 0]); ax_i2 = fig.add_subplot(gs[1, 1]); ax_i3 = fig.add_subplot(gs[1, 2])
    ax_o1 = fig.add_subplot(gs[2, 0]); ax_o2 = fig.add_subplot(gs[2, 1]); ax_o3 = fig.add_subplot(gs[2, 2])

    from matplotlib.colors import TwoSlopeNorm
    vmax_in = max(1, int(np.percentile(np.abs(F_real_in), 99))) if np.any(F_real_in != 0) else 1
    norm_in = TwoSlopeNorm(vmin=-vmax_in, vcenter=0, vmax=vmax_in)

    F_real_in_safe  = np.clip(np.nan_to_num(F_real_in,  nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_resid_in_safe = np.clip(np.nan_to_num(F_resid_in, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_pair_in_safe  = np.clip(np.nan_to_num(F_pair_in,  nan=0, posinf=0, neginf=0),  0,   1e6)

    im_i1 = ax_i1.imshow(F_real_in_safe,  cmap='seismic', interpolation='nearest', origin='upper',
                         extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_in)
    im_i2 = ax_i2.imshow(F_resid_in_safe, cmap='seismic', interpolation='nearest', origin='upper',
                         extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_in)
    im_i3 = ax_i3.imshow(F_pair_in_safe,  cmap='Reds',    interpolation='nearest', origin='upper',
                         extent=(-0.5, img_w-0.5, img_h-0.5, -0.5))

    vmax_out = max(1, int(np.percentile(np.abs(F_real_out), 99))) if np.any(F_real_out != 0) else 1
    norm_out = TwoSlopeNorm(vmin=-vmax_out, vcenter=0, vmax=vmax_out)

    F_real_out_safe  = np.clip(np.nan_to_num(F_real_out,  nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_resid_out_safe = np.clip(np.nan_to_num(F_resid_out, nan=0, posinf=0, neginf=0), -1e6, 1e6)
    F_pair_out_safe  = np.clip(np.nan_to_num(F_pair_out,  nan=0, posinf=0, neginf=0),  0,   1e6)

    im_o1 = ax_o1.imshow(F_real_out_safe,  cmap='seismic', interpolation='nearest', origin='upper',
                         extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_out)
    im_o2 = ax_o2.imshow(F_resid_out_safe, cmap='seismic', interpolation='nearest', origin='upper',
                         extent=(-0.5, img_w-0.5, img_h-0.5, -0.5), norm=norm_out)
    im_o3 = ax_o3.imshow(F_pair_out_safe,  cmap='Reds',    interpolation='nearest', origin='upper',
                         extent=(-0.5, img_w-0.5, img_h-0.5, -0.5))

    for ax in [ax_i1, ax_i2, ax_i3, ax_o1, ax_o2, ax_o3]:
        circle = plt.Circle((cx, cy), radius * 1.05, fill=False, color='yellow', linewidth=1, linestyle='--', alpha=0.8)
        ax.add_patch(circle)
        ax.set_xlabel("x [px]"); ax.set_ylabel("y [px]")

    ax_i1.set_title(f"INSIDE: Real ({count_real_inside:,})")
    ax_i2.set_title(f"INSIDE: Residual ({count_residual_inside:,})")
    ax_i3.set_title(f"INSIDE: Matched (real) ({int(np.sum(match_in)):,})")

    ax_o1.set_title(f"OUTSIDE: Real ({count_real_outside:,})")
    ax_o2.set_title(f"OUTSIDE: Residual ({count_residual_outside:,})")
    ax_o3.set_title(f"OUTSIDE: Matched (real) ({int(np.sum(match_out)):,})")

    fig.suptitle(f"ROI Analysis: {t0:.3f}–{t1:.3f}s | Inside: {cr_inside:.1f}% cancelled | Outside: {cr_outside:.1f}%",
                 fontsize=14, y=0.98)

    for ax in [ax_main, ax_i1, ax_i2, ax_i3, ax_o1, ax_o2, ax_o3]:
        ax.format_coord = lambda x, y: ""
        ax.set_navigate(False)
        for im in ax.get_images():
            im.set_interpolation('nearest')
    return fig

# ==================================== Main =========================================
def main():
    print("Loading combined events data...")
    combined = load_combined(COMBINED_PATH)

    # Choose cancellation mode
    if CANCEL_MODE == "window":
        residual_real_events, residual_predicted_events = run_cancellation_window(combined)
    elif CANCEL_MODE == "bin":
        residual_real_events, residual_predicted_events = run_cancellation_bin(combined)
    else:
        print(f"[warn] Unknown CANCEL_MODE={CANCEL_MODE!r}; falling back to 'window'")
        residual_real_events, residual_predicted_events = run_cancellation_window(combined)

    print("Exporting analysis panels...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    export_windows_panel_images(combined, residual_real_events, residual_predicted_events, WINDOWS,
                                OUTPUT_DIR, use_gray=False)
    export_windows_panel_images(combined, residual_real_events, residual_predicted_events, WINDOWS,
                                OUTPUT_DIR, use_gray=True)

    print("Creating ROI analysis visualization...")
    w0 = WINDOWS[0]
    fig_roi = create_roi_analysis_figure(
        combined, residual_real_events, residual_predicted_events,
        w0, DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS, IMG_W, IMG_H
    )
    roi_filename = f"roi_analysis_{w0[0]:.3f}s_to_{w0[1]:.3f}s.png"
    fig_roi.savefig(os.path.join(OUTPUT_DIR, roi_filename), dpi=150, bbox_inches="tight")
    print(f"Saved ROI analysis: {roi_filename}")

    # Summary
    print("\nAnalysis complete!")
    print(f"Time windows analyzed: {len(WINDOWS)}")
    if CANCEL_MODE == "window":
        print(f"Cancellation parameters: Δt={DT_SECONDS*1e3:.2f}ms, ε_t={EPS_T*1e3:.1f}ms, "
              f"spatial_tol={'adaptive' if ADAPTIVE_SPATIAL_TOL else R_PIX}px, mutual_nn={MUTUAL_NN}, polarity={POLARITY_MODE}")
    else:
        print(f"Cancellation parameters: BIN={BIN_MS:.1f}ms (phase_align={PHASE_ALIGN_BINS}, overlap={SAFE_OVERLAP_BINS}), "
              f"spatial_tol={R_PIX}px, polarity={POLARITY_MODE}")

    try:
        plt.show(block=False); plt.pause(0.1)
    except Exception:
        pass

if __name__ == "__main__":
    main()
