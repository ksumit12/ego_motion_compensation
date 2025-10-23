#!/usr/bin/env python3
"""
Scalable DT and Tolerance Analysis for Large Datasets
Streaming temporal sliding window with uniform spatial grid for 43M+ events
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import deque, defaultdict
from math import ceil, floor
import gc
import psutil

# Set random seed for reproducibility
np.random.seed(42)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available - using pure NumPy")

# Optimized data types
EVENT_DTYPE = np.dtype([
    ('x', '<f4'),    # float32 x coordinate
    ('y', '<f4'),    # float32 y coordinate  
    ('p', 'i1'),     # int8 polarity
    ('t', '<f4'),    # float32 time in seconds
    ('id', '<i4')    # int32 original index for tie-breaking
])

# =============== Configuration ===============
WINDOW_PREDICTIONS_DIR = "/mnt/storage/window_predictions_5s"
DT_VALUES_MS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]  # 11 values instead of 21

# Tolerance ranges to test
SPATIAL_TOLERANCE_RANGE = (1.0, 4.0, 1.0)  # (min, max, step) in pixels - 4 values
TEMPORAL_TOLERANCE_RANGE = (2.0, 8.0, 2.0)  # (min, max, step) in milliseconds - 4 values

# Streaming settings
# Streaming settings - optimized for EC2
STREAM_BATCH_SIZE = 5_000_000  # Larger batches for better performance
MICRO_BUCKET_TIME = 0.001  # 1ms micro-buckets for deterministic matching
GRID_CELL_SIZE_FACTOR = 1.0  # cell_size = spatial_tol * factor

# Polarity mode
POLARITY_MODE = "opposite"  # "opposite" | "equal" | "ignore"

# Disc center coordinates and radius
DISC_CENTER_X = 665.2710509177888
DISC_CENTER_Y = 337.4668998290816
DISC_RADIUS = 264

# Time windows - Single 5-second window
WINDOWS = [(5.000, 10.000)]

# Output settings
OUTPUT_DIR = "./dt_tolerance_analysis_results_5s"

# =============== Core Classes ===============

def get_window_dir(base_dir, window_idx, t0, t1):
    """Build window directory path consistently."""
    return os.path.join(base_dir, f"window_{window_idx + 1}_{t0:.3f}s_to_{t1:.3f}s")

def resolve_window_files(base_dir, window_idx, t0, t1, dt_ms):
    """Return (real_path, pred_path) for a given window and dt."""
    window_dir = get_window_dir(base_dir, window_idx, t0, t1)
    real_path = os.path.join(window_dir, "real_events.npy")
    pred_path = os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npy")
    return window_dir, real_path, pred_path

def paths_exist_or_warn(real_path, pred_path):
    """Validate presence of real and predicted files; print helpful messages."""
    ok = True
    if not os.path.exists(real_path):
        print(f"  Error: Missing {real_path}")
        ok = False
    if not os.path.exists(pred_path):
        print(f"    Warning: Missing {pred_path}")
        ok = False
    return ok
class UniformSpatialGrid:
    """Uniform spatial grid for O(1) neighborhood queries"""
    
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)  # (gx, gy) -> list of predicted events
        self.fifo = deque()  # FIFO queue for temporal eviction
        
    def key(self, x, y):
        """Get grid cell key for coordinates"""
        return (int(floor(x / self.cell_size)), int(floor(y / self.cell_size)))
    
    def insert(self, event):
        """Insert predicted event into grid"""
        k = self.key(event['x'], event['y'])
        self.grid[k].append(event)
        self.fifo.append(event)
    
    def evict_until(self, min_time):
        """Evict events older than min_time"""
        while self.fifo and self.fifo[0]['t'] < min_time:
            event = self.fifo.popleft()
            k = self.key(event['x'], event['y'])
            bucket = self.grid.get(k, [])
            if bucket and event in bucket:
                bucket.remove(event)
                if not bucket:
                    del self.grid[k]
    
    def get_neighbors(self, center_key, radius):
        """Get all events in 3x3 neighborhood around center_key"""
        gx, gy = center_key
        neighbors = []
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                bucket = self.grid.get(key, [])
                neighbors.extend(bucket)
        
        return neighbors
    
    def remove_event(self, event):
        """Remove specific event from grid"""
        k = self.key(event['x'], event['y'])
        bucket = self.grid.get(k, [])
        if bucket and event in bucket:
            bucket.remove(event)
            if not bucket:
                del self.grid[k]

class MatchingKernel:
    """Encapsulate matching logic for easier swapping/testing."""
    @staticmethod
    def best_match(real_event, candidates, spatial_tol, temporal_tol_s):
        if not candidates:
            return -1
        cand_x = np.array([c['x'] for c in candidates])
        cand_y = np.array([c['y'] for c in candidates])
        cand_t = np.array([c['t'] for c in candidates])
        cand_p = np.array([c['p'] for c in candidates])
        cand_id = np.array([c['id'] for c in candidates])
        return find_best_match_numba(
            real_event['x'], real_event['y'], real_event['t'], real_event['p'],
            cand_x, cand_y, cand_t, cand_p, cand_id, spatial_tol, temporal_tol_s
        )

class EventStreamer:
    """Stream events from file in time-sorted batches"""
    
    def __init__(self, file_path, batch_size=STREAM_BATCH_SIZE):
        self.file_path = file_path
        self.batch_size = batch_size
        try:
            self.mmap = np.load(file_path, mmap_mode='r')
        except Exception as e:
            raise RuntimeError(f"Failed to open {file_path} with memmap: {e}")
        self.total_events = len(self.mmap)
        self.current_idx = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= self.total_events:
            raise StopIteration
        
        end_idx = min(self.current_idx + self.batch_size, self.total_events)
        try:
            chunk = self.mmap[self.current_idx:end_idx]
        except Exception as e:
            raise RuntimeError(f"Failed to slice memmap {self.file_path} [{self.current_idx}:{end_idx}]: {e}")
        
        # Convert to structured array with optimized dtype
        if chunk.ndim != 2 or chunk.shape[1] < 4:
            raise ValueError(f"Invalid event array shape in {self.file_path}: {chunk.shape}, expected (N,4+)")
        structured_chunk = np.zeros(len(chunk), dtype=EVENT_DTYPE)
        try:
            structured_chunk['x'] = chunk[:, 0].astype(np.float32)
            structured_chunk['y'] = chunk[:, 1].astype(np.float32)
            structured_chunk['p'] = chunk[:, 2].astype(np.int8)
            structured_chunk['t'] = chunk[:, 3].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed dtype conversion for {self.file_path}: {e}")
        structured_chunk['id'] = np.arange(self.current_idx, end_idx, dtype=np.int32)
        
        # Sort by time for deterministic processing
        try:
            if not np.all(structured_chunk['t'][1:] >= structured_chunk['t'][:-1]):
                # Not sorted; perform stable in-chunk sort and warn once
                time_order = np.argsort(structured_chunk['t'], kind='mergesort')
                structured_chunk = structured_chunk[time_order]
        except Exception as e:
            raise RuntimeError(f"Failed time sort for {self.file_path}: {e}")
        
        self.current_idx = end_idx
        return structured_chunk
    
    def close(self):
        """Close memory mapping"""
        if hasattr(self, 'mmap'):
            del self.mmap

# =============== Core Functions ===============
def circle_mask(x, y, cx, cy, r, scale=1.05):
    """Create boolean mask for points inside scaled circle"""
    return (x - cx)**2 + (y - cy)**2 <= (r * scale)**2

if HAS_NUMBA:
    @numba.njit
    def find_best_match_numba(real_x, real_y, real_t, real_p, candidates_x, candidates_y, 
                             candidates_t, candidates_p, candidates_id, spatial_tol, temporal_tol):
        """Find best match using Numba acceleration"""
        best_idx = -1
        best_cost = 1e30
        
        spatial_tol2 = spatial_tol * spatial_tol
        
        for i in range(len(candidates_x)):
            # Spatial distance check
            dx = candidates_x[i] - real_x
            dy = candidates_y[i] - real_y
            dist2 = dx * dx + dy * dy
            
            if dist2 > spatial_tol2:
                continue
            
            # Temporal distance check
            dt = abs(candidates_t[i] - real_t)
            if dt > temporal_tol:
                continue
            
            # Polarity check (opposite mode)
            if real_p == candidates_p[i]:
                continue
            
            # Lexicographic cost: temporal first, then spatial, then ID for tie-breaking
            cost = dt * 1000.0 + dist2 + candidates_id[i] * 1e-6
            
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        
        return best_idx
else:
    def find_best_match_numba(real_x, real_y, real_t, real_p, candidates_x, candidates_y, 
                             candidates_t, candidates_p, candidates_id, spatial_tol, temporal_tol):
        """Fallback to pure NumPy implementation"""
        if len(candidates_x) == 0:
            return -1
        
        # Spatial distance check
        dx = candidates_x - real_x
        dy = candidates_y - real_y
        dist2 = dx * dx + dy * dy
        spatial_mask = dist2 <= spatial_tol * spatial_tol
        
        # Temporal distance check
        dt = np.abs(candidates_t - real_t)
        temporal_mask = dt <= temporal_tol
        
        # Polarity check (opposite mode)
        polarity_mask = real_p != candidates_p
        
        # Combined mask
        valid_mask = spatial_mask & temporal_mask & polarity_mask
        
        if not np.any(valid_mask):
            return -1
        
        # Find best match among valid candidates
        valid_indices = np.where(valid_mask)[0]
        valid_dt = dt[valid_indices]
        valid_dist2 = dist2[valid_indices]
        valid_ids = candidates_id[valid_indices]
        
        # Lexicographic cost: temporal first, then spatial, then ID for tie-breaking
        costs = valid_dt * 1000.0 + valid_dist2 + valid_ids * 1e-6
        
        best_local_idx = np.argmin(costs)
        return valid_indices[best_local_idx]

def stream_cancellation(real_streamer, pred_streamer, dt_seconds, spatial_tol, temporal_tol):
    """
    Main streaming cancellation algorithm with temporal sliding window
    """
    print(f"    [DEBUG] Starting stream_cancellation: dt={dt_seconds:.3f}s, spatial={spatial_tol}px, temporal={temporal_tol}ms")
    
    spatial_tol_s = spatial_tol * 1e-3  # Convert to seconds
    temporal_tol_s = temporal_tol * 1e-3  # Convert to seconds
    
    # Initialize spatial grid
    cell_size = spatial_tol * GRID_CELL_SIZE_FACTOR
    grid = UniformSpatialGrid(cell_size)
    print(f"    [DEBUG] Grid initialized with cell_size={cell_size:.2f}px")
    
    # Statistics
    total_real = 0
    total_matched = 0
    total_roi_real = 0
    total_roi_matched = 0
    
    # Progress tracking
    last_progress_time = time.time()
    progress_interval = 2.0  # More frequent updates
    
    # Process real events in micro-buckets
    print(f"    [DEBUG] Loading first batches...")
    real_batch = next(real_streamer, None)
    pred_batch = next(pred_streamer, None)
    
    if real_batch is None:
        print(f"    [DEBUG] No real events found!")
        return 0.0, 0, 0
    
    print(f"    [DEBUG] Real batch loaded: {len(real_batch):,} events")
    print(f"    [DEBUG] Pred batch loaded: {len(pred_batch):,} events" if pred_batch is not None else "    [DEBUG] No pred events found!")
    
    # Convert predicted times: u = tp - dt
    if pred_batch is not None:
        pred_batch['t'] = pred_batch['t'] - dt_seconds
        print(f"    [DEBUG] Predicted times adjusted by dt={dt_seconds:.3f}s")
    
    real_idx = 0
    pred_idx = 0
    
    while real_batch is not None and real_idx < len(real_batch):
        # Get current real event
        real_event = real_batch[real_idx]
        real_time = real_event['t']
        
        # Check if in ROI
        in_roi = circle_mask(real_event['x'], real_event['y'], 
                           DISC_CENTER_X, DISC_CENTER_Y, DISC_RADIUS)
        if in_roi:
            total_roi_real += 1
        
        total_real += 1
        
        # Update temporal window: add predicted events within [tr - εt, tr + εt]
        while pred_batch is not None and pred_idx < len(pred_batch):
            pred_event = pred_batch[pred_idx]
            pred_time = pred_event['t']
            
            if pred_time <= real_time + temporal_tol_s:
                # Add to grid if within temporal window
                if abs(pred_time - real_time) <= temporal_tol_s:
                    grid.insert(pred_event)
                pred_idx += 1
            else:
                break
        
        # Evict old predicted events
        grid.evict_until(real_time - temporal_tol_s)
        
        # Find best match for current real event (no ROI prefilter)
        grid_key = grid.key(real_event['x'], real_event['y'])
        candidates = grid.get_neighbors(grid_key, spatial_tol)
        
        best_idx = MatchingKernel.best_match(real_event, candidates, spatial_tol, temporal_tol_s)
        
        if best_idx >= 0:
            # Mark as matched
            matched_pred = candidates[best_idx]
            grid.remove_event(matched_pred)
            total_matched += 1
            if in_roi:
                total_roi_matched += 1
        
        real_idx += 1
        
        # Progress heartbeat
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            rate = (total_roi_matched / total_roi_real * 100.0) if total_roi_real > 0 else 0.0
            print(f"      [PROGRESS] {total_real:,} real events processed, {total_roi_real:,} ROI events, {total_roi_matched:,} matches ({rate:.1f}%)")
            last_progress_time = current_time
        
        # Load next batch if needed
        if real_idx >= len(real_batch):
            real_batch = next(real_streamer, None)
            real_idx = 0
        
        if pred_batch is not None and pred_idx >= len(pred_batch):
            pred_batch = next(pred_streamer, None)
            pred_idx = 0
            if pred_batch is not None:
                pred_batch['t'] = pred_batch['t'] - dt_seconds
    
    # Calculate cancellation rate
    roi_cancellation_rate = (total_roi_matched / total_roi_real * 100.0) if total_roi_real > 0 else 0.0
    
    print(f"    [DEBUG] Stream cancellation completed:")
    print(f"    [DEBUG]   Total real events: {total_real:,}")
    print(f"    [DEBUG]   Total ROI real events: {total_roi_real:,}")
    print(f"    [DEBUG]   Total ROI matches: {total_roi_matched:,}")
    print(f"    [DEBUG]   Cancellation rate: {roi_cancellation_rate:.2f}%")
    
    return roi_cancellation_rate, total_roi_real, total_roi_matched

def analyze_dt_tolerance_combinations():
    """Main streaming analysis function - scalable for 43M+ events"""
    print("=== Streaming DT and Tolerance Analysis (Scalable) ===")
    
    # Generate tolerance values
    spatial_values = np.arange(SPATIAL_TOLERANCE_RANGE[0], 
                              SPATIAL_TOLERANCE_RANGE[1] + SPATIAL_TOLERANCE_RANGE[2], 
                              SPATIAL_TOLERANCE_RANGE[2])
    temporal_values = np.arange(TEMPORAL_TOLERANCE_RANGE[0], 
                               TEMPORAL_TOLERANCE_RANGE[1] + TEMPORAL_TOLERANCE_RANGE[2], 
                               TEMPORAL_TOLERANCE_RANGE[2])
    
    print(f"DT values: {DT_VALUES_MS}")
    print(f"Spatial tolerances: {spatial_values}")
    print(f"Temporal tolerances: {temporal_values}")
    print(f"Stream batch size: {STREAM_BATCH_SIZE:,}")
    print(f"Numba acceleration: {HAS_NUMBA}")
    
    total_combinations = len(DT_VALUES_MS) * len(spatial_values) * len(temporal_values)
    print(f"Total combinations: {total_combinations}")
    
    # Initialize results
    results = []
    combination_count = 0
    start_time = time.time()
    
    # Process each window
    for window_idx, window in enumerate(WINDOWS):
        t0, t1 = window
        print(f"\nProcessing Window {window_idx + 1}: {t0:.3f}s to {t1:.3f}s")
        
        # Resolve window files
        window_dir = get_window_dir(WINDOW_PREDICTIONS_DIR, window_idx, t0, t1)
        real_path = os.path.join(window_dir, "real_events.npy")
        
        if not os.path.exists(real_path):
            print(f"  Error: Missing {real_path}")
            continue
            
        print(f"  Using streaming approach - no full data loading")
        
        # Process each dt value
        for dt_idx, dt_ms in enumerate(DT_VALUES_MS):
            print(f"  [DT {dt_idx + 1}/{len(DT_VALUES_MS)}] Processing dt={dt_ms}ms...")
            
            # Check if predicted events file exists
            pred_path = os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npy")
            if not os.path.exists(pred_path):
                print(f"    [ERROR] Missing {pred_path}")
                continue
            
            print(f"    [DEBUG] Found pred file: {pred_path}")
            pred_size = os.path.getsize(pred_path) / (1024*1024)  # MB
            print(f"    [DEBUG] Pred file size: {pred_size:.1f} MB")
        
            dt_seconds = dt_ms * 1e-3
            
            # Test all tolerance combinations
            for spatial_tol in spatial_values:
                for temporal_tol in temporal_values:
                    combination_count += 1
                    elapsed_time = time.time() - start_time
                    print(f"    [COMBO {combination_count}/{total_combinations}] Testing spatial={spatial_tol:.1f}px, temporal={temporal_tol:.1f}ms... (elapsed: {elapsed_time:.1f}s)")
                    
                    # Create streamers for this combination
                    try:
                        real_streamer = EventStreamer(real_path, STREAM_BATCH_SIZE)
                        pred_streamer = EventStreamer(pred_path, STREAM_BATCH_SIZE)
                        
                        # Run streaming cancellation
                        roi_cancellation_rate, total_roi_real, total_roi_matched = stream_cancellation(
                            real_streamer, pred_streamer, dt_seconds, spatial_tol, temporal_tol
                        )
                        
                        # Store results
                        results.append({
                            'dt_ms': dt_ms,
                            'window_idx': window_idx + 1,
                            'window_start': t0,
                            'window_end': t1,
                            'spatial_tolerance': spatial_tol,
                            'temporal_tolerance': temporal_tol,
                            'cancellation_rate': roi_cancellation_rate,
                            'total_roi_real': total_roi_real,
                            'total_roi_cancelled': total_roi_matched,
                            'total_matched_pairs': total_roi_matched
                        })
                        
                        print(f"      Rate: {roi_cancellation_rate:.2f}% ({total_roi_matched:,}/{total_roi_real:,})")
                    except Exception as e:
                        print(f"      Error: {e}")
                        continue
                    finally:
                        # Clean up streamers
                        try:
                            real_streamer.close()
                            pred_streamer.close()
                        except Exception:
                            pass
                        gc.collect()
            
            print(f"    Completed dt={dt_ms}ms")
    
    return results

def create_simple_plots(results_df, output_dir):
    """Create organized plots: overall and per-window dt vs cancellation, plus heatmaps."""
    os.makedirs(output_dir, exist_ok=True)

    # Overall dt vs cancellation (averaged over windows and tolerances)
    fig, ax = plt.subplots(figsize=(10, 6))
    dt_avg = results_df.groupby('dt_ms')['cancellation_rate'].mean().sort_index()
    ax.plot(dt_avg.index, dt_avg.values, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('dt (ms)')
    ax.set_ylabel('Cancellation Rate (%)')
    ax.set_title('Overall: Cancellation Rate vs dt (avg over windows & tolerances)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_dt_vs_cancellation_rate.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save overall dt vs CSV
    dt_avg.reset_index().to_csv(os.path.join(output_dir, "overall_dt_vs.csv"), index=False)

    # Per-window dt vs cancellation (averaged over tolerances)
    for (widx, wstart, wend), wdf in results_df.groupby(['window_idx', 'window_start', 'window_end']):
        window_dir_name = f"window_{int(widx)}_{float(wstart):.3f}s_to_{float(wend):.3f}s"
        window_out_dir = os.path.join(output_dir, window_dir_name)
        os.makedirs(window_out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        w_dt_avg = wdf.groupby('dt_ms')['cancellation_rate'].mean().sort_index()
        ax.plot(w_dt_avg.index, w_dt_avg.values, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('dt (ms)')
        ax.set_ylabel('Cancellation Rate (%)')
        ax.set_title(f'Window {int(widx)}: {float(wstart):.3f}s to {float(wend):.3f}s')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(window_out_dir, "dt_vs_cancellation_rate.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # Save per-window CSV
        w_dt_avg.reset_index().to_csv(os.path.join(window_out_dir, "dt_vs.csv"), index=False)

        # Heatmap at best dt (per window)
        best_dt = w_df_dtmax = w_dt_avg.idxmax()
        best_dt_data = wdf[wdf['dt_ms'] == best_dt]
        pivot_data = best_dt_data.groupby(['spatial_tolerance', 'temporal_tolerance'])['cancellation_rate'].mean().reset_index()
        pivot_table = pivot_data.pivot_table(values='cancellation_rate', index='temporal_tolerance', columns='spatial_tolerance')
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot_table.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels(pivot_table.columns)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels(pivot_table.index)
        ax.set_xlabel('Spatial Tolerance (pixels)')
        ax.set_ylabel('Temporal Tolerance (ms)')
        ax.set_title(f'Window {int(widx)} Heatmap (best dt = {int(best_dt)}ms)')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cancellation Rate (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(window_out_dir, f"tolerance_heatmap_dt_{int(best_dt)}ms.png"), dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved plots to {output_dir}")

def run_one_combination(args):
    """Worker for multiprocessing: runs one (dt, spatial, temporal, window)."""
    (window_idx, t0, t1, dt_ms, spatial_tol, temporal_tol, base_dir) = args
    print(f"[WORKER] Starting: dt={dt_ms}ms, spatial={spatial_tol}px, temporal={temporal_tol}ms")
    try:
        window_dir = get_window_dir(base_dir, window_idx, t0, t1)
        real_path = os.path.join(window_dir, "real_events.npy")
        pred_path = os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npy")
        if not os.path.exists(real_path) or not os.path.exists(pred_path):
            print(f"[WORKER] Missing files: real={os.path.exists(real_path)}, pred={os.path.exists(pred_path)}")
            return None
        dt_seconds = dt_ms * 1e-3
        print(f"[WORKER] Creating streamers...")
        real_streamer = EventStreamer(real_path, STREAM_BATCH_SIZE)
        pred_streamer = EventStreamer(pred_path, STREAM_BATCH_SIZE)
        print(f"[WORKER] Running stream_cancellation...")
        rate, total_roi_real, total_roi_matched = stream_cancellation(
            real_streamer, pred_streamer, dt_seconds, spatial_tol, temporal_tol
        )
        print(f"[WORKER] Completed: rate={rate:.2f}%")
        real_streamer.close(); pred_streamer.close(); gc.collect()
        return {
            'dt_ms': dt_ms,
            'window_idx': window_idx + 1,
            'window_start': t0,
            'window_end': t1,
            'spatial_tolerance': spatial_tol,
            'temporal_tolerance': temporal_tol,
            'cancellation_rate': rate,
            'total_roi_real': total_roi_real,
            'total_roi_cancelled': total_roi_matched,
            'total_matched_pairs': total_roi_matched
        }
    except Exception as e:
        return {'error': str(e), 'dt_ms': dt_ms, 'window_idx': window_idx + 1,
                'spatial_tolerance': spatial_tol, 'temporal_tolerance': temporal_tol}

def find_best_parameters(results_df):
    """Find the best parameter combinations"""
    print("\n=== Best Parameters ===")
    
    # Best overall combination
    best_overall = results_df.loc[results_df['cancellation_rate'].idxmax()]
    print(f"Best overall: dt={int(best_overall['dt_ms'])}ms, "
          f"spatial={best_overall['spatial_tolerance']:.1f}px, "
          f"temporal={best_overall['temporal_tolerance']:.1f}ms, "
          f"rate={best_overall['cancellation_rate']:.2f}%")
    
    # Best dt (averaged across all tolerances)
    dt_avg = results_df.groupby('dt_ms')['cancellation_rate'].mean()
    best_dt = dt_avg.idxmax()
    print(f"Best dt (averaged): {int(best_dt)}ms ({dt_avg[best_dt]:.2f}%)")
    
    return best_overall

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("=== Scalable DT and Tolerance Analysis ===")
    print(f"DT values: {DT_VALUES_MS}")
    print(f"Spatial range: {SPATIAL_TOLERANCE_RANGE[0]} to {SPATIAL_TOLERANCE_RANGE[1]} (step: {SPATIAL_TOLERANCE_RANGE[2]})")
    print(f"Temporal range: {TEMPORAL_TOLERANCE_RANGE[0]} to {TEMPORAL_TOLERANCE_RANGE[1]} (step: {TEMPORAL_TOLERANCE_RANGE[2]})")
    print(f"Polarity mode: {POLARITY_MODE}")
    print(f"ROI: Circle center ({DISC_CENTER_X:.1f}, {DISC_CENTER_Y:.1f}), radius {DISC_RADIUS:.0f}px")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Streaming batch size: {STREAM_BATCH_SIZE:,}")
    print(f"Numba acceleration: {HAS_NUMBA}")
    print(f"Processing: Full dataset (streaming, no subsampling)")
    
    # Run analysis (optionally with multiprocessing)
    use_multiprocessing = True
    if use_multiprocessing:
        try:
            from multiprocessing import Pool, cpu_count
            print("\nRunning in multiprocessing mode")
            tasks = []
            for window_idx, (t0, t1) in enumerate(WINDOWS):
                for dt_ms in DT_VALUES_MS:
                    window_dir = get_window_dir(WINDOW_PREDICTIONS_DIR, window_idx, t0, t1)
                    real_path = os.path.join(window_dir, "real_events.npy")
                    pred_path = os.path.join(window_dir, f"pred_events_dt_{dt_ms:02d}ms.npy")
                    if not os.path.exists(real_path) or not os.path.exists(pred_path):
                        continue
                    spatial_values = np.arange(SPATIAL_TOLERANCE_RANGE[0], SPATIAL_TOLERANCE_RANGE[1] + SPATIAL_TOLERANCE_RANGE[2], SPATIAL_TOLERANCE_RANGE[2])
                    temporal_values = np.arange(TEMPORAL_TOLERANCE_RANGE[0], TEMPORAL_TOLERANCE_RANGE[1] + TEMPORAL_TOLERANCE_RANGE[2], TEMPORAL_TOLERANCE_RANGE[2])
                    for spatial_tol in spatial_values:
                        for temporal_tol in temporal_values:
                            tasks.append((window_idx, t0, t1, dt_ms, spatial_tol, temporal_tol, WINDOW_PREDICTIONS_DIR))
            workers = min(cpu_count(), 8)  # Reduced for EC2 stability
            print(f"Dispatching {len(tasks)} combinations to {workers} workers...")
            with Pool(processes=workers) as pool:
                results_list = pool.map(run_one_combination, tasks)
            results = [r for r in results_list if r and 'error' not in r]
            errors = [r for r in results_list if r and 'error' in r]
            if errors:
                print(f"Encountered {len(errors)} task errors; first: {errors[0]['error']}")
        except Exception as e:
            print(f"Multiprocessing failed, falling back to single process: {e}")
            results = analyze_dt_tolerance_combinations()
    else:
        results = analyze_dt_tolerance_combinations()
    
    if not results:
        print("No results generated. Check if prediction data exists.")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_filename = os.path.join(OUTPUT_DIR, "streaming_analysis_results.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"\nSaved results: {csv_filename}")
    
    # Create plots
    print("\nCreating plots...")
    create_simple_plots(results_df, OUTPUT_DIR)
    
    # Find best parameters
    best_parameters = find_best_parameters(results_df)
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total combinations tested: {len(results_df)}")
    print(f"Average cancellation rate: {results_df['cancellation_rate'].mean():.2f}%")
    print(f"Best cancellation rate: {results_df['cancellation_rate'].max():.2f}%")
    print(f"Worst cancellation rate: {results_df['cancellation_rate'].min():.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal analysis time: {elapsed_time:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()