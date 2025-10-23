#!/usr/bin/env python3
"""
Benchmark: KDTree vs Uniform Grid for Spatial Indexing
Tests performance on actual event data from window_predictions folder
"""

import time
import numpy as np
import os
from collections import defaultdict, deque
from math import floor
import gc
import psutil

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("SciPy not available - KDTree tests will be skipped")

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available - using pure NumPy")

# Configuration
WINDOW_PREDICTIONS_DIR = "/media/sumit/New Volume/window_predictions_5s"
TEST_SPATIAL_TOLERANCES = [1.0, 2.0, 3.0, 5.0, 10.0]  # pixels
TEST_QUERY_COUNTS = [1000, 5000, 10000, 50000]  # number of queries to test
BATCH_SIZE = 1000000  # events per batch for streaming (1M events)

class UniformSpatialGrid:
    """Uniform spatial grid for O(1) neighborhood queries"""
    
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)  # (gx, gy) -> list of events
        self.fifo = deque()  # FIFO queue for temporal eviction
        
    def key(self, x, y):
        """Get grid cell key for coordinates"""
        return (int(floor(x / self.cell_size)), int(floor(y / self.cell_size)))
    
    def insert(self, event):
        """Insert event into grid"""
        k = self.key(event[0], event[1])  # event = [x, y, p, t]
        self.grid[k].append(event)
        self.fifo.append(event)
    
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
    
    def query_radius(self, query_points, radius):
        """Query all points within radius of query_points"""
        results = []
        
        for qx, qy in query_points:
            grid_key = self.key(qx, qy)
            candidates = self.get_neighbors(grid_key, radius)
            
            # Filter by actual distance
            valid_candidates = []
            for candidate in candidates:
                dx = candidate[0] - qx
                dy = candidate[1] - qy
                dist2 = dx * dx + dy * dy
                if dist2 <= radius * radius:
                    valid_candidates.append(candidate)
            
            results.append(valid_candidates)
        
        return results
    
    def clear(self):
        """Clear all data"""
        self.grid.clear()
        self.fifo.clear()

def load_event_batch(file_path, batch_size=BATCH_SIZE):
    """Load a batch of events from file"""
    if not os.path.exists(file_path):
        return None
    
    data = np.load(file_path, mmap_mode='r')
    total_events = len(data)
    
    # Take a random sample for testing
    if total_events > batch_size:
        indices = np.random.choice(total_events, batch_size, replace=False)
        batch = data[indices]
    else:
        batch = data
    
    return batch

def generate_query_points(data, num_queries):
    """Generate random query points within the data bounds"""
    if len(data) == 0:
        return np.array([])
    
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    
    query_x = np.random.uniform(x_min, x_max, num_queries)
    query_y = np.random.uniform(y_min, y_max, num_queries)
    
    return np.column_stack([query_x, query_y])

def benchmark_kdtree(data, query_points, radius):
    """Benchmark KDTree performance"""
    if not HAS_SCIPY or len(data) == 0:
        return None, None
    
    # Build KDTree
    build_start = time.time()
    tree = cKDTree(data[:, :2])  # Only x, y coordinates
    build_time = time.time() - build_start
    
    # Query KDTree
    query_start = time.time()
    results = tree.query_ball_point(query_points, radius)
    query_time = time.time() - build_start  # Total time including build
    
    # Count results
    total_matches = sum(len(matches) for matches in results)
    
    return build_time, query_time, total_matches

def benchmark_uniform_grid(data, query_points, radius):
    """Benchmark Uniform Grid performance"""
    if len(data) == 0:
        return None, None, None
    
    # Build Uniform Grid
    build_start = time.time()
    grid = UniformSpatialGrid(radius)  # cell_size = radius
    
    for event in data:
        grid.insert(event)
    build_time = time.time() - build_start
    
    # Query Uniform Grid
    query_start = time.time()
    results = grid.query_radius(query_points, radius)
    query_time = time.time() - build_start  # Total time including build
    
    # Count results
    total_matches = sum(len(matches) for matches in results)
    
    return build_time, query_time, total_matches

def run_comprehensive_benchmark():
    """Run comprehensive benchmark on actual data"""
    print("=== KDTree vs Uniform Grid Benchmark ===")
    print(f"Testing on data from: {WINDOW_PREDICTIONS_DIR}")
    
    # Find available data files
    window_dir = os.path.join(WINDOW_PREDICTIONS_DIR, "window_1_5.000s_to_10.000s")
    
    real_path = os.path.join(window_dir, "real_events.npy")
    pred_path = os.path.join(window_dir, "pred_events_dt_01ms.npy")
    
    if not os.path.exists(real_path):
        print(f"Error: {real_path} not found")
        return
    
    if not os.path.exists(pred_path):
        print(f"Error: {pred_path} not found")
        return
    
    print(f"Real events: {real_path}")
    print(f"Predicted events: {pred_path}")
    
    # Load test data
    print("\nLoading test data...")
    print("WARNING: This will load 43M events - ensure you have sufficient RAM!")
    real_data = load_event_batch(real_path, BATCH_SIZE)
    pred_data = load_event_batch(pred_path, BATCH_SIZE)
    
    if real_data is None or pred_data is None:
        print("Error: Could not load data")
        return
    
    print(f"Loaded {len(real_data):,} real events")
    print(f"Loaded {len(pred_data):,} predicted events")
    
    # Results storage
    results = []
    
    # Test different spatial tolerances and query counts
    for spatial_tol in TEST_SPATIAL_TOLERANCES:
        print(f"\n--- Testing Spatial Tolerance: {spatial_tol}px ---")
        
        for num_queries in TEST_QUERY_COUNTS:
            print(f"  Testing {num_queries:,} queries...")
            
            # Generate query points
            query_points = generate_query_points(real_data, num_queries)
            
            if len(query_points) == 0:
                continue
            
            # Test KDTree
            kdtree_build, kdtree_total, kdtree_matches = benchmark_kdtree(
                pred_data, query_points, spatial_tol
            )
            
            # Test Uniform Grid
            grid_build, grid_total, grid_matches = benchmark_uniform_grid(
                pred_data, query_points, spatial_tol
            )
            
            # Store results
            result = {
                'spatial_tolerance': spatial_tol,
                'num_queries': num_queries,
                'num_events': len(pred_data),
                'kdtree_build_time': kdtree_build,
                'kdtree_total_time': kdtree_total,
                'kdtree_matches': kdtree_matches,
                'grid_build_time': grid_build,
                'grid_total_time': grid_total,
                'grid_matches': grid_matches
            }
            
            results.append(result)
            
            # Print results
            if kdtree_total is not None and grid_total is not None:
                speedup = kdtree_total / grid_total
                print(f"    KDTree: {kdtree_total:.4f}s ({kdtree_matches:,} matches)")
                print(f"    Grid:   {grid_total:.4f}s ({grid_matches:,} matches)")
                print(f"    Speedup: {speedup:.2f}x {'Grid' if speedup > 1 else 'KDTree'}")
            elif kdtree_total is not None:
                print(f"    KDTree: {kdtree_total:.4f}s ({kdtree_matches:,} matches)")
                print(f"    Grid:   Not available")
            elif grid_total is not None:
                print(f"    KDTree: Not available")
                print(f"    Grid:   {grid_total:.4f}s ({grid_matches:,} matches)")
            
            # Cleanup
            gc.collect()
    
    # Summary analysis
    print("\n=== Summary Analysis ===")
    
    valid_results = [r for r in results if r['kdtree_total_time'] is not None and r['grid_total_time'] is not None]
    
    if valid_results:
        kdtree_times = [r['kdtree_total_time'] for r in valid_results]
        grid_times = [r['grid_total_time'] for r in valid_results]
        
        avg_kdtree = np.mean(kdtree_times)
        avg_grid = np.mean(grid_times)
        avg_speedup = avg_kdtree / avg_grid
        
        print(f"Average KDTree time: {avg_kdtree:.4f}s")
        print(f"Average Grid time:   {avg_grid:.4f}s")
        print(f"Average speedup:     {avg_speedup:.2f}x {'Grid' if avg_speedup > 1 else 'KDTree'}")
        
        # Best cases
        best_grid_speedup = max(valid_results, key=lambda x: x['kdtree_total_time'] / x['grid_total_time'])
        best_kdtree_speedup = min(valid_results, key=lambda x: x['kdtree_total_time'] / x['grid_total_time'])
        
        print(f"\nBest Grid speedup: {best_grid_speedup['kdtree_total_time'] / best_grid_speedup['grid_total_time']:.2f}x")
        print(f"  (spatial_tol={best_grid_speedup['spatial_tolerance']}px, queries={best_grid_speedup['num_queries']:,})")
        
        print(f"Best KDTree speedup: {best_kdtree_speedup['kdtree_total_time'] / best_kdtree_speedup['grid_total_time']:.2f}x")
        print(f"  (spatial_tol={best_kdtree_speedup['spatial_tolerance']}px, queries={best_kdtree_speedup['num_queries']:,})")
    
    # Memory usage analysis
    print(f"\n=== Memory Usage Analysis ===")
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")
    
    # Save detailed results
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        output_file = "spatial_indexing_benchmark_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

def test_streaming_scenario():
    """Test streaming scenario with temporal window"""
    print("\n=== Streaming Scenario Test ===")
    
    window_dir = os.path.join(WINDOW_PREDICTIONS_DIR, "window_1_5.000s_to_10.000s")
    real_path = os.path.join(window_dir, "real_events.npy")
    pred_path = os.path.join(window_dir, "pred_events_dt_01ms.npy")
    
    if not os.path.exists(real_path) or not os.path.exists(pred_path):
        print("Data files not found for streaming test")
        return
    
    # Load small batches for streaming test (sample from 43M)
    real_data = load_event_batch(real_path, 100000)  # 100K events for streaming test
    pred_data = load_event_batch(pred_path, 100000)  # 100K events for streaming test
    
    if real_data is None or pred_data is None:
        return
    
    spatial_tol = 3.0
    temporal_tol = 5.0  # ms
    
    print(f"Testing streaming scenario with {len(real_data):,} real and {len(pred_data):,} predicted events")
    print(f"Spatial tolerance: {spatial_tol}px, Temporal tolerance: {temporal_tol}ms")
    
    # Simulate streaming: process real events one by one, maintain predicted window
    grid = UniformSpatialGrid(spatial_tol)
    
    # Add predicted events to grid (simulating temporal window)
    for event in pred_data:
        grid.insert(event)
    
    # Process real events
    matches_found = 0
    start_time = time.time()
    
    for i, real_event in enumerate(real_data):
        if i % 10000 == 0:
            print(f"  Processed {i:,} real events, found {matches_found:,} matches")
        
        # Query for matches
        grid_key = grid.key(real_event[0], real_event[1])
        candidates = grid.get_neighbors(grid_key, spatial_tol)
        
        # Simple matching (find first valid match)
        for candidate in candidates:
            dx = candidate[0] - real_event[0]
            dy = candidate[1] - real_event[1]
            dist2 = dx * dx + dy * dy
            
            if dist2 <= spatial_tol * spatial_tol:
                matches_found += 1
                break  # One match per real event
    
    total_time = time.time() - start_time
    
    print(f"Streaming test completed:")
    print(f"  Time: {total_time:.2f}s")
    print(f"  Events processed: {len(real_data):,}")
    print(f"  Matches found: {matches_found:,}")
    print(f"  Rate: {len(real_data) / total_time:.0f} events/second")

def main():
    """Main benchmark execution"""
    print("Spatial Indexing Benchmark")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(WINDOW_PREDICTIONS_DIR):
        print(f"Error: Data directory not found: {WINDOW_PREDICTIONS_DIR}")
        print("Please update WINDOW_PREDICTIONS_DIR in the script")
        return
    
    # Run comprehensive benchmark
    run_comprehensive_benchmark()
    
    # Test streaming scenario
    test_streaming_scenario()
    
    print("\nBenchmark completed!")

if __name__ == "__main__":
    main()
