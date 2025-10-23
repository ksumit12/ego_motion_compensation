#!/usr/bin/env python3
"""
Quick KDTree vs Uniform Grid Performance Test
Simple benchmark on your actual event data
"""

import time
import numpy as np
import os
from collections import defaultdict
from math import floor

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("SciPy not available - install with: pip install scipy")

class SimpleUniformGrid:
    """Simple uniform grid for comparison"""
    
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
    
    def key(self, x, y):
        return (int(floor(x / self.cell_size)), int(floor(y / self.cell_size)))
    
    def insert(self, x, y):
        k = self.key(x, y)
        self.grid[k].append((x, y))
    
    def query_radius(self, qx, qy, radius):
        """Find all points within radius of query point"""
        grid_key = self.key(qx, qy)
        candidates = []
        
        # Check 3x3 neighborhood
        gx, gy = grid_key
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (gx + dx, gy + dy)
                bucket = self.grid.get(key, [])
                candidates.extend(bucket)
        
        # Filter by actual distance
        results = []
        for cx, cy in candidates:
            dist2 = (cx - qx)**2 + (cy - qy)**2
            if dist2 <= radius**2:
                results.append((cx, cy))
        
        return results

def quick_test():
    """Quick performance test"""
    print("=== Quick KDTree vs Uniform Grid Test ===")
    
    # Try to find your data
    possible_paths = [
        "/home/sumit/anu_research/ego_motion/window_predictions/window_2_8.200s_to_8.210s/real_events.npy",
        "/home/sumit/anu_research/ego_motion/window_predictions/window_3_9.000s_to_9.010s/real_events.npy",
        "./window_predictions/window_2_8.200s_to_8.210s/real_events.npy",
        "./window_predictions/window_3_9.000s_to_9.010s/real_events.npy"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("Could not find event data. Creating synthetic data for test...")
        # Create synthetic data
        np.random.seed(42)
        data = np.random.rand(100000, 4)  # x, y, p, t
        data[:, 0] *= 1280  # x: 0-1280
        data[:, 1] *= 720   # y: 0-720
        data[:, 2] = np.random.choice([-1, 1], len(data))  # polarity
        data[:, 3] = np.random.rand(len(data)) * 5.0  # time: 0-5s
    else:
        print(f"Using data from: {data_path}")
        # Load actual data
        full_data = np.load(data_path, mmap_mode='r')
        # Take a sample for testing
        if len(full_data) > 100000:
            indices = np.random.choice(len(full_data), 100000, replace=False)
            data = full_data[indices]
        else:
            data = full_data
    
    print(f"Testing with {len(data):,} events")
    
    # Extract coordinates
    coords = data[:, :2]  # x, y coordinates
    
    # Test parameters
    spatial_tolerances = [1.0, 2.0, 5.0, 10.0]
    num_queries = 1000
    
    # Generate query points
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    
    query_x = np.random.uniform(x_min, x_max, num_queries)
    query_y = np.random.uniform(y_min, y_max, num_queries)
    query_points = np.column_stack([query_x, query_y])
    
    print(f"Testing {num_queries:,} queries at different spatial tolerances")
    print()
    
    results = []
    
    for spatial_tol in spatial_tolerances:
        print(f"Spatial tolerance: {spatial_tol}px")
        
        # Test KDTree
        if HAS_SCIPY:
            # Build KDTree
            build_start = time.time()
            tree = cKDTree(coords)
            build_time = time.time() - build_start
            
            # Query KDTree
            query_start = time.time()
            kdtree_results = tree.query_ball_point(query_points, spatial_tol)
            query_time = time.time() - query_start
            
            kdtree_total = build_time + query_time
            kdtree_matches = sum(len(matches) for matches in kdtree_results)
            
            print(f"  KDTree: {kdtree_total:.4f}s ({kdtree_matches:,} matches)")
        else:
            kdtree_total = None
            kdtree_matches = 0
            print(f"  KDTree: Not available (SciPy missing)")
        
        # Test Uniform Grid
        # Build Grid
        build_start = time.time()
        grid = SimpleUniformGrid(spatial_tol)
        
        for x, y in coords:
            grid.insert(x, y)
        build_time = time.time() - build_start
        
        # Query Grid
        query_start = time.time()
        grid_matches = 0
        for qx, qy in query_points:
            matches = grid.query_radius(qx, qy, spatial_tol)
            grid_matches += len(matches)
        query_time = time.time() - query_start
        
        grid_total = build_time + query_time
        
        print(f"  Grid:   {grid_total:.4f}s ({grid_matches:,} matches)")
        
        # Compare results
        if kdtree_total is not None:
            speedup = kdtree_total / grid_total
            winner = "Grid" if speedup > 1 else "KDTree"
            print(f"  Winner: {winner} ({speedup:.2f}x faster)")
            
            results.append({
                'spatial_tol': spatial_tol,
                'kdtree_time': kdtree_total,
                'grid_time': grid_total,
                'speedup': speedup,
                'winner': winner
            })
        
        print()
    
    # Summary
    if results:
        print("=== Summary ===")
        avg_speedup = np.mean([r['speedup'] for r in results])
        grid_wins = sum(1 for r in results if r['winner'] == 'Grid')
        kdtree_wins = sum(1 for r in results if r['winner'] == 'KDTree')
        
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Grid wins: {grid_wins}/{len(results)} tests")
        print(f"KDTree wins: {kdtree_wins}/{len(results)} tests")
        
        if avg_speedup > 1:
            print("→ Uniform Grid is generally faster for this data")
        else:
            print("→ KDTree is generally faster for this data")
    
    print("\nTest completed!")

if __name__ == "__main__":
    quick_test()
