import numpy as np
from skimage.measure import ransac, CircleModel
import pandas as pd
import os

def load_event_data(path):
    """Load event data and convert time from microseconds to seconds, optimized version."""
    print(f"    Analyzing file size...")
    file_size = os.path.getsize(path) / (1024**2)  # Size in MB
    print(f"    File size: {file_size:.1f} MB")
    
    try:
        # Try fast pandas loading first
        print(f"   ⚡ Using optimized pandas loading...")
        df = pd.read_csv(path, names=['x', 'y', 'polarity', 'timestamp'], 
                        dtype={'x': np.float32, 'y': np.float32, 
                               'polarity': np.int8, 'timestamp': np.float64})
        
        # Convert timestamp from microseconds to seconds
        df['timestamp'] = df['timestamp'] / 1e6
        
        # Convert to numpy array
        events = df.values.astype(np.float32)
        print(f"   Loaded {len(events):,} events successfully")
        return events
        
    except Exception as e:
        print(f"    Fast loading failed ({e}), falling back to line-by-line parsing...")
        
        # Fallback to original method with progress
        clean_lines = []
        total_lines = 0
        
        # Count lines first for progress bar
        with open(path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"    Processing {total_lines:,} lines...")
        
        skipped_lines = 0
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i % 1000000 == 0 and i > 0:  # Progress every 1M lines
                    print(f"      🔄 Processed {i:,}/{total_lines:,} lines ({(i/total_lines)*100:.1f}%)")
                
                parts = line.strip().split(',')
                if len(parts) != 4:
                    skipped_lines += 1
                    continue
                try:
                    x, y, p, t = map(float, parts)
                    clean_lines.append([x, y, p, t / 1e6])  # convert time
                except ValueError:
                    skipped_lines += 1
                    continue
        
        if skipped_lines > 0:
            print(f"     Skipped {skipped_lines:,} invalid lines")
        
        events = np.array(clean_lines, dtype=np.float32)
        print(f"   Loaded {len(events):,} valid events")
        return events

def load_center_from_tracker(path):
    """Load center coordinates using RANSAC circle fitting"""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    points = data[:, 1:3]
    model, _ = ransac(points, CircleModel, min_samples=3, residual_threshold=2.0, max_trials=1000)
    x0, y0, _ = model.params
    return np.array([x0, y0, 0])

def apply_rotation(x, y, center, omega_z, dt):
    """Apply 2D rotation transformation to predict future event position"""
    r = np.array([x, y, 0]) - center
    theta = omega_z * dt
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = rot @ r[:2] + center[:2]
    return rotated 