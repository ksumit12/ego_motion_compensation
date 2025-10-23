#!/usr/bin/env python3
"""
Fix the combined_events_with_predictions.npy file by properly interleaving real and predicted events by time.
The current file has all real events first, then all predicted events, which breaks the video generation.
"""
import numpy as np
import os

def fix_combined_events():
    input_path = "./combined_events_with_predictions.npy"
    output_path = "./combined_events_with_predictions_fixed.npy"
    
    print("Loading combined events...")
    data = np.load(input_path, mmap_mode='r')
    
    print(f"Total events: {len(data):,}")
    print(f"Real events: {np.sum(data[:, 4] == 0):,}")
    print(f"Predicted events: {np.sum(data[:, 4] == 1):,}")
    
    # Split into real and predicted
    real_events = data[data[:, 4] == 0]
    pred_events = data[data[:, 4] == 1]
    
    print(f"Real time range: {real_events[:, 3].min():.6f}s to {real_events[:, 3].max():.6f}s")
    print(f"Pred time range: {pred_events[:, 3].min():.6f}s to {pred_events[:, 3].max():.6f}s")
    
    # Combine and sort by time
    print("Combining and sorting by time...")
    combined = np.vstack([real_events, pred_events])
    combined = combined[np.argsort(combined[:, 3])]
    
    print(f"Fixed data shape: {combined.shape}")
    print(f"Time range: {combined[:, 3].min():.6f}s to {combined[:, 3].max():.6f}s")
    
    # Verify the fix
    print("Verifying fix...")
    sample_size = 10000
    sample = combined[:sample_size]
    real_in_sample = np.sum(sample[:, 4] == 0)
    pred_in_sample = np.sum(sample[:, 4] == 1)
    print(f"Sample ({sample_size} events): {real_in_sample} real, {pred_in_sample} predicted")
    
    # Save the fixed file
    print(f"Saving fixed file to {output_path}...")
    np.save(output_path, combined)
    
    print("Done! Now update visualize_event_comparison.py to use the fixed file.")
    print(f"Change COMBINED_PATH to: {output_path}")

if __name__ == "__main__":
    fix_combined_events()












