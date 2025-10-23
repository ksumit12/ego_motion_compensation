#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def check_file_integrity(filename):
    """Check if the numpy file is valid and not corrupted."""
    print(f"Checking file integrity: {filename}")
    
    try:
        # Try loading without pickle first
        events = np.load(filename)
        print(f"✓ File loaded successfully without pickle")
        return events, "normal"
    except ValueError as e:
        if "pickled data" in str(e):
            try:
                # Try loading with pickle
                events = np.load(filename, allow_pickle=True)
                print(f"✓ File loaded successfully with pickle")
                return events, "pickle"
            except Exception as e2:
                print(f"✗ Failed to load with pickle: {e2}")
                return None, "corrupted"
        else:
            print(f"✗ Failed to load: {e}")
            return None, "corrupted"
    except Exception as e:
        print(f"✗ General error loading file: {e}")
        return None, "corrupted"

def analyze_data_structure(events):
    """Analyze the structure and content of the events data."""
    print("\nData Structure Analysis:")
    print(f"Shape: {events.shape}")
    print(f"Data type: {events.dtype}")
    print(f"Memory usage: {events.nbytes / 1e6:.1f} MB")
    
    if len(events.shape) != 2:
        print(f"✗ Expected 2D array, got {len(events.shape)}D")
        return False
    
    if events.shape[1] not in [4, 5]:
        print(f"✗ Expected 4 or 5 columns, got {events.shape[1]}")
        return False
    
    print(f"✓ Valid array structure")
    
    # Check data ranges
    print(f"\nData Ranges:")
    for i, col_name in enumerate(['x', 'y', 'polarity', 'timestamp']):
        if i < events.shape[1]:
            col_data = events[:, i]
            print(f"{col_name}: {col_data.min():.6f} to {col_data.max():.6f}")
            
            # Check for NaN or inf
            if np.any(np.isnan(col_data)):
                print(f"✗ {col_name} contains NaN values")
                return False
            if np.any(np.isinf(col_data)):
                print(f"✗ {col_name} contains infinite values")
                return False
    
    # Check polarity values
    polarity = events[:, 2]
    unique_pol = np.unique(polarity)
    print(f"Polarity values: {unique_pol}")
    if not all(p in [0, 1, -1] for p in unique_pol):
        print(f"⚠ Unusual polarity values found")
    
    # Check timestamps are reasonable
    timestamps = events[:, 3]
    duration = timestamps.max() - timestamps.min()
    print(f"Duration: {duration:.3f} seconds")
    
    if duration <= 0:
        print(f"✗ Invalid duration")
        return False
    
    print(f"✓ Data ranges look reasonable")
    return True

def detect_prediction_structure(events):
    """Try to detect if this contains real+predicted events and how they're organized."""
    print("\nPrediction Structure Detection:")
    
    timestamps = events[:, 3]
    n_events = len(events)
    
    # Method 1: Check if it's split-half (real first, predicted second)
    print("Testing split-half structure...")
    half_point = n_events // 2
    first_half = events[:half_point]
    second_half = events[half_point:]
    
    first_time_range = (first_half[:, 3].min(), first_half[:, 3].max())
    second_time_range = (second_half[:, 3].min(), second_half[:, 3].max())
    
    print(f"First half time: {first_time_range[0]:.6f} to {first_time_range[1]:.6f}")
    print(f"Second half time: {second_time_range[0]:.6f} to {second_time_range[1]:.6f}")
    
    # Check for time offset pattern (predicted events ~5ms later)
    time_offset = second_half[:, 3].mean() - first_half[:, 3].mean()
    print(f"Average time offset: {time_offset:.6f}s")
    
    if 0.004 <= time_offset <= 0.006:  # Around 5ms
        print("✓ Split-half structure detected (real first, predicted second)")
        return "split_half", first_half, second_half
    
    # Method 2: Check if events are interleaved
    print("\nTesting interleaved structure...")
    sorted_events = events[np.argsort(timestamps)]
    time_diffs = np.diff(sorted_events[:, 3])
    
    # Look for 5ms gaps
    dt_5ms = np.abs(time_diffs - 0.005) < 0.001
    dt_5ms_rate = np.mean(dt_5ms)
    print(f"Events with ~5ms spacing: {dt_5ms_rate*100:.1f}%")
    
    if dt_5ms_rate > 0.3:  # At least 30% have 5ms spacing
        print("✓ Interleaved structure detected")
        return "interleaved", None, None
    
    # Method 3: Single event type (all real or all predicted)
    print("\nTesting single event type...")
    print("? Could be all real events or all predicted events")
    return "single", events, None

def validate_predictions(real_events, pred_events):
    """Validate the accuracy of predictions."""
    print("\nValidating Prediction Accuracy:")
    
    if pred_events is None:
        print("No predicted events to validate")
        return
    
    if len(real_events) != len(pred_events):
        print(f"⚠ Different number of real ({len(real_events)}) and predicted ({len(pred_events)}) events")
        # Take the smaller number for comparison
        n_compare = min(len(real_events), len(pred_events))
        real_events = real_events[:n_compare]
        pred_events = pred_events[:n_compare]
    
    # Temporal validation
    print(f"\nTemporal Validation ({len(real_events)} pairs):")
    time_deltas = pred_events[:, 3] - real_events[:, 3]
    
    print(f"Time delta mean: {time_deltas.mean():.6f}s")
    print(f"Time delta std: {time_deltas.std():.6f}s")
    print(f"Time delta range: {time_deltas.min():.6f}s to {time_deltas.max():.6f}s")
    
    # Check how many are close to 5ms
    correct_timing = np.abs(time_deltas - 0.005) < 0.001
    temporal_accuracy = np.mean(correct_timing) * 100
    print(f"Temporal accuracy: {temporal_accuracy:.1f}% within ±1ms of 5ms")
    
    if temporal_accuracy > 95:
        print("🟢 Excellent temporal accuracy")
    elif temporal_accuracy > 80:
        print("🟡 Good temporal accuracy")
    else:
        print("🔴 Poor temporal accuracy")
    
    # Simple spatial validation (just check if positions are reasonable)
    print(f"\nSpatial Validation:")
    real_positions = real_events[:, :2]
    pred_positions = pred_events[:, :2]
    
    # Calculate displacement
    displacements = np.sqrt(np.sum((pred_positions - real_positions)**2, axis=1))
    
    print(f"Position displacement mean: {displacements.mean():.2f} pixels")
    print(f"Position displacement std: {displacements.std():.2f} pixels")
    print(f"Position displacement range: {displacements.min():.2f} to {displacements.max():.2f} pixels")
    
    # Check if displacements are reasonable for motion prediction
    reasonable_disp = displacements < 50  # Less than 50 pixels seems reasonable
    spatial_quality = np.mean(reasonable_disp) * 100
    print(f"Reasonable displacements: {spatial_quality:.1f}% under 50 pixels")
    
    if spatial_quality > 90:
        print("🟢 Spatial displacements look reasonable")
    elif spatial_quality > 70:
        print("🟡 Spatial displacements mostly reasonable")
    else:
        print("🔴 Spatial displacements seem too large")

def create_validation_plots(events, structure_type, real_events=None, pred_events=None):
    """Create plots to visualize the validation results."""
    print("\nCreating validation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Predicted Events Validation', fontsize=16)
    
    # Plot 1: Timestamp distribution
    axes[0, 0].hist(events[:, 3], bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Event Count')
    axes[0, 0].set_title('Timestamp Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spatial distribution
    sample_size = min(10000, len(events))
    sample_indices = np.random.choice(len(events), sample_size, replace=False)
    sample_events = events[sample_indices]
    
    axes[0, 1].scatter(sample_events[:, 0], sample_events[:, 1], s=1, alpha=0.5)
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    axes[0, 1].set_title(f'Spatial Distribution ({sample_size} events)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Polarity distribution
    polarity = events[:, 2]
    unique_pol, counts = np.unique(polarity, return_counts=True)
    axes[1, 0].bar(unique_pol, counts)
    axes[1, 0].set_xlabel('Polarity')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Polarity Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Structure-specific plot
    if structure_type == "split_half" and real_events is not None and pred_events is not None:
        # Plot time delta distribution
        time_deltas = pred_events[:, 3] - real_events[:, 3]
        axes[1, 1].hist(time_deltas * 1000, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(5.0, color='red', linestyle='--', label='Expected: 5ms')
        axes[1, 1].set_xlabel('Time Delta (ms)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Prediction Time Offset')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Just show timestamp vs index
        axes[1, 1].plot(events[:1000, 3], marker='.', markersize=1)
        axes[1, 1].set_xlabel('Event Index')
        axes[1, 1].set_ylabel('Timestamp (s)')
        axes[1, 1].set_title('Timestamp Sequence (first 1000 events)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('predicted_events_validation.png', dpi=150, bbox_inches='tight')
    print("✓ Validation plots saved: predicted_events_validation.png")
    plt.show()

def main():
    import sys
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "predicted_events.npy"
    
    print("PREDICTED EVENTS VALIDATION")
    print("=" * 50)
    
    # Step 1: Check file integrity
    events, file_type = check_file_integrity(filename)
    if events is None:
        print(f"\n❌ VALIDATION FAILED: File {filename} is corrupted or unreadable")
        return
    
    print(f"✓ File is readable (type: {file_type})")
    
    # Step 2: Analyze data structure
    if not analyze_data_structure(events):
        print(f"\n❌ VALIDATION FAILED: Data structure is invalid")
        return
    
    # Step 3: Detect prediction structure
    structure_type, real_events, pred_events = detect_prediction_structure(events)
    print(f"✓ Structure type: {structure_type}")
    
    # Step 4: Validate predictions (if applicable)
    if structure_type == "split_half":
        validate_predictions(real_events, pred_events)
    else:
        print(f"\nSkipping prediction validation (structure: {structure_type})")
    
    # Step 5: Create visualization
    create_validation_plots(events, structure_type, real_events, pred_events)
    
    # Final summary
    print(f"\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print(f"=" * 50)
    print(f"File: {filename}")
    print(f"Status: Valid and readable")
    print(f"Events: {len(events):,}")
    print(f"Structure: {structure_type}")
    print(f"Data type: {file_type}")
    
    duration = events[:, 3].max() - events[:, 3].min()
    print(f"Duration: {duration:.3f} seconds")
    print(f"Event rate: {len(events)/duration:,.0f} events/second")
    
    print(f"\n✓ VALIDATION COMPLETED")
    print(f"Results saved in: predicted_events_validation.png")

if __name__ == "__main__":
    main()