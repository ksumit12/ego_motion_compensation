import numpy as np

# Load the saved numpy array
EVENT_FILE = "combined_events_with_predictions.npy"

print("Loading numpy array...")
try:
    data = np.load(EVENT_FILE)
    print(f"✓ Loaded array shape: {data.shape}")
    print(f"✓ Data type: {data.dtype}")
    print(f"✓ Total events: {len(data):,}")
    
    # Print first 30 entries
    print("\n" + "="*80)
    print("FIRST 30 ENTRIES:")
        print("="*80)
    print("Format: [x, y, p, t, event_type]")
    print("event_type: 0.0 = real, 1.0 = predicted")
        print("="*80)
        
    for i in range(min(10000, len(data))):
        x, y, p, t, event_type = data[i]
        event_label = "REAL" if event_type == 0.0 else "PRED"
        print(f"{i+1:2d}: [{x:7.2f}, {y:7.2f}, {p:1.0f}, {t:8.6f}, {event_type:1.0f}] {event_label}")
    
    # Summary of event types
    if len(data) > 0:
        real_count = np.sum(data[:, 4] == 0.0)
        pred_count = np.sum(data[:, 4] == 1.0)
        print(f"\nSummary:")
        print(f"  - Real events: {real_count:,}")
        print(f"  - Predicted events: {pred_count:,}")
        print(f"  - Time range: {data[0, 3]:.6f}s to {data[-1, 3]:.6f}s")
        
except FileNotFoundError:
    print(f"❌ File not found: {EVENT_FILE}")
    print("Make sure to run main_2motion_test_2_fixed.py first to generate the combined events file.")
except Exception as e:
    print(f"❌ Error loading file: {e}")
