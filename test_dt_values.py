import numpy as np
from motion_model import load_event_data, load_center_from_tracker, apply_rotation

# Test different DT values
DT_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
EVENTS_PATH = "sample.csv"
CENTER_PATH = "spin-dot_track.csv"

def analyze_dt_values():
    """Analyze how different DT values affect prediction"""
    
    # Load sample of events
    events = load_event_data(EVENTS_PATH)
    events = events[np.argsort(events[:, 3])][:1000]  # First 1000 events
    center = load_center_from_tracker(CENTER_PATH)
    
    print("DT Value Analysis for Event Camera Ego-Motion Cancellation")
    print("=" * 60)
    print(f"{'DT (s)':<10} {'DT (ms)':<10} {'Events Ahead':<15} {'Typical Use Case'}")
    print("-" * 60)
    
    for dt in DT_VALUES:
        # Calculate how many events ahead this predicts
        if len(events) > 1:
            avg_event_interval = np.mean(np.diff(events[:, 3]))
            events_ahead = int(dt / avg_event_interval)
        else:
            events_ahead = 0
            
        # Determine use case
        if dt <= 0.005:
            use_case = "Real-time cancellation"
        elif dt <= 0.01:
            use_case = "Fast cancellation"
        elif dt <= 0.05:
            use_case = "Medium prediction"
        else:
            use_case = "Long prediction (not recommended)"
            
        print(f"{dt:<10.3f} {dt*1000:<10.1f} {events_ahead:<15} {use_case}")
    
    print("\nRecommendations:")
    print("- For real-time ego-motion cancellation: DT = 0.001-0.005s (1-5ms)")
    print("- For fast processing: DT = 0.01s (10ms)")
    print("- Avoid DT > 0.05s for event camera applications")

if __name__ == "__main__":
    analyze_dt_values() 