#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Test the tolerance analysis functions
def test_tolerance_analysis():
    # Create synthetic data for testing
    np.random.seed(42)
    n_events = 1000
    
    # Generate synthetic real events
    real_events = np.column_stack([
        np.random.uniform(100, 200, n_events),  # x
        np.random.uniform(100, 200, n_events),  # y
        np.random.randint(0, 2, n_events),      # polarity
        np.random.uniform(5.0, 5.1, n_events),  # timestamp
        np.zeros(n_events)                      # flag (real)
    ])
    
    # Generate synthetic predicted events (slightly offset)
    pred_events = np.column_stack([
        real_events[:, 0] + np.random.normal(0, 1.5, n_events),  # x + noise
        real_events[:, 1] + np.random.normal(0, 1.5, n_events),  # y + noise
        real_events[:, 2],                                        # same polarity
        real_events[:, 3] + np.random.normal(0, 0.002, n_events), # timestamp + noise
        np.ones(n_events)                                         # flag (predicted)
    ])
    
    # Combine events
    combined = np.vstack([real_events, pred_events])
    
    # Test tolerance analysis
    from visualize_time_window import analyze_tolerance_effects, plot_tolerance_analysis
    
    print("Testing tolerance analysis...")
    results = analyze_tolerance_effects(combined, (5.0, 5.1), 
                                      [1.0, 2.0, 3.0], 
                                      [1.0, 2.0, 3.0])
    
    print(f"Generated {len(results)} tolerance combinations")
    for r in results:
        print(f"  {r['temporal_ms']}ms, {r['spatial_px']}px: {r['cancellation_rate']:.1f}%")
    
    # Create plot
    fig = plot_tolerance_analysis(results, "test_tolerance_analysis.png")
    print("Saved test tolerance analysis plot")
    
    return results

if __name__ == "__main__":
    test_tolerance_analysis()


