#!/usr/bin/env python3
"""
Test script to verify that bilinear interpolation doesn't change cancellation metrics.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_cancellation_test(use_bilinear):
    """Run cancellation with specified interpolation mode and return metrics"""
    import visualize_time_window as vw
    vw.USE_BILINEAR_INTERP = use_bilinear
    
    combined = vw.load_combined(vw.COMBINED_PATH)
    residual_real, residual_pred = vw.run_cancellation(combined, vw.BIN_MS, vw.R_PIX)
    
    total_real = int(np.sum(combined[:, 4] == 0.0))
    total_pred = int(np.sum(combined[:, 4] == 1.0))
    real_cancelled = total_real - len(residual_real)
    pred_cancelled = total_pred - len(residual_pred)
    real_rate = (real_cancelled / total_real * 100) if total_real > 0 else 0
    pred_rate = (pred_cancelled / total_pred * 100) if total_pred > 0 else 0
    
    return {
        'real_cancelled': real_cancelled,
        'pred_cancelled': pred_cancelled,
        'real_rate': real_rate,
        'pred_rate': pred_rate,
        'residual_real_shape': residual_real.shape,
        'residual_pred_shape': residual_pred.shape
    }

def test_metrics_unchanged():
    """Test that cancellation metrics are identical between bilinear and nearest neighbor modes"""
    print("Testing with bilinear interpolation...")
    try:
        metrics_bilinear = run_cancellation_test(True)
        print(f"Bilinear: Real cancelled={metrics_bilinear['real_cancelled']}, Pred cancelled={metrics_bilinear['pred_cancelled']}")
        print(f"Bilinear: Real rate={metrics_bilinear['real_rate']:.2f}%, Pred rate={metrics_bilinear['pred_rate']:.2f}%")
    except Exception as e:
        print(f"Bilinear test failed: {e}")
        return False
    
    print("\nTesting with nearest neighbor...")
    try:
        metrics_nearest = run_cancellation_test(False)
        print(f"Nearest: Real cancelled={metrics_nearest['real_cancelled']}, Pred cancelled={metrics_nearest['pred_cancelled']}")
        print(f"Nearest: Real rate={metrics_nearest['real_rate']:.2f}%, Pred rate={metrics_nearest['pred_rate']:.2f}%")
    except Exception as e:
        print(f"Nearest test failed: {e}")
        return False
    
    # Compare metrics
    print("\n=== METRICS COMPARISON ===")
    
    metrics_match = (
        metrics_bilinear['real_cancelled'] == metrics_nearest['real_cancelled'] and
        metrics_bilinear['pred_cancelled'] == metrics_nearest['pred_cancelled'] and
        abs(metrics_bilinear['real_rate'] - metrics_nearest['real_rate']) < 0.01 and
        abs(metrics_bilinear['pred_rate'] - metrics_nearest['pred_rate']) < 0.01 and
        metrics_bilinear['residual_real_shape'] == metrics_nearest['residual_real_shape'] and
        metrics_bilinear['residual_pred_shape'] == metrics_nearest['residual_pred_shape']
    )
    
    if metrics_match:
        print("SUCCESS: All cancellation metrics are identical between bilinear and nearest neighbor modes")
        return True
    else:
        print("FAILURE: Metrics differ between bilinear and nearest neighbor modes")
        print(f"Real cancelled: bilinear={metrics_bilinear['real_cancelled']} vs nearest={metrics_nearest['real_cancelled']}")
        print(f"Pred cancelled: bilinear={metrics_bilinear['pred_cancelled']} vs nearest={metrics_nearest['pred_cancelled']}")
        print(f"Real rate: bilinear={metrics_bilinear['real_rate']:.2f}% vs nearest={metrics_nearest['real_rate']:.2f}%")
        print(f"Pred rate: bilinear={metrics_bilinear['pred_rate']:.2f}% vs nearest={metrics_nearest['pred_rate']:.2f}%")
        print(f"Residual real shape: bilinear={metrics_bilinear['residual_real_shape']} vs nearest={metrics_nearest['residual_real_shape']}")
        print(f"Residual pred shape: bilinear={metrics_bilinear['residual_pred_shape']} vs nearest={metrics_nearest['residual_pred_shape']}")
        return False

def test_image_difference():
    """Test that images are visually different but have similar global sums"""
    print("\n=== IMAGE DIFFERENCE TEST ===")
    
    import visualize_time_window as vw
    
    try:
        combined = vw.load_combined(vw.COMBINED_PATH)
        window = vw.WINDOWS[0]
        
        # Get events in window
        start_time, end_time = window
        time_mask = (combined[:, 3] >= start_time) & (combined[:, 3] < end_time)
        window_events = combined[time_mask]
        all_events = window_events[:, :3]
        
        # Test both methods
        vw.USE_BILINEAR_INTERP = True
        img_bilinear = vw.create_per_pixel_count_image(vw.IMG_W, vw.IMG_H, all_events)
        sum_bilinear = float(np.sum(img_bilinear))
        
        vw.USE_BILINEAR_INTERP = False
        img_nearest = vw.create_per_pixel_count_image(vw.IMG_W, vw.IMG_H, all_events)
        sum_nearest = float(np.sum(img_nearest))
        
        # Analyze differences
        diff = img_bilinear - img_nearest
        max_diff = float(np.max(np.abs(diff)))
        nonzero_diff_pixels = int(np.sum(np.abs(diff) > 1e-6))
        sum_ratio = sum_bilinear / sum_nearest if sum_nearest != 0 else 1.0
        
        print(f"Bilinear sum: {sum_bilinear:.6f}")
        print(f"Nearest sum: {sum_nearest:.6f}")
        print(f"Sum ratio: {sum_ratio:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Pixels with difference > 1e-6: {nonzero_diff_pixels}")
        
        # Verify: similar sums but visual differences
        success = abs(sum_ratio - 1.0) < 0.01 and nonzero_diff_pixels > 0
        if success:
            print("SUCCESS: Images are visually different but have similar global sums")
        else:
            print("FAILURE: Image test failed")
        return success
            
    except Exception as e:
        print(f"Image test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing bilinear interpolation implementation...")
    
    metrics_ok = test_metrics_unchanged()
    images_ok = test_image_difference()
    
    if metrics_ok and images_ok:
        print("\nALL TESTS PASSED: Bilinear interpolation works correctly")
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
