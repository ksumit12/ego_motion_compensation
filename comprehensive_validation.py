#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, kstest, shapiro
import time

class Validator:
    def __init__(self, events_file="combined_events_with_predictions.npy"):
        self.events_file = events_file
        self.events = None
        self.real_events = None
        self.pred_events = None
        self.actual_center = None
        
        try:
            from motion_model import load_center_from_tracker
            center_data = load_center_from_tracker("spin-dot_track.csv")
            self.actual_center = (float(center_data[0]), float(center_data[1]))
            print(f"Loaded center: ({self.actual_center[0]:.2f}, {self.actual_center[1]:.2f})")
        except:
            self.actual_center = (640, 360)
        
        self.EXPECTED_DT = 0.005
        self.EXPECTED_OMEGA = 2.5 * np.pi
    
    def load_and_verify_structure(self):
        print("Loading data...")
        self.events = np.load(self.events_file)
        print(f"Loaded {self.events.shape}")
        
        n_total = len(self.events)
        n_half = n_total // 2
        first_half = self.events[:n_half]
        second_half = self.events[n_half:]
        
        time_offset = second_half[:,3].mean() - first_half[:,3].mean()
        print(f"Time offset: {time_offset:.6f}s (expected: {self.EXPECTED_DT:.6f}s)")
        
        if abs(time_offset - self.EXPECTED_DT) < 0.001:
            self.real_events = first_half
            self.pred_events = second_half
            print("Structure OK")
            return True
        else:
            print("Structure failed")
            return False
    
    def verify_motion_physics(self):
        print("Checking physics...")
        
        n_sample = min(5000, len(self.real_events))
        sample_real = self.real_events[:n_sample]
        sample_pred = self.pred_events[:n_sample]
        
        center = self.actual_center
        
        real_distances = np.sqrt((sample_real[:, 0] - center[0])**2 + (sample_real[:, 1] - center[1])**2)
        pred_distances = np.sqrt((sample_pred[:, 0] - center[0])**2 + (sample_pred[:, 1] - center[1])**2)
        
        distance_diff = pred_distances - real_distances
        distance_error = np.mean(np.abs(distance_diff))
        print(f"Distance error: {distance_error:.3f} pixels")
        
        real_angles = np.arctan2(sample_real[:, 1] - center[1], sample_real[:, 0] - center[0])
        pred_angles = np.arctan2(sample_pred[:, 1] - center[1], sample_pred[:, 0] - center[0])
        
        angle_diff = pred_angles - real_angles
        angle_diff = np.mod(angle_diff + np.pi, 2*np.pi) - np.pi
        
        expected_angle = self.EXPECTED_OMEGA * self.EXPECTED_DT
        angle_error = np.mean(np.abs(angle_diff - expected_angle))
        print(f"Angle error: {angle_error:.6f} rad")
        
        return distance_error < 0.1 and angle_error < 0.01
    
    def cross_validate_predictions(self):
        print("Cross-validating...")
        
        n_sample = min(1000, len(self.real_events))
        sample_real = self.real_events[:n_sample]
        sample_pred = self.pred_events[:n_sample]
        
        center = self.actual_center
        angle = self.EXPECTED_OMEGA * self.EXPECTED_DT
        
        errors = []
        for real_event, pred_event in zip(sample_real, sample_pred):
            x_real, y_real = real_event[0], real_event[1]
            x_pred, y_pred = pred_event[0], pred_event[1]
            
            x_c = x_real - center[0]
            y_c = y_real - center[1]
            
            x_rot = np.cos(angle) * x_c - np.sin(angle) * y_c + center[0]
            y_rot = np.sin(angle) * x_c + np.cos(angle) * y_c + center[1]
            
            error = np.sqrt((x_pred - x_rot)**2 + (y_pred - y_rot)**2)
            errors.append(error)
        
        mean_error = np.mean(errors)
        print(f"Mean error: {mean_error:.4f} pixels")
        
        return mean_error < 1.0
    
    def test_edge_cases(self):
        print("Testing edge cases...")
        
        width, height = 1280, 720
        real_x, real_y = self.real_events[:, 0], self.real_events[:, 1]
        pred_x, pred_y = self.pred_events[:, 0], self.pred_events[:, 1]
        
        in_bounds = (
            (pred_x >= 0) & (pred_x < width) &
            (pred_y >= 0) & (pred_y < height)
        )
        
        bounds_ok = np.mean(in_bounds)
        print(f"Events in bounds: {bounds_ok*100:.1f}%")
        
        real_times = self.real_events[:, 3]
        pred_times = self.pred_events[:, 3]
        
        time_overlap = np.any(pred_times < real_times)
        print(f"Time overlap: {'Yes' if time_overlap else 'No'}")
    
    def statistical_analysis(self):
        print("Statistical analysis...")
        
        real_x, real_y = self.real_events[:, 0], self.real_events[:, 1]
        pred_x, pred_y = self.pred_events[:, 0], self.pred_events[:, 1]
        
        n_sample = min(10000, len(self.real_events))
        sample_real = self.real_events[:n_sample]
        sample_pred = self.pred_events[:n_sample]
        
        corr_x, _ = pearsonr(sample_real[:, 0], sample_pred[:, 0])
        corr_y, _ = pearsonr(sample_real[:, 1], sample_pred[:, 1])
        
        print(f"X correlation: {corr_x:.3f}")
        print(f"Y correlation: {corr_y:.3f}")
        
        return corr_x > 0.95 and corr_y > 0.95
    
    def generate_report(self):
        print("="*50)
        print("VALIDATION REPORT")
        print("="*50)
        
        print("Physics: OK" if self.verify_motion_physics() else "Physics: FAILED")
        print("Cross-validation: OK" if self.cross_validate_predictions() else "Cross-validation: FAILED")
        print("Statistics: OK" if self.statistical_analysis() else "Statistics: FAILED")
        
        self.test_edge_cases()
    
    def run_validation(self):
        print("Starting validation...")
        
        if not self.load_and_verify_structure():
            print("Failed structure verification")
            return False
        
        self.generate_report()
        return True

def main():
    validator = Validator()
    success = validator.run_validation()
    
    if success:
        print("Validation completed")
    else:
        print("Validation failed")

if __name__ == "__main__":
    main() 