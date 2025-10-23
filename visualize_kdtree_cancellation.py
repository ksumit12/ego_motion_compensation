#!/usr/bin/env python3
"""
Animated Visualization of KDTree-based Ego-Motion Cancellation
Shows how spatial-temporal matching works with KDTree queries
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import cKDTree
import time
import os

# Set matplotlib backend for headless operation
import matplotlib
matplotlib.use('Agg')

class CancellationVisualizer:
    def __init__(self, width=640, height=480, spatial_tol=3.0, temporal_tol=2.0):
        self.width = width
        self.height = height
        self.spatial_tol = spatial_tol
        self.temporal_tol = temporal_tol  # ms
        
        # Event storage
        self.real_events = []  # List of (x, y, t, polarity)
        self.pred_events = []  # List of (x, y, t, polarity)
        self.matches = []      # List of (real_idx, pred_idx, distance)
        
        # Animation parameters
        self.current_time = 0.0
        self.time_step = 0.1  # ms
        self.max_time = 50.0  # ms
        
        # Motion parameters (simple rotation)
        self.center_x = width / 2
        self.center_y = height / 2
        self.omega = 0.1  # rad/ms
        
    def generate_synthetic_events(self):
        """Generate synthetic events for visualization"""
        np.random.seed(42)  # For reproducible results
        
        # Generate real events (simulating a rotating pattern)
        n_real = 200
        angles = np.linspace(0, 4*np.pi, n_real)
        radius = 100
        
        for i, angle in enumerate(angles):
            # Rotating pattern
            x = self.center_x + radius * np.cos(angle)
            y = self.center_y + radius * np.sin(angle)
            t = i * 0.5  # ms
            polarity = 1 if i % 2 == 0 else -1
            
            self.real_events.append([x, y, t, polarity])
        
        # Generate predicted events (motion-compensated)
        for i, (x, y, t, pol) in enumerate(self.real_events):
            # Simple motion model: predict where event would be if camera was stationary
            dt = 2.0  # ms prediction horizon
            predicted_t = t + dt
            
            # Add some prediction error
            noise_x = np.random.normal(0, 1.0)
            noise_y = np.random.normal(0, 1.0)
            
            pred_x = x + noise_x
            pred_y = y + noise_y
            
            self.pred_events.append([pred_x, pred_y, predicted_t, -pol])  # Opposite polarity
        
        self.real_events = np.array(self.real_events)
        self.pred_events = np.array(self.pred_events)
        
    def find_matches_at_time(self, current_time):
        """Find matches using KDTree at current time"""
        # Get events within temporal window
        time_window = self.temporal_tol
        
        # Real events that have occurred
        real_mask = self.real_events[:, 2] <= current_time
        real_xy = self.real_events[real_mask, :2]
        real_indices = np.where(real_mask)[0]
        
        # Predicted events within time window
        pred_mask = (self.pred_events[:, 2] >= current_time - time_window) & \
                   (self.pred_events[:, 2] <= current_time + time_window)
        pred_xy = self.pred_events[pred_mask, :2]
        pred_indices = np.where(pred_mask)[0]
        
        if len(real_xy) == 0 or len(pred_xy) == 0:
            return [], []
        
        # Build KDTree for predicted events
        tree = cKDTree(pred_xy)
        
        # Query for nearest neighbors within spatial tolerance
        distances, indices = tree.query(real_xy, k=1, distance_upper_bound=self.spatial_tol)
        
        # Find valid matches
        valid_matches = []
        used_pred = set()
        
        for i, (dist, pred_idx) in enumerate(zip(distances, indices)):
            if np.isfinite(dist) and pred_idx not in used_pred:
                real_idx = real_indices[i]
                pred_idx_global = pred_indices[pred_idx]
                
                # Check polarity (opposite polarity matching)
                if self.real_events[real_idx, 3] != self.pred_events[pred_idx_global, 3]:
                    valid_matches.append((real_idx, pred_idx_global, dist))
                    used_pred.add(pred_idx)
        
        return valid_matches, real_indices
    
    def create_animation(self, output_path="kdtree_cancellation_animation.mp4"):
        """Create animated visualization"""
        print("Creating KDTree cancellation animation...")
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title('KDTree-based Ego-Motion Cancellation\nRed: Real Events, Blue: Predicted Events, Green: Matches', 
                    fontsize=14, pad=20)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add ROI circle
        roi_circle = plt.Circle((self.center_x, self.center_y), 150, 
                               fill=False, color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(roi_circle)
        
        # Initialize plot elements
        real_scatter = ax.scatter([], [], c='red', s=30, alpha=0.7, label='Real Events')
        pred_scatter = ax.scatter([], [], c='blue', s=30, alpha=0.7, label='Predicted Events')
        match_lines = []
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add info text
        info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame):
            current_time = frame * self.time_step
            
            # Clear previous match lines
            for line in match_lines:
                line.remove()
            match_lines.clear()
            
            # Find matches at current time
            matches, real_indices = self.find_matches_at_time(current_time)
            
            # Get events to display
            real_mask = self.real_events[:, 2] <= current_time
            pred_mask = (self.pred_events[:, 2] >= current_time - self.temporal_tol) & \
                       (self.pred_events[:, 2] <= current_time + self.temporal_tol)
            
            real_xy = self.real_events[real_mask, :2]
            pred_xy = self.pred_events[pred_mask, :2]
            
            # Update scatter plots
            real_scatter.set_offsets(real_xy)
            pred_scatter.set_offsets(pred_xy)
            
            # Draw match lines
            for real_idx, pred_idx, dist in matches:
                real_pos = self.real_events[real_idx, :2]
                pred_pos = self.pred_events[pred_idx, :2]
                
                line = ax.plot([real_pos[0], pred_pos[0]], 
                              [real_pos[1], pred_pos[1]], 
                              'g-', alpha=0.6, linewidth=1)[0]
                match_lines.append(line)
            
            # Update info text
            total_real = len(real_xy)
            total_pred = len(pred_xy)
            total_matches = len(matches)
            cancellation_rate = (total_matches / total_real * 100) if total_real > 0 else 0
            
            info_text.set_text(f'Time: {current_time:.1f} ms\n'
                              f'Real Events: {total_real}\n'
                              f'Predicted Events: {total_pred}\n'
                              f'Matches: {total_matches}\n'
                              f'Cancellation Rate: {cancellation_rate:.1f}%\n'
                              f'Spatial Tolerance: {self.spatial_tol} px\n'
                              f'Temporal Tolerance: {self.temporal_tol} ms')
            
            return [real_scatter, pred_scatter] + match_lines + [info_text]
        
        # Create animation
        frames = int(self.max_time / self.time_step)
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                      interval=100, blit=False, repeat=True)
        
        # Save animation
        print(f"Saving animation to {output_path}...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='KDTree Visualization'), bitrate=1800)
        
        try:
            anim.save(output_path, writer=writer)
            print(f"Animation saved successfully: {output_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Trying alternative method...")
            # Fallback: save as GIF
            gif_path = output_path.replace('.mp4', '.gif')
            anim.save(gif_path, writer='pillow', fps=5)
            print(f"Animation saved as GIF: {gif_path}")
        
        plt.close(fig)
        return output_path

def create_kdtree_explanation_plot():
    """Create a static plot explaining KDTree concept"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: KDTree Structure
    ax1.set_title('KDTree Spatial Indexing', fontsize=14)
    
    # Generate sample points
    np.random.seed(42)
    points = np.random.rand(20, 2) * 100
    
    # Draw KDTree structure (simplified)
    ax1.scatter(points[:, 0], points[:, 1], c='blue', s=50, alpha=0.7)
    
    # Draw some query circles
    query_points = [(30, 30), (70, 70)]
    colors = ['red', 'green']
    
    for i, (qx, qy) in enumerate(query_points):
        ax1.scatter(qx, qy, c=colors[i], s=100, marker='x', linewidth=3)
        circle = plt.Circle((qx, qy), 15, fill=False, color=colors[i], linestyle='--', alpha=0.7)
        ax1.add_patch(circle)
        
        # Find nearest neighbors
        tree = cKDTree(points)
        distances, indices = tree.query([qx, qy], k=3, distance_upper_bound=15)
        
        for j, (dist, idx) in enumerate(zip(distances, indices)):
            if np.isfinite(dist):
                ax1.plot([qx, points[idx, 0]], [qy, points[idx, 1]], 
                        color=colors[i], alpha=0.5, linewidth=2)
    
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Temporal Matching
    ax2.set_title('Spatial-Temporal Matching Process', fontsize=14)
    
    # Timeline
    times = np.linspace(0, 10, 100)
    real_events = np.sin(times) * 20 + 50
    pred_events = np.sin(times + 0.5) * 20 + 50
    
    ax2.plot(times, real_events, 'r-', linewidth=2, label='Real Events')
    ax2.plot(times, pred_events, 'b-', linewidth=2, label='Predicted Events')
    
    # Highlight matching window
    match_time = 5.0
    ax2.axvline(match_time, color='green', linestyle='--', alpha=0.7, label='Current Time')
    ax2.axvspan(match_time - 1, match_time + 1, alpha=0.2, color='green', label='Temporal Window')
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Event Intensity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kdtree_explanation.png', dpi=150, bbox_inches='tight')
    print("Explanation plot saved: kdtree_explanation.png")
    plt.close()

def main():
    """Main function to create visualization"""
    print("=== KDTree Cancellation Visualization ===")
    
    # Create explanation plot
    create_kdtree_explanation_plot()
    
    # Create animated visualization
    visualizer = CancellationVisualizer(
        width=640, 
        height=480, 
        spatial_tol=3.0, 
        temporal_tol=2.0
    )
    
    # Generate synthetic events
    print("Generating synthetic events...")
    visualizer.generate_synthetic_events()
    print(f"Generated {len(visualizer.real_events)} real events and {len(visualizer.pred_events)} predicted events")
    
    # Create animation
    output_path = visualizer.create_animation("kdtree_cancellation_animation.mp4")
    
    print(f"\nVisualization complete!")
    print(f"Files created:")
    print(f"- kdtree_explanation.png (static explanation)")
    print(f"- {output_path} (animated visualization)")
    print(f"\nThe animation shows:")
    print(f"- Red dots: Real events appearing over time")
    print(f"- Blue dots: Predicted events (motion-compensated)")
    print(f"- Green lines: Successful spatial-temporal matches")
    print(f"- Gray circle: Region of Interest (ROI)")
    print(f"- Info panel: Real-time statistics")

if __name__ == "__main__":
    main()

