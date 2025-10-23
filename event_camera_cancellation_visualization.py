#!/usr/bin/env python3
"""
Event Camera Ego-Motion Cancellation Visualization

This script creates a clear, professional visualization that shows how
KDTree is used for efficient spatial-temporal matching in event camera
ego-motion cancellation. Perfect for presentations and educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import time

# Animation parameters - Clear and professional
FPS = 1  # Slow for clarity
DURATION_SECONDS = 20  # Good length for presentations
TOTAL_FRAMES = FPS * DURATION_SECONDS

# Visual parameters
FIGURE_SIZE = (16, 10)
EVENT_SIZE = 150
REAL_EVENT_COLOR = 'steelblue'
PREDICTED_EVENT_COLOR = 'lightcoral'
MATCHED_EVENT_COLOR = 'forestgreen'
CANCELLATION_COLOR = 'darkorange'

class EventCameraVisualization:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        self.fig.suptitle('Event Camera Ego-Motion Cancellation: Spatial-Temporal Matching', 
                         fontsize=20, fontweight='bold', color='navy')
        
        # Setup subplots
        self.setup_subplots()
        
        # Generate event camera data
        self.generate_event_data()
        
        # Initialize animation variables
        self.current_frame = 0
        self.matched_events = []
        self.cancelled_events = []
        
    def setup_subplots(self):
        """Setup the four subplot areas with event camera terminology"""
        # Top-left: KDTree building
        self.ax_kdtree = self.axes[0, 0]
        self.ax_kdtree.set_title('Step 1: Building Spatial Index (KDTree)', fontsize=16, fontweight='bold', color='darkred')
        self.ax_kdtree.set_xlabel('Pixel X Coordinate')
        self.ax_kdtree.set_ylabel('Pixel Y Coordinate')
        self.ax_kdtree.grid(True, alpha=0.3)
        
        # Top-right: Spatial matching
        self.ax_spatial = self.axes[0, 1]
        self.ax_spatial.set_title('Step 2: Spatial Matching Process', fontsize=16, fontweight='bold', color='darkgreen')
        self.ax_spatial.set_xlabel('Pixel X Coordinate')
        self.ax_spatial.set_ylabel('Pixel Y Coordinate')
        self.ax_spatial.grid(True, alpha=0.3)
        
        # Bottom-left: Algorithm explanation
        self.ax_explanation = self.axes[1, 0]
        self.ax_explanation.set_title('Ego-Motion Cancellation Algorithm', fontsize=16, fontweight='bold', color='darkblue')
        self.ax_explanation.axis('off')
        
        # Bottom-right: Cancellation results
        self.ax_results = self.axes[1, 1]
        self.ax_results.set_title('Step 3: Ego-Motion Cancellation Results', fontsize=16, fontweight='bold', color='purple')
        self.ax_results.set_xlabel('Pixel X Coordinate')
        self.ax_results.set_ylabel('Pixel Y Coordinate')
        self.ax_results.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def generate_event_data(self):
        """Generate event camera data for demonstration"""
        np.random.seed(42)  # For reproducible results
        
        # Generate "real" events (blue dots) - actual camera events
        self.real_events = np.array([
            [100, 150], [200, 200], [300, 100], [150, 300],
            [250, 250], [350, 200], [100, 350], [400, 100]
        ])
        self.real_times = np.array([10, 20, 30, 40, 50, 60, 70, 80])  # milliseconds
        self.real_polarities = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # +1 or -1
        
        # Generate "predicted" events (red dots) - ego-motion predicted events
        self.predicted_events = np.array([
            [120, 160], [180, 190], [320, 110], [140, 310],
            [270, 260], [330, 190], [90, 340], [410, 90],
            [150, 180], [220, 220], [280, 120], [160, 280]
        ])
        self.predicted_times = np.array([12, 22, 32, 42, 52, 62, 72, 82, 15, 25, 35, 45])  # milliseconds
        self.predicted_polarities = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # +1 or -1
        
        # Build KDTree for predicted events (spatial index)
        self.predicted_tree = cKDTree(self.predicted_events)
        
        # Spatial tolerance (pixels)
        self.spatial_tolerance = 50
        # Temporal tolerance (milliseconds)
        self.temporal_tolerance = 4
        # Time offset (dt) in milliseconds
        self.dt_ms = 2
        
    def draw_kdtree_building(self, ax, progress):
        """Draw the KDTree building process"""
        # Calculate how many predicted events to show as indexed
        num_indexed = int(progress * len(self.predicted_events))
        
        # Show all predicted events
        ax.scatter(self.predicted_events[:, 0], self.predicted_events[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Predicted Events', edgecolors='black', linewidth=1)
        
        # Show indexed events (events that are part of the spatial index)
        if num_indexed > 0:
            indexed_events = self.predicted_events[:num_indexed]
            ax.scatter(indexed_events[:, 0], indexed_events[:, 1], 
                      c=CANCELLATION_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                      label=f'Indexed Events ({num_indexed})',
                      edgecolors='black', linewidth=2)
        
        # Draw spatial index connections
        if num_indexed > 1:
            for i in range(1, num_indexed):
                # Draw lines showing spatial organization
                ax.plot([self.predicted_events[i-1, 0], self.predicted_events[i, 0]], 
                       [self.predicted_events[i-1, 1], self.predicted_events[i, 1]], 
                       'r-', linewidth=2, alpha=0.7)
        
        ax.legend(fontsize=12)
        ax.set_title(f'Step 1: Building Spatial Index ({num_indexed}/{len(self.predicted_events)})', 
                    fontsize=16, fontweight='bold', color='darkred')
    
    def draw_spatial_matching(self, ax, progress):
        """Draw the spatial matching process"""
        # Show all events
        ax.scatter(self.predicted_events[:, 0], self.predicted_events[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Predicted Events', edgecolors='black', linewidth=1)
        ax.scatter(self.real_events[:, 0], self.real_events[:, 1], 
                  c=REAL_EVENT_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Real Events', edgecolors='black', linewidth=2)
        
        # Calculate how many real events to process
        num_processed = int(progress * len(self.real_events))
        
        # Show spatial matching
        for i in range(min(num_processed, len(self.real_events))):
            real_event = self.real_events[i]
            
            # Find spatially nearby predicted events using KDTree
            nearby_indices = self.predicted_tree.query_ball_point(real_event, self.spatial_tolerance)
            
            if len(nearby_indices) > 0:
                # Draw spatial connection lines
                for idx in nearby_indices:
                    predicted_event = self.predicted_events[idx]
                    ax.plot([real_event[0], predicted_event[0]], 
                           [real_event[1], predicted_event[1]], 
                           'g-', linewidth=3, alpha=0.8)
                
                # Highlight spatially matched events
                ax.scatter(real_event[0], real_event[1], 
                          c='green', s=EVENT_SIZE*1.5, alpha=0.9,
                          edgecolors='black', linewidth=3)
                
                matched_events = self.predicted_events[nearby_indices]
                ax.scatter(matched_events[:, 0], matched_events[:, 1], 
                          c='green', s=EVENT_SIZE*1.2, alpha=0.9,
                          edgecolors='black', linewidth=2)
        
        ax.legend(fontsize=12)
        ax.set_title(f'Step 2: Spatial Matching ({num_processed}/{len(self.real_events)})', 
                    fontsize=16, fontweight='bold', color='darkgreen')
    
    def draw_algorithm_explanation(self, ax, progress):
        """Draw algorithm explanation for event camera ego-motion cancellation"""
        if progress < 0.3:
            explanation = """Event Camera Ego-Motion Cancellation

Problem: Remove camera motion from event stream
• Real events: Actual brightness changes captured
• Predicted events: Expected events from camera motion
• Goal: Cancel out ego-motion to see only scene motion

Traditional Approach:
• Check every predicted event for each real event
• Time complexity: O(N × M) - computationally expensive!
• Not suitable for real-time event camera processing

Smart Spatial Index Approach:
• Organize predicted events in KDTree structure
• Fast spatial queries: O(log M) per real event
• Time complexity: O(N × log M) - much faster!
• Enables real-time ego-motion cancellation"""
            
        elif progress < 0.6:
            explanation = """Spatial-Temporal Matching Process

Step 1: Build Spatial Index
• Organize predicted events spatially using KDTree
• Create hierarchical spatial structure
• Split space recursively (X-axis, then Y-axis)
• Each region contains spatially nearby events

Step 2: Spatial Matching
• For each real event:
  - Query KDTree for spatially nearby predicted events
  - Apply spatial tolerance threshold
  - Get candidate matches within radius

Step 3: Temporal Filtering
• Apply temporal gate: |t_pred - (t_real + dt)| ≤ ε_t
• Filter candidates by time difference
• Ensure causality constraints

Step 4: Polarity Matching
• Check event polarities match criteria
• Opposite polarities for cancellation
• Final match selection"""
            
        elif progress < 0.9:
            explanation = """Event Camera Applications

Robotics and Autonomous Vehicles:
• Remove camera shake from event streams
• Stabilize vision for navigation
• Real-time motion compensation

Computer Vision:
• Motion segmentation
• Object tracking in moving cameras
• Background subtraction

Augmented Reality:
• Stabilize event-based displays
• Motion compensation for AR overlays
• Real-time scene understanding

Scientific Imaging:
• High-speed motion analysis
• Microscopy with moving samples
• Astronomical observations

Why KDTree is Essential:
• Real-time processing requirements
• High event rates (millions per second)
• Low latency constraints
• Power efficiency in mobile devices"""
            
        else:
            explanation = """Algorithm Performance Benefits

Computational Efficiency:
• Traditional: O(N²) - quadratic complexity
• KDTree: O(N log N) - near linear complexity
• Speedup: 100x-1000x for large event streams

Real-Time Capability:
• Processes millions of events per second
• Low latency: <1ms processing time
• Suitable for real-time applications

Memory Efficiency:
• Linear memory usage: O(N)
• Cache-friendly data structure
• Optimized for modern processors

Accuracy:
• Exact spatial-temporal matching
• Configurable tolerance parameters
• Robust to noise and outliers

This makes ego-motion cancellation
practical for real-world event camera systems!"""
        
        ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
                fontsize=13, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def draw_cancellation_results(self, ax, progress):
        """Draw the ego-motion cancellation results"""
        # Show all events
        ax.scatter(self.predicted_events[:, 0], self.predicted_events[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Predicted Events', edgecolors='black', linewidth=1)
        ax.scatter(self.real_events[:, 0], self.real_events[:, 1], 
                  c=REAL_EVENT_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Real Events', edgecolors='black', linewidth=2)
        
        # Calculate how many cancellations to show
        num_cancellations = int(progress * len(self.real_events))
        total_cancelled = 0
        
        # Show cancelled events
        for i in range(min(num_cancellations, len(self.real_events))):
            real_event = self.real_events[i]
            real_time = self.real_times[i]
            target_time = real_time + self.dt_ms
            
            # Find spatially nearby predicted events
            nearby_indices = self.predicted_tree.query_ball_point(real_event, self.spatial_tolerance)
            
            if len(nearby_indices) > 0:
                # Apply temporal gate
                candidate_times = self.predicted_times[nearby_indices]
                temporal_mask = np.abs(candidate_times - target_time) <= self.temporal_tolerance
                
                if np.any(temporal_mask):
                    # Take the best match
                    temporal_candidates = [nearby_indices[j] for j in range(len(nearby_indices)) if temporal_mask[j]]
                    best_match_idx = temporal_candidates[0]
                    predicted_event = self.predicted_events[best_match_idx]
                    
                    # Draw cancellation line
                    ax.plot([real_event[0], predicted_event[0]], 
                           [real_event[1], predicted_event[1]], 
                           'g-', linewidth=4, alpha=0.9)
                    
                    # Highlight cancelled events
                    ax.scatter(real_event[0], real_event[1], 
                              c='green', s=EVENT_SIZE*1.5, alpha=0.9,
                              edgecolors='black', linewidth=3)
                    ax.scatter(predicted_event[0], predicted_event[1], 
                              c='green', s=EVENT_SIZE*1.2, alpha=0.9,
                              edgecolors='black', linewidth=2)
                    
                    total_cancelled += 1
        
        # Add cancellation count
        ax.text(0.02, 0.98, f'Events Cancelled: {total_cancelled}', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        ax.legend(fontsize=12)
        ax.set_title(f'Step 3: Ego-Motion Cancellation ({total_cancelled} cancelled)', 
                    fontsize=16, fontweight='bold', color='purple')
    
    def animate_frame(self, frame_num):
        """Animate a single frame"""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Re-setup subplot titles and labels
        self.setup_subplots()
        
        # Calculate animation progress
        progress = frame_num / TOTAL_FRAMES
        
        # Draw all four views
        self.draw_kdtree_building(self.ax_kdtree, progress)
        self.draw_spatial_matching(self.ax_spatial, progress)
        self.draw_algorithm_explanation(self.ax_explanation, progress)
        self.draw_cancellation_results(self.ax_results, progress)
        
        # Add progress indicator
        self.fig.text(0.02, 0.02, f'Progress: {progress*100:.1f}%', 
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        return []
    
    def create_animation(self):
        """Create and save the event camera animation"""
        print("Creating event camera ego-motion cancellation visualization...")
        print(f"Duration: {DURATION_SECONDS} seconds")
        print(f"Frames: {TOTAL_FRAMES}")
        print(f"FPS: {FPS}")
        print("Professional visualization for event camera research!")
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, 
            self.animate_frame, 
            frames=TOTAL_FRAMES,
            interval=1000/FPS,  # milliseconds per frame
            blit=False,
            repeat=True
        )
        
        # Save animation
        output_file = "event_camera_cancellation_visualization.mp4"
        print(f"Saving event camera animation to: {output_file}")
        
        try:
            anim.save(output_file, writer='ffmpeg', fps=FPS, bitrate=1800)
            print(f"✓ Event camera animation saved successfully: {output_file}")
        except Exception as e:
            print(f"✗ Error saving animation: {e}")
            print("Trying alternative format...")
            try:
                output_file_gif = "event_camera_cancellation_visualization.gif"
                anim.save(output_file_gif, writer='pillow', fps=FPS)
                print(f"✓ Event camera animation saved as GIF: {output_file_gif}")
            except Exception as e2:
                print(f"✗ Error saving GIF: {e2}")
                print("Showing animation in window instead...")
                plt.show()
        
        return anim

def main():
    """Main function to create the event camera animation"""
    print("=" * 80)
    print("Event Camera Ego-Motion Cancellation Visualization")
    print("=" * 80)
    print("This visualization demonstrates:")
    print("• How KDTree enables efficient spatial indexing for event cameras")
    print("• Spatial-temporal matching process for ego-motion cancellation")
    print("• Real-time processing requirements for event camera systems")
    print("• Applications in robotics, AR, and computer vision")
    print("=" * 80)
    print("Perfect for event camera research presentations!")
    print("=" * 80)
    
    # Create event camera animation
    visualizer = EventCameraVisualization()
    animation_obj = visualizer.create_animation()
    
    print("\nEvent camera visualization complete!")
    print("The animation clearly demonstrates how KDTree enables")
    print("efficient ego-motion cancellation in event camera systems.")

if __name__ == "__main__":
    main()












