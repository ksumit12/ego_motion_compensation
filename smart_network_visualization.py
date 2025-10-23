#!/usr/bin/env python3
"""
Smart Network Visualization for Presentations

This script creates a clear, professional visualization that shows how
KDTree works as a smart network for efficient spatial searching.
Perfect for presentations and educational purposes.
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
SEARCH_COLOR = 'lightcoral'
TARGET_COLOR = 'steelblue'
NETWORK_COLOR = 'forestgreen'
CONNECTION_COLOR = 'darkorange'

class SmartNetworkVisualization:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        self.fig.suptitle('Smart Network Algorithm: Efficient Spatial Search', 
                         fontsize=20, fontweight='bold', color='navy')
        
        # Setup subplots
        self.setup_subplots()
        
        # Generate simple data
        self.generate_simple_data()
        
        # Initialize animation variables
        self.current_frame = 0
        self.network_points = []
        self.connections = []
        
    def setup_subplots(self):
        """Setup the four subplot areas with professional titles"""
        # Top-left: Network building
        self.ax_network = self.axes[0, 0]
        self.ax_network.set_title('Step 1: Building Smart Network', fontsize=16, fontweight='bold', color='darkred')
        self.ax_network.set_xlabel('X Coordinate')
        self.ax_network.set_ylabel('Y Coordinate')
        self.ax_network.grid(True, alpha=0.3)
        
        # Top-right: Connection process
        self.ax_connections = self.axes[0, 1]
        self.ax_connections.set_title('Step 2: Finding Connections', fontsize=16, fontweight='bold', color='darkgreen')
        self.ax_connections.set_xlabel('X Coordinate')
        self.ax_connections.set_ylabel('Y Coordinate')
        self.ax_connections.grid(True, alpha=0.3)
        
        # Bottom-left: Algorithm explanation
        self.ax_explanation = self.axes[1, 0]
        self.ax_explanation.set_title('Algorithm Overview', fontsize=16, fontweight='bold', color='darkblue')
        self.ax_explanation.axis('off')
        
        # Bottom-right: Results
        self.ax_results = self.axes[1, 1]
        self.ax_results.set_title('Step 3: Final Matches', fontsize=16, fontweight='bold', color='purple')
        self.ax_results.set_xlabel('X Coordinate')
        self.ax_results.set_ylabel('Y Coordinate')
        self.ax_results.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def generate_simple_data(self):
        """Generate clear data points for demonstration"""
        np.random.seed(42)  # For reproducible results
        
        # Generate "target" points (blue dots) - what we're looking for
        self.target_points = np.array([
            [100, 150], [200, 200], [300, 100], [150, 300],
            [250, 250], [350, 200], [100, 350], [400, 100]
        ])
        
        # Generate "search" points (red dots) - what we're searching with
        self.search_points = np.array([
            [120, 160], [180, 190], [320, 110], [140, 310],
            [270, 260], [330, 190], [90, 340], [410, 90],
            [150, 180], [220, 220], [280, 120], [160, 280]
        ])
        
        # Build the smart network (KDTree)
        self.smart_network = cKDTree(self.search_points)
        
        # Search radius (how close things need to be to connect)
        self.connection_distance = 50
        
    def draw_network_building(self, ax, progress):
        """Draw the network building process"""
        # Calculate how many network points to show
        num_network_points = int(progress * len(self.search_points))
        
        # Show all search points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        
        # Show network points (points that are part of the organized structure)
        if num_network_points > 0:
            network_points = self.search_points[:num_network_points]
            ax.scatter(network_points[:, 0], network_points[:, 1], 
                      c=CONNECTION_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                      label=f'Network Points ({num_network_points})',
                      edgecolors='black', linewidth=2)
        
        # Draw network connections
        if num_network_points > 1:
            for i in range(1, num_network_points):
                # Draw lines connecting network points
                ax.plot([self.search_points[i-1, 0], self.search_points[i, 0]], 
                       [self.search_points[i-1, 1], self.search_points[i, 1]], 
                       'r-', linewidth=2, alpha=0.7)
        
        ax.legend(fontsize=12)
        ax.set_title(f'Step 1: Building Smart Network ({num_network_points}/{len(self.search_points)})', 
                    fontsize=16, fontweight='bold', color='darkred')
    
    def draw_connection_process(self, ax, progress):
        """Draw how connections are found"""
        # Show all points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        ax.scatter(self.target_points[:, 0], self.target_points[:, 1], 
                  c=TARGET_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Target Points', edgecolors='black', linewidth=2)
        
        # Calculate how many connections to show
        num_connections = int(progress * len(self.target_points))
        
        # Show connections
        for i in range(min(num_connections, len(self.target_points))):
            target_point = self.target_points[i]
            
            # Find nearby search points using the smart network
            nearby_indices = self.smart_network.query_ball_point(target_point, self.connection_distance)
            
            if len(nearby_indices) > 0:
                # Draw connection lines
                for idx in nearby_indices:
                    search_point = self.search_points[idx]
                    ax.plot([target_point[0], search_point[0]], 
                           [target_point[1], search_point[1]], 
                           'g-', linewidth=3, alpha=0.8)
                
                # Highlight connected points
                ax.scatter(target_point[0], target_point[1], 
                          c='green', s=EVENT_SIZE*1.5, alpha=0.9,
                          edgecolors='black', linewidth=3)
                
                connected_points = self.search_points[nearby_indices]
                ax.scatter(connected_points[:, 0], connected_points[:, 1], 
                          c='green', s=EVENT_SIZE*1.2, alpha=0.9,
                          edgecolors='black', linewidth=2)
        
        ax.legend(fontsize=12)
        ax.set_title(f'Step 2: Finding Connections ({num_connections}/{len(self.target_points)})', 
                    fontsize=16, fontweight='bold', color='darkgreen')
    
    def draw_algorithm_explanation(self, ax, progress):
        """Draw algorithm explanation suitable for presentations"""
        if progress < 0.3:
            explanation = """Smart Network Algorithm Overview

Problem: Find nearby points efficiently
• We have target points (what we're looking for)
• We have search points (potential matches)
• Need to find closest matches quickly

Traditional Approach:
• Check every search point for each target
• Time complexity: O(N × M) - very slow!
• Like checking every house in a city

Smart Network Approach:
• Organize search points in a tree structure
• Jump directly to nearby regions
• Time complexity: O(N × log M) - much faster!
• Like having a GPS for point searching"""
            
        elif progress < 0.6:
            explanation = """How the Smart Network Works

Step 1: Build Network Structure
• Organize search points spatially
• Create hierarchical tree structure
• Split space recursively (X-axis, then Y-axis)
• Each region contains nearby points

Step 2: Query Process
• For each target point:
  - Navigate tree to find relevant region
  - Check only points in that region
  - Apply distance threshold
  - Return nearby points

Benefits:
• Fast spatial queries: O(log N)
• Memory efficient: O(N)
• Scales well with large datasets
• Used in GPS, databases, games"""
            
        elif progress < 0.9:
            explanation = """Real-World Applications

Navigation Systems:
• GPS finds nearby restaurants, gas stations
• Route planning with traffic data
• Location-based services

Computer Graphics:
• Collision detection in games
• Ray tracing for realistic lighting
• Spatial data structures

Data Analysis:
• Clustering similar data points
• Nearest neighbor classification
• Anomaly detection

Database Systems:
• Spatial indexing for geographic data
• Range queries on large datasets
• Performance optimization"""
            
        else:
            explanation = """Key Advantages

Performance:
• Traditional: O(N²) - quadratic time
• Smart Network: O(N log N) - near linear
• Speedup: 100x-1000x for large datasets

Scalability:
• Handles millions of points efficiently
• Memory usage grows linearly
• Parallel processing friendly

Accuracy:
• Finds exact nearest neighbors
• Configurable distance thresholds
• Handles high-dimensional data

Reliability:
• Deterministic results
• Robust to data distribution
• Well-tested algorithms

This is why smart networks are essential
for modern computing applications!"""
        
        ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
                fontsize=13, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def draw_results(self, ax, progress):
        """Draw the final results"""
        # Show all points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        ax.scatter(self.target_points[:, 0], self.target_points[:, 1], 
                  c=TARGET_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Target Points', edgecolors='black', linewidth=2)
        
        # Calculate how many matches to show
        num_matches = int(progress * len(self.target_points))
        total_matches = 0
        
        # Show matches
        for i in range(min(num_matches, len(self.target_points))):
            target_point = self.target_points[i]
            
            # Find nearby search points
            nearby_indices = self.smart_network.query_ball_point(target_point, self.connection_distance)
            
            if len(nearby_indices) > 0:
                # Take the closest match
                best_match_idx = nearby_indices[0]
                search_point = self.search_points[best_match_idx]
                
                # Draw match line
                ax.plot([target_point[0], search_point[0]], 
                       [target_point[1], search_point[1]], 
                       'g-', linewidth=4, alpha=0.9)
                
                # Highlight matched points
                ax.scatter(target_point[0], target_point[1], 
                          c='green', s=EVENT_SIZE*1.5, alpha=0.9,
                          edgecolors='black', linewidth=3)
                ax.scatter(search_point[0], search_point[1], 
                          c='green', s=EVENT_SIZE*1.2, alpha=0.9,
                          edgecolors='black', linewidth=2)
                
                total_matches += 1
        
        # Add match count
        ax.text(0.02, 0.98, f'Matches Found: {total_matches}', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        ax.legend(fontsize=12)
        ax.set_title(f'Step 3: Final Matches ({total_matches} matches)', 
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
        self.draw_network_building(self.ax_network, progress)
        self.draw_connection_process(self.ax_connections, progress)
        self.draw_algorithm_explanation(self.ax_explanation, progress)
        self.draw_results(self.ax_results, progress)
        
        # Add progress indicator
        self.fig.text(0.02, 0.02, f'Progress: {progress*100:.1f}%', 
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        return []
    
    def create_animation(self):
        """Create and save the professional animation"""
        print("Creating smart network visualization for presentations...")
        print(f"Duration: {DURATION_SECONDS} seconds")
        print(f"Frames: {TOTAL_FRAMES}")
        print(f"FPS: {FPS}")
        print("Professional and clear - perfect for presentations!")
        
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
        output_file = "smart_network_visualization.mp4"
        print(f"Saving smart network animation to: {output_file}")
        
        try:
            anim.save(output_file, writer='ffmpeg', fps=FPS, bitrate=1800)
            print(f"✓ Smart network animation saved successfully: {output_file}")
        except Exception as e:
            print(f"✗ Error saving animation: {e}")
            print("Trying alternative format...")
            try:
                output_file_gif = "smart_network_visualization.gif"
                anim.save(output_file_gif, writer='pillow', fps=FPS)
                print(f"✓ Smart network animation saved as GIF: {output_file_gif}")
            except Exception as e2:
                print(f"✗ Error saving GIF: {e2}")
                print("Showing animation in window instead...")
                plt.show()
        
        return anim

def main():
    """Main function to create the smart network animation"""
    print("=" * 80)
    print("Smart Network Algorithm Visualization")
    print("=" * 80)
    print("This visualization demonstrates:")
    print("• How smart networks organize spatial data efficiently")
    print("• The connection process between target and search points")
    print("• Clear explanations suitable for presentations")
    print("• Real-world applications and benefits")
    print("=" * 80)
    print("Professional visualization for educational purposes!")
    print("=" * 80)
    
    # Create smart network animation
    visualizer = SmartNetworkVisualization()
    animation_obj = visualizer.create_animation()
    
    print("\nSmart network visualization complete!")
    print("The animation clearly demonstrates how efficient")
    print("spatial search algorithms work in practice.")

if __name__ == "__main__":
    main()












