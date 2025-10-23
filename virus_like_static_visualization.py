#!/usr/bin/env python3
"""
Static Virus-like KDTree Visualization

This creates a static image showing the virus-like spreading pattern
of KDTree structure - perfect for presentations and papers.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import matplotlib.patches as mpatches

# Visual parameters
FIGURE_SIZE = (20, 12)
EVENT_SIZE = 200
SPREAD_COLOR = 'red'
TARGET_COLOR = 'blue'
NETWORK_COLOR = 'green'
INFECTION_COLOR = 'orange'

class StaticVirusVisualization:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
        self.fig.suptitle('Smart Network Growth: Like a Virus Spreading (But Good!)', 
                         fontsize=24, fontweight='bold', color='darkblue')
        
        # Setup subplots
        self.setup_subplots()
        
        # Generate simple data
        self.generate_simple_data()
        
    def setup_subplots(self):
        """Setup the six subplot areas"""
        # Top-left: Initial state
        self.ax_initial = self.axes[0, 0]
        self.ax_initial.set_title('Step 1: Starting Points', fontsize=18, fontweight='bold', color='darkblue')
        self.ax_initial.set_xlabel('Space')
        self.ax_initial.set_ylabel('Space')
        self.ax_initial.grid(True, alpha=0.3)
        
        # Top-center: Network growing
        self.ax_growing = self.axes[0, 1]
        self.ax_growing.set_title('Step 2: Network Growing', fontsize=18, fontweight='bold', color='darkred')
        self.ax_growing.set_xlabel('Space')
        self.ax_growing.set_ylabel('Space')
        self.ax_growing.grid(True, alpha=0.3)
        
        # Top-right: Full network
        self.ax_full = self.axes[0, 2]
        self.ax_full.set_title('Step 3: Complete Network', fontsize=18, fontweight='bold', color='darkgreen')
        self.ax_full.set_xlabel('Space')
        self.ax_full.set_ylabel('Space')
        self.ax_full.grid(True, alpha=0.3)
        
        # Bottom-left: Finding connections
        self.ax_connections = self.axes[1, 0]
        self.ax_connections.set_title('Step 4: Finding Connections', fontsize=18, fontweight='bold', color='darkorange')
        self.ax_connections.set_xlabel('Space')
        self.ax_connections.set_ylabel('Space')
        self.ax_connections.grid(True, alpha=0.3)
        
        # Bottom-center: Final matches
        self.ax_matches = self.axes[1, 1]
        self.ax_matches.set_title('Step 5: Final Matches', fontsize=18, fontweight='bold', color='darkpurple')
        self.ax_matches.set_xlabel('Space')
        self.ax_matches.set_ylabel('Space')
        self.ax_matches.grid(True, alpha=0.3)
        
        # Bottom-right: Explanation
        self.ax_explanation = self.axes[1, 2]
        self.ax_explanation.set_title('Why This is Amazing!', fontsize=18, fontweight='bold', color='darkred')
        self.ax_explanation.axis('off')
        
        plt.tight_layout()
        
    def generate_simple_data(self):
        """Generate simple, clear data points"""
        np.random.seed(42)  # For reproducible results
        
        # Generate "target" points (blue dots) - what we're looking for
        self.target_points = np.array([
            [100, 150], [200, 200], [300, 100], [150, 300],
            [250, 250], [350, 200], [100, 350], [400, 100],
            [180, 180], [320, 320], [120, 280], [280, 120]
        ])
        
        # Generate "search" points (red dots) - what we're searching with
        self.search_points = np.array([
            [120, 160], [180, 190], [320, 110], [140, 310],
            [270, 260], [330, 190], [90, 340], [410, 90],
            [150, 180], [220, 220], [280, 120], [160, 280],
            [200, 150], [300, 200], [100, 250], [350, 300]
        ])
        
        # Build the "smart network" (KDTree)
        self.smart_network = cKDTree(self.search_points)
        
        # Search radius (how close things need to be to connect)
        self.connection_distance = 60
        
    def draw_initial_state(self, ax):
        """Draw the initial state - just points"""
        # Show all points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.7, 
                  label='Search Points', edgecolors='black', linewidth=2)
        ax.scatter(self.target_points[:, 0], self.target_points[:, 1], 
                  c=TARGET_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Target Points', edgecolors='black', linewidth=2)
        
        ax.legend(fontsize=14)
        ax.text(0.02, 0.98, 'We start with scattered points', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    def draw_growing_network(self, ax):
        """Draw the network growing"""
        # Show all points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        ax.scatter(self.target_points[:, 0], self.target_points[:, 1], 
                  c=TARGET_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Target Points', edgecolors='black', linewidth=2)
        
        # Show some connections growing
        num_connections = 6
        for i in range(num_connections):
            if i < len(self.search_points) - 1:
                ax.plot([self.search_points[i, 0], self.search_points[i+1, 0]], 
                       [self.search_points[i, 1], self.search_points[i+1, 1]], 
                       'r-', linewidth=3, alpha=0.8)
        
        # Highlight connected points
        connected_points = self.search_points[:num_connections+1]
        ax.scatter(connected_points[:, 0], connected_points[:, 1], 
                  c=INFECTION_COLOR, s=EVENT_SIZE*1.2, alpha=0.9,
                  label=f'Connected Points ({num_connections+1})',
                  edgecolors='black', linewidth=2)
        
        ax.legend(fontsize=14)
        ax.text(0.02, 0.98, 'Network starts growing', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    def draw_full_network(self, ax):
        """Draw the complete network"""
        # Show all points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        ax.scatter(self.target_points[:, 0], self.target_points[:, 1], 
                  c=TARGET_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Target Points', edgecolors='black', linewidth=2)
        
        # Show full network connections
        for i in range(len(self.search_points) - 1):
            ax.plot([self.search_points[i, 0], self.search_points[i+1, 0]], 
                   [self.search_points[i, 1], self.search_points[i+1, 1]], 
                   'r-', linewidth=2, alpha=0.6)
        
        # Highlight all connected points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c=INFECTION_COLOR, s=EVENT_SIZE*1.1, alpha=0.8,
                  label=f'Full Network ({len(self.search_points)})',
                  edgecolors='black', linewidth=1)
        
        ax.legend(fontsize=14)
        ax.text(0.02, 0.98, 'Complete smart network', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    def draw_connections(self, ax):
        """Draw how connections are found"""
        # Show all points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        ax.scatter(self.target_points[:, 0], self.target_points[:, 1], 
                  c=TARGET_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Target Points', edgecolors='black', linewidth=2)
        
        # Show some connections being found
        num_targets = 6
        for i in range(min(num_targets, len(self.target_points))):
            target_point = self.target_points[i]
            
            # Find nearby search points
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
        
        ax.legend(fontsize=14)
        ax.text(0.02, 0.98, f'Finding connections ({num_targets} targets)', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    def draw_final_matches(self, ax):
        """Draw the final matches"""
        # Show all points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        ax.scatter(self.target_points[:, 0], self.target_points[:, 1], 
                  c=TARGET_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                  label='Target Points', edgecolors='black', linewidth=2)
        
        # Show final matches
        total_matches = 0
        for i in range(len(self.target_points)):
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
        ax.text(0.02, 0.98, f'Final Matches: {total_matches}', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
        
        ax.legend(fontsize=14)
        ax.text(0.02, 0.02, 'Perfect matches found!', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    def draw_explanation(self, ax):
        """Draw the explanation"""
        explanation = """Why This Network is Amazing!

Traditional Way (Slow):
• Check every single point
• Takes forever with many points
• Like searching house by house

Smart Network Way (Fast):
• Jump directly to nearby points
• Super fast even with millions of points
• Like having GPS for everything

Real World Uses:
• GPS navigation systems
• Social media friend suggestions
• Online shopping recommendations
• Medical diagnosis systems
• Weather prediction models
• Stock market analysis
• Image recognition
• Voice assistants

The network makes computers super smart!
It's like giving them a superpower!"""
        
        ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
                fontsize=16, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def create_visualization(self):
        """Create the complete static visualization"""
        print("Creating static virus-like visualization...")
        
        # Draw all six views
        self.draw_initial_state(self.ax_initial)
        self.draw_growing_network(self.ax_growing)
        self.draw_full_network(self.ax_full)
        self.draw_connections(self.ax_connections)
        self.draw_final_matches(self.ax_matches)
        self.draw_explanation(self.ax_explanation)
        
        # Save the visualization
        output_file = "virus_like_kdtree_static.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Static virus-like visualization saved: {output_file}")
        
        # Show the plot
        plt.show()
        
        return self.fig

def main():
    """Main function to create the static visualization"""
    print("=" * 80)
    print("🦠 Static Virus-like KDTree Visualization")
    print("=" * 80)
    print("This visualization shows:")
    print("🌐 Step-by-step network growth")
    print("🦠 How connections spread like a virus")
    print("💡 Why this is amazing for computers")
    print("🎯 Perfect for presentations and papers")
    print("=" * 80)
    print("No technical knowledge required!")
    print("=" * 80)
    
    # Create visualization
    visualizer = StaticVirusVisualization()
    fig = visualizer.create_visualization()
    
    print("\n🦠 Static virus-like visualization complete!")
    print("The visualization shows how smart networks work")
    print("in a way that anyone can understand!")

if __name__ == "__main__":
    main()












