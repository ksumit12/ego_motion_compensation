#!/usr/bin/env python3
"""
Virus-like KDTree Visualization for Laypeople

This script creates a simple, intuitive visualization that shows KDTree
as a spreading network/virus-like pattern that anyone can understand.
No technical knowledge required!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import time

# Animation parameters - Very slow and clear
FPS = 1  # Very slow
DURATION_SECONDS = 20  # Shorter but clearer
TOTAL_FRAMES = FPS * DURATION_SECONDS

# Visual parameters
FIGURE_SIZE = (16, 10)
EVENT_SIZE = 150
SPREAD_COLOR = 'red'
TARGET_COLOR = 'blue'
NETWORK_COLOR = 'green'
INFECTION_COLOR = 'orange'

class VirusLikeVisualization:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        self.fig.suptitle('How Smart Networks Find Connections (Like a Virus Spreading)', 
                         fontsize=20, fontweight='bold', color='darkblue')
        
        # Setup subplots
        self.setup_subplots()
        
        # Generate simple data
        self.generate_simple_data()
        
        # Initialize animation variables
        self.current_frame = 0
        self.infected_points = []
        self.connections = []
        
    def setup_subplots(self):
        """Setup the four subplot areas with simple titles"""
        # Top-left: The "virus" spreading
        self.ax_virus = self.axes[0, 0]
        self.ax_virus.set_title('🌐 Smart Network Growing', fontsize=16, fontweight='bold', color='darkred')
        self.ax_virus.set_xlabel('Space')
        self.ax_virus.set_ylabel('Space')
        self.ax_virus.grid(True, alpha=0.3)
        
        # Top-right: How it spreads
        self.ax_spread = self.axes[0, 1]
        self.ax_spread.set_title('🦠 How It Spreads', fontsize=16, fontweight='bold', color='darkgreen')
        self.ax_spread.set_xlabel('Space')
        self.ax_spread.set_ylabel('Space')
        self.ax_spread.grid(True, alpha=0.3)
        
        # Bottom-left: Simple explanation
        self.ax_explanation = self.axes[1, 0]
        self.ax_explanation.set_title('💡 What\'s Happening?', fontsize=16, fontweight='bold', color='darkblue')
        self.ax_explanation.axis('off')
        
        # Bottom-right: Results
        self.ax_results = self.axes[1, 1]
        self.ax_results.set_title('🎯 Final Connections', fontsize=16, fontweight='bold', color='purple')
        self.ax_results.set_xlabel('Space')
        self.ax_results.set_ylabel('Space')
        self.ax_results.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def generate_simple_data(self):
        """Generate simple, clear data points"""
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
        
        # Build the "smart network" (KDTree)
        self.smart_network = cKDTree(self.search_points)
        
        # Search radius (how close things need to be to connect)
        self.connection_distance = 50
        
    def draw_virus_spreading(self, ax, progress):
        """Draw the virus-like spreading pattern"""
        # Calculate how many "infections" to show
        num_infected = int(progress * len(self.search_points))
        
        # Show all search points
        ax.scatter(self.search_points[:, 0], self.search_points[:, 1], 
                  c='lightgray', s=EVENT_SIZE, alpha=0.5, 
                  label='Search Points', edgecolors='black', linewidth=1)
        
        # Show "infected" points (points that are part of the network)
        if num_infected > 0:
            infected_points = self.search_points[:num_infected]
            ax.scatter(infected_points[:, 0], infected_points[:, 1], 
                      c=INFECTION_COLOR, s=EVENT_SIZE*1.2, alpha=0.8, 
                      label=f'Network Points ({num_infected})',
                      edgecolors='black', linewidth=2)
        
        # Draw "infection" spreading lines
        if num_infected > 1:
            for i in range(1, num_infected):
                # Draw lines connecting infected points
                ax.plot([self.search_points[i-1, 0], self.search_points[i, 0]], 
                       [self.search_points[i-1, 1], self.search_points[i, 1]], 
                       'r-', linewidth=2, alpha=0.7)
        
        ax.legend(fontsize=12)
        ax.set_title(f'🌐 Smart Network Growing ({num_infected}/{len(self.search_points)})', 
                    fontsize=16, fontweight='bold', color='darkred')
    
    def draw_spread_process(self, ax, progress):
        """Draw how the spreading works"""
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
        
        ax.legend(fontsize=12)
        ax.set_title(f'🦠 Making Connections ({num_connections}/{len(self.target_points)})', 
                    fontsize=16, fontweight='bold', color='darkgreen')
    
    def draw_simple_explanation(self, ax, progress):
        """Draw simple explanation for laypeople"""
        if progress < 0.3:
            explanation = """🌐 What is a Smart Network?

Think of it like a virus spreading, but GOOD!

• We have many points scattered around
• The network grows by connecting nearby points
• It's like a spider web that builds itself
• The network helps us find things quickly

Why is this useful?
• Without network: Check every point (slow!)
• With network: Jump directly to nearby points (fast!)

It's like having a GPS for finding things!"""
            
        elif progress < 0.6:
            explanation = """🦠 How Does It Spread?

The network grows step by step:

1. Start with one point
2. Find nearby points within reach
3. Connect them with lines
4. Repeat for all points
5. Now we have a smart network!

The "infection" spreads outward:
• Each point can "infect" nearby points
• Connected points form clusters
• The network remembers where everything is

Result: Super fast searching!"""
            
        elif progress < 0.9:
            explanation = """🎯 What Happens Next?

The network helps us find matches:

1. We have target points (what we want)
2. We have search points (what we're looking with)
3. The network finds nearby search points
4. We connect targets to nearby searches
5. These connections are our "matches"

It's like:
• Target = Person looking for a friend
• Search = Potential friends nearby
• Network = Smart way to find them
• Match = Found a friend!"""
            
        else:
            explanation = """✅ Why This is Amazing!

Traditional way:
• Check every single point
• Takes forever with many points
• Like searching house by house

Smart Network way:
• Jump directly to nearby points
• Super fast even with millions of points
• Like having a GPS for everything

Real world uses:
• GPS navigation
• Social media friend suggestions
• Online shopping recommendations
• Medical diagnosis systems

The network makes computers super smart!"""
        
        ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
                fontsize=14, verticalalignment='top',
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
        ax.set_title(f'🎯 Final Connections ({total_matches} matches)', 
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
        self.draw_virus_spreading(self.ax_virus, progress)
        self.draw_spread_process(self.ax_spread, progress)
        self.draw_simple_explanation(self.ax_explanation, progress)
        self.draw_results(self.ax_results, progress)
        
        # Add progress indicator
        self.fig.text(0.02, 0.02, f'Progress: {progress*100:.1f}%', 
                     fontsize=14, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        return []
    
    def create_animation(self):
        """Create and save the virus-like animation"""
        print("Creating virus-like visualization for laypeople...")
        print(f"Duration: {DURATION_SECONDS} seconds")
        print(f"Frames: {TOTAL_FRAMES}")
        print(f"FPS: {FPS}")
        print("Simple and intuitive - no technical knowledge needed!")
        
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
        output_file = "virus_like_kdtree_visualization.mp4"
        print(f"Saving virus-like animation to: {output_file}")
        
        try:
            anim.save(output_file, writer='ffmpeg', fps=FPS, bitrate=1800)
            print(f"✓ Virus-like animation saved successfully: {output_file}")
        except Exception as e:
            print(f"✗ Error saving animation: {e}")
            print("Trying alternative format...")
            try:
                output_file_gif = "virus_like_kdtree_visualization.gif"
                anim.save(output_file_gif, writer='pillow', fps=FPS)
                print(f"✓ Virus-like animation saved as GIF: {output_file_gif}")
            except Exception as e2:
                print(f"✗ Error saving GIF: {e2}")
                print("Showing animation in window instead...")
                plt.show()
        
        return anim

def main():
    """Main function to create the virus-like animation"""
    print("=" * 80)
    print("🦠 Virus-like KDTree Visualization for Everyone")
    print("=" * 80)
    print("This visualization shows:")
    print("🌐 How smart networks grow like a virus spreading")
    print("🦠 How connections are made between nearby points")
    print("💡 Simple explanations that anyone can understand")
    print("🎯 How the network helps find matches quickly")
    print("=" * 80)
    print("No technical knowledge required!")
    print("Perfect for presentations to non-technical audiences!")
    print("=" * 80)
    
    # Create virus-like animation
    visualizer = VirusLikeVisualization()
    animation_obj = visualizer.create_animation()
    
    print("\n🦠 Virus-like visualization complete!")
    print("The animation shows how smart networks work")
    print("in a way that anyone can understand!")

if __name__ == "__main__":
    main()
