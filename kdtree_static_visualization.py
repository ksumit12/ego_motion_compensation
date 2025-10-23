#!/usr/bin/env python3
"""
Static KDTree Structure Visualization

This script creates a static visualization showing the KDTree structure
in a tree-like format, similar to a "virus" or hierarchical map.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
import matplotlib.patches as mpatches

# Visual parameters
FIGURE_SIZE = (16, 12)
EVENT_SIZE = 100
PRED_COLOR = 'orange'
TREE_COLOR = 'purple'
SPLIT_COLOR = 'red'

class KDTreeStaticVisualization:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
        self.fig.suptitle('KDTree Structure Visualization - Tree-like Hierarchical Map', 
                         fontsize=18, fontweight='bold')
        
        # Setup subplots
        self.setup_subplots()
        
        # Generate synthetic event data
        self.generate_event_data()
        
    def setup_subplots(self):
        """Setup the four subplot areas"""
        # Top-left: KDTree structure with recursive splitting
        self.ax_tree = self.axes[0, 0]
        self.ax_tree.set_title('KDTree Structure - Recursive Space Division', fontweight='bold', fontsize=14)
        self.ax_tree.set_xlabel('X (pixels)')
        self.ax_tree.set_ylabel('Y (pixels)')
        self.ax_tree.grid(True, alpha=0.3)
        
        # Top-right: Tree hierarchy visualization
        self.ax_hierarchy = self.axes[0, 1]
        self.ax_hierarchy.set_title('KDTree Hierarchy - Tree Structure', fontweight='bold', fontsize=14)
        self.ax_hierarchy.axis('off')
        
        # Bottom-left: Query process visualization
        self.ax_query = self.axes[1, 0]
        self.ax_query.set_title('Spatial Query Process', fontweight='bold', fontsize=14)
        self.ax_query.set_xlabel('X (pixels)')
        self.ax_query.set_ylabel('Y (pixels)')
        self.ax_query.grid(True, alpha=0.3)
        
        # Bottom-right: Algorithm explanation
        self.ax_explanation = self.axes[1, 1]
        self.ax_explanation.set_title('Algorithm Explanation', fontweight='bold', fontsize=14)
        self.ax_explanation.axis('off')
        
        plt.tight_layout()
        
    def generate_event_data(self):
        """Generate synthetic event data for visualization"""
        np.random.seed(42)  # For reproducible results
        
        # Generate predicted events (orange dots)
        self.pred_events = np.array([
            [120, 160], [180, 190], [320, 110], [140, 310],
            [270, 260], [330, 190], [90, 340], [410, 90],
            [150, 180], [220, 220], [280, 120], [160, 280],
            [200, 150], [300, 200], [100, 250], [350, 300]
        ])
        
        # Build KDTree for predicted events
        self.pred_tree = cKDTree(self.pred_events)
        
    def draw_kdtree_with_regions(self, ax, points, depth=0, bounds=None, max_depth=4):
        """Draw KDTree structure with clear region divisions"""
        if bounds is None:
            bounds = [0, 500, 0, 400]  # xmin, xmax, ymin, ymax
            
        if len(points) == 0 or depth > max_depth:
            return
            
        if len(points) == 1:
            # Leaf node - draw point with region
            rect = Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2],
                            fill=True, facecolor='lightblue', alpha=0.3, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
            
            ax.scatter(points[0, 0], points[0, 1], c=TREE_COLOR, s=EVENT_SIZE, 
                      alpha=0.9, edgecolors='black', linewidth=3, zorder=10)
            ax.text(points[0, 0], points[0, 1]-30, f'Leaf\n{len(points)}', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            return
            
        # Choose splitting dimension (alternate between x and y)
        split_dim = depth % 2
        
        # Sort points by splitting dimension
        sorted_indices = np.argsort(points[:, split_dim])
        sorted_points = points[sorted_indices]
        
        # Find median
        median_idx = len(sorted_points) // 2
        median_value = sorted_points[median_idx, split_dim]
        
        # Draw current region
        region_color = plt.cm.viridis(depth / max_depth)
        rect = Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2],
                        fill=True, facecolor=region_color, alpha=0.2, 
                        edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Draw splitting line with label
        if split_dim == 0:  # Split on x-axis
            ax.axvline(x=median_value, color=SPLIT_COLOR, linestyle='-', linewidth=4, alpha=0.8)
            ax.text(median_value, bounds[3]-20, f'X={median_value:.0f}', 
                   ha='center', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            left_bounds = [bounds[0], median_value, bounds[2], bounds[3]]
            right_bounds = [median_value, bounds[1], bounds[2], bounds[3]]
        else:  # Split on y-axis
            ax.axhline(y=median_value, color=SPLIT_COLOR, linestyle='-', linewidth=4, alpha=0.8)
            ax.text(bounds[1]-20, median_value, f'Y={median_value:.0f}', 
                   ha='right', va='center', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            left_bounds = [bounds[0], bounds[1], bounds[2], median_value]
            right_bounds = [bounds[0], bounds[1], median_value, bounds[3]]
        
        # Add depth label
        ax.text(bounds[0]+10, bounds[3]-10, f'Depth {depth}', 
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Recursively draw left and right subtrees
        left_points = sorted_points[:median_idx]
        right_points = sorted_points[median_idx:]
        
        if len(left_points) > 0:
            self.draw_kdtree_with_regions(ax, left_points, depth + 1, left_bounds, max_depth)
        if len(right_points) > 0:
            self.draw_kdtree_with_regions(ax, right_points, depth + 1, right_bounds, max_depth)
    
    def draw_tree_hierarchy(self, ax):
        """Draw the tree hierarchy structure"""
        # Create a tree-like visualization
        tree_structure = """
        KDTree Hierarchy Structure
        
        Root Node (All Events)
        ├── Left Subtree (X < median)
        │   ├── Left-Left (Y < median)
        │   │   ├── Leaf 1
        │   │   └── Leaf 2
        │   └── Left-Right (Y ≥ median)
        │       ├── Leaf 3
        │       └── Leaf 4
        └── Right Subtree (X ≥ median)
            ├── Right-Left (Y < median)
            │   ├── Leaf 5
            │   └── Leaf 6
            └── Right-Right (Y ≥ median)
                ├── Leaf 7
                └── Leaf 8
        
        Query Process:
        1. Start at root
        2. Compare query point with split
        3. Traverse to appropriate subtree
        4. Continue until leaf node
        5. Check distance to all points in leaf
        """
        
        ax.text(0.05, 0.95, tree_structure, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    def draw_query_process(self, ax):
        """Draw the spatial query process"""
        # Show all events
        ax.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                  c=PRED_COLOR, s=EVENT_SIZE, alpha=0.7, 
                  label='Predicted Events', edgecolors='black', linewidth=2)
        
        # Draw KDTree structure
        self.draw_kdtree_with_regions(ax, self.pred_events, max_depth=3)
        
        # Add query point
        query_point = np.array([200, 200])
        ax.scatter(query_point[0], query_point[1], 
                  c='red', s=EVENT_SIZE*2, 
                  label='Query Point', edgecolors='black', linewidth=3)
        
        # Draw query radius
        circle = Circle(query_point, 50, fill=False, 
                       color='red', linewidth=3, linestyle='--')
        ax.add_patch(circle)
        
        # Query KDTree
        candidates = self.pred_tree.query_ball_point(query_point, 50)
        if len(candidates) > 0:
            candidate_points = self.pred_events[candidates]
            ax.scatter(candidate_points[:, 0], candidate_points[:, 1], 
                      c='green', s=EVENT_SIZE*1.5, alpha=0.9, 
                      label=f'Candidates ({len(candidates)})',
                      edgecolors='black', linewidth=2)
        
        ax.legend(fontsize=10)
        ax.set_title('Spatial Query: O(log N) Complexity', fontweight='bold', fontsize=14)
    
    def draw_algorithm_explanation(self, ax):
        """Draw algorithm explanation"""
        explanation = """
KDTree Algorithm for Event Matching:

1. BUILD PHASE:
   • Organize predicted events in KDTree
   • Recursively split space by X and Y
   • Create hierarchical structure
   • Time: O(N log N)

2. QUERY PHASE:
   • For each real event:
     - Query spatial candidates: O(log N)
     - Apply temporal gate: O(k) where k = candidates
     - Find best match: O(k)
   • Total: O(N log N)

3. BENEFITS:
   • Much faster than brute force O(N²)
   • Memory efficient O(N)
   • Handles high-dimensional data
   • Scales well with dataset size

4. SPATIAL CRITERIA:
   • Distance: √((x₁-x₂)² + (y₁-y₂)²) ≤ tolerance
   • Uses KDTree.query_ball_point()
   • Returns all points within radius

5. TEMPORAL CRITERIA:
   • Gate: |t_pred - (t_real + dt)| ≤ tolerance
   • Ensures causality
   • Filters spatial candidates
        """
        
        ax.text(0.05, 0.95, explanation, transform=ax.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    def create_visualization(self):
        """Create the complete static visualization"""
        print("Creating static KDTree visualization...")
        
        # Top-left: KDTree structure
        self.ax_tree.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                            c=PRED_COLOR, s=EVENT_SIZE, alpha=0.7, 
                            label='Predicted Events', edgecolors='black', linewidth=2)
        self.draw_kdtree_with_regions(self.ax_tree, self.pred_events, max_depth=4)
        self.ax_tree.legend(fontsize=10)
        
        # Top-right: Tree hierarchy
        self.draw_tree_hierarchy(self.ax_hierarchy)
        
        # Bottom-left: Query process
        self.draw_query_process(self.ax_query)
        
        # Bottom-right: Algorithm explanation
        self.draw_algorithm_explanation(self.ax_explanation)
        
        # Save the visualization
        output_file = "kdtree_static_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Static visualization saved: {output_file}")
        
        # Show the plot
        plt.show()
        
        return self.fig

def main():
    """Main function to create the static visualization"""
    print("=" * 70)
    print("Static KDTree Structure Visualization")
    print("=" * 70)
    print("This visualization shows:")
    print("1. KDTree structure with recursive space division")
    print("2. Tree hierarchy and traversal process")
    print("3. Spatial query process with O(log N) complexity")
    print("4. Complete algorithm explanation")
    print("=" * 70)
    
    # Create visualization
    visualizer = KDTreeStaticVisualization()
    fig = visualizer.create_visualization()
    
    print("\nStatic visualization complete!")
    print("The visualization shows the KDTree structure")
    print("in a tree-like hierarchical format.")

if __name__ == "__main__":
    main()











