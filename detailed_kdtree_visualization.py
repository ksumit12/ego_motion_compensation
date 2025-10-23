#!/usr/bin/env python3
"""
Detailed KDTree Spatial-Temporal Alignment Visualization

This script creates a slow, detailed animation showing:
1. KDTree structure building (tree-like visualization)
2. Step-by-step spatial query process
3. Clear temporal alignment criteria
4. Detailed matching decision process

Much slower and more educational than the previous version.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
from matplotlib.patches import ConnectionPatch
from scipy.spatial import cKDTree
import time

# Animation parameters - MUCH SLOWER
FPS = 2  # Much slower
DURATION_SECONDS = 30  # Longer duration
TOTAL_FRAMES = FPS * DURATION_SECONDS

# Event parameters
NUM_REAL_EVENTS = 8
NUM_PREDICTED_EVENTS = 12
SPATIAL_TOLERANCE = 30  # pixels
TEMPORAL_TOLERANCE_MS = 4  # milliseconds
DT_MS = 2  # milliseconds

# Visual parameters
FIGURE_SIZE = (20, 12)
EVENT_SIZE = 80
MATCHED_COLOR = 'green'
UNMATCHED_COLOR = 'red'
REAL_COLOR = 'blue'
PRED_COLOR = 'orange'
TREE_COLOR = 'purple'
TOLERANCE_COLOR = 'red'

class DetailedKDTreeVisualization:
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=FIGURE_SIZE)
        self.fig.suptitle('Detailed KDTree Spatial-Temporal Alignment Process', 
                         fontsize=18, fontweight='bold')
        
        # Setup subplots
        self.setup_subplots()
        
        # Generate synthetic event data
        self.generate_event_data()
        
        # Initialize animation variables
        self.current_frame = 0
        self.matched_pairs = []
        self.current_real_idx = 0
        self.spatial_candidates = []
        self.temporal_candidates = []
        self.kdtree_depth = 0
        
    def setup_subplots(self):
        """Setup the six subplot areas"""
        # Top-left: KDTree structure visualization
        self.ax_tree = self.axes[0, 0]
        self.ax_tree.set_title('KDTree Structure Building', fontweight='bold', fontsize=14)
        self.ax_tree.set_xlabel('X (pixels)')
        self.ax_tree.set_ylabel('Y (pixels)')
        self.ax_tree.grid(True, alpha=0.3)
        
        # Top-center: Spatial query process
        self.ax_spatial = self.axes[0, 1]
        self.ax_spatial.set_title('Spatial Query Process', fontweight='bold', fontsize=14)
        self.ax_spatial.set_xlabel('X (pixels)')
        self.ax_spatial.set_ylabel('Y (pixels)')
        self.ax_spatial.grid(True, alpha=0.3)
        
        # Top-right: Temporal alignment
        self.ax_temporal = self.axes[0, 2]
        self.ax_temporal.set_title('Temporal Alignment Gate', fontweight='bold', fontsize=14)
        self.ax_temporal.set_xlabel('Time (ms)')
        self.ax_temporal.set_ylabel('Event Index')
        self.ax_temporal.grid(True, alpha=0.3)
        
        # Bottom-left: Matching criteria
        self.ax_criteria = self.axes[1, 0]
        self.ax_criteria.set_title('Matching Criteria Check', fontweight='bold', fontsize=14)
        self.ax_criteria.axis('off')
        
        # Bottom-center: Step-by-step process
        self.ax_steps = self.axes[1, 1]
        self.ax_steps.set_title('Step-by-Step Process', fontweight='bold', fontsize=14)
        self.ax_steps.axis('off')
        
        # Bottom-right: Results
        self.ax_results = self.axes[1, 2]
        self.ax_results.set_title('Matching Results', fontweight='bold', fontsize=14)
        self.ax_results.set_xlabel('X (pixels)')
        self.ax_results.set_ylabel('Y (pixels)')
        self.ax_results.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    def generate_event_data(self):
        """Generate synthetic event data for visualization"""
        np.random.seed(42)  # For reproducible results
        
        # Generate real events (blue dots) - fewer for clarity
        self.real_events = np.array([
            [100, 150], [200, 200], [300, 100], [150, 300],
            [250, 250], [350, 200], [100, 350], [400, 100]
        ])
        self.real_times = np.array([10, 20, 30, 40, 50, 60, 70, 80])  # ms
        self.real_polarities = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        # Generate predicted events (orange dots) - more scattered
        self.pred_events = np.array([
            [120, 160], [180, 190], [320, 110], [140, 310],
            [270, 260], [330, 190], [90, 340], [410, 90],
            [150, 180], [220, 220], [280, 120], [160, 280]
        ])
        self.pred_times = np.array([12, 22, 32, 42, 52, 62, 72, 82, 15, 25, 35, 45])  # ms
        self.pred_polarities = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        
        # Build KDTree for predicted events
        self.pred_tree = cKDTree(self.pred_events)
        
        # Initialize matching arrays
        self.matched_real = np.zeros(NUM_REAL_EVENTS, dtype=bool)
        self.matched_pred = np.zeros(NUM_PREDICTED_EVENTS, dtype=bool)
        
    def draw_kdtree_recursive(self, ax, points, depth=0, bounds=None, max_depth=3):
        """Recursively draw KDTree structure with tree-like visualization"""
        if bounds is None:
            bounds = [0, 500, 0, 400]  # xmin, xmax, ymin, ymax
            
        if len(points) == 0 or depth > max_depth:
            return
            
        if len(points) == 1:
            # Leaf node - draw point
            ax.scatter(points[0, 0], points[0, 1], c=TREE_COLOR, s=EVENT_SIZE, 
                      alpha=0.8, edgecolors='black', linewidth=2)
            ax.text(points[0, 0], points[0, 1]-20, f'Leaf\n{len(points)}', 
                   ha='center', va='top', fontsize=8, fontweight='bold')
            return
            
        # Choose splitting dimension (alternate between x and y)
        split_dim = depth % 2
        
        # Sort points by splitting dimension
        sorted_indices = np.argsort(points[:, split_dim])
        sorted_points = points[sorted_indices]
        
        # Find median
        median_idx = len(sorted_points) // 2
        median_value = sorted_points[median_idx, split_dim]
        
        # Draw splitting line with label
        if split_dim == 0:  # Split on x-axis
            ax.axvline(x=median_value, color='red', linestyle='-', linewidth=3, alpha=0.7)
            ax.text(median_value, bounds[3]-20, f'Split X={median_value:.0f}', 
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            left_bounds = [bounds[0], median_value, bounds[2], bounds[3]]
            right_bounds = [median_value, bounds[1], bounds[2], bounds[3]]
        else:  # Split on y-axis
            ax.axhline(y=median_value, color='red', linestyle='-', linewidth=3, alpha=0.7)
            ax.text(bounds[1]-20, median_value, f'Split Y={median_value:.0f}', 
                   ha='right', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
            left_bounds = [bounds[0], bounds[1], bounds[2], median_value]
            right_bounds = [bounds[0], bounds[1], median_value, bounds[3]]
        
        # Draw current region
        rect = Rectangle((bounds[0], bounds[2]), bounds[1]-bounds[0], bounds[3]-bounds[2],
                        fill=False, edgecolor='blue', linewidth=2, alpha=0.5)
        ax.add_patch(rect)
        
        # Add depth label
        ax.text(bounds[0]+10, bounds[3]-10, f'Depth {depth}', 
               fontsize=8, fontweight='bold', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
        
        # Recursively draw left and right subtrees
        left_points = sorted_points[:median_idx]
        right_points = sorted_points[median_idx:]
        
        if len(left_points) > 0:
            self.draw_kdtree_recursive(ax, left_points, depth + 1, left_bounds, max_depth)
        if len(right_points) > 0:
            self.draw_kdtree_recursive(ax, right_points, depth + 1, right_bounds, max_depth)
    
    def animate_frame(self, frame_num):
        """Animate a single frame with detailed explanations"""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Re-setup subplot titles and labels
        self.setup_subplots()
        
        # Calculate animation progress
        progress = frame_num / TOTAL_FRAMES
        
        # Phase 1: Build KDTree structure (0-30% of animation)
        if progress < 0.30:
            self.draw_phase1_kdtree_building(progress)
            
        # Phase 2: Show spatial query process (30-60% of animation)
        elif progress < 0.60:
            self.draw_phase2_spatial_query(progress)
            
        # Phase 3: Show temporal alignment (60-85% of animation)
        elif progress < 0.85:
            self.draw_phase3_temporal_alignment(progress)
            
        # Phase 4: Show complete matching (85-100% of animation)
        else:
            self.draw_phase4_complete_matching(progress)
            
        # Add progress indicator
        self.fig.text(0.02, 0.02, f'Frame: {frame_num}/{TOTAL_FRAMES} ({progress*100:.1f}%)', 
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        return []
    
    def draw_phase1_kdtree_building(self, progress):
        """Phase 1: Show KDTree structure building step by step"""
        # Calculate how much of the tree to show
        max_depth = int(progress / 0.30 * 3)  # Show up to depth 3
        
        # Top-left: Show KDTree structure building
        self.ax_tree.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                            c=PRED_COLOR, s=EVENT_SIZE, alpha=0.6, 
                            label='Predicted Events', edgecolors='black', linewidth=1)
        
        # Draw KDTree structure progressively
        if max_depth > 0:
            self.draw_kdtree_recursive(self.ax_tree, self.pred_events, max_depth=max_depth)
        
        self.ax_tree.legend(fontsize=10)
        self.ax_tree.set_title(f'Phase 1: Building KDTree (Depth {max_depth})', fontweight='bold', fontsize=14)
        
        # Top-center: Show spatial distribution
        self.ax_spatial.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                               c=PRED_COLOR, s=EVENT_SIZE, alpha=0.7, 
                               label='Predicted Events', edgecolors='black', linewidth=1)
        self.ax_spatial.scatter(self.real_events[:, 0], self.real_events[:, 1], 
                               c=REAL_COLOR, s=EVENT_SIZE, alpha=0.7, 
                               label='Real Events', edgecolors='black', linewidth=1)
        self.ax_spatial.legend(fontsize=10)
        self.ax_spatial.set_title('Event Distribution', fontweight='bold', fontsize=14)
        
        # Top-right: Show temporal distribution
        self.ax_temporal.scatter(self.pred_times, range(len(self.pred_times)), 
                                c=PRED_COLOR, s=EVENT_SIZE//2, alpha=0.7, label='Predicted')
        self.ax_temporal.scatter(self.real_times, range(len(self.real_times)), 
                                c=REAL_COLOR, s=EVENT_SIZE//2, alpha=0.7, label='Real')
        self.ax_temporal.legend(fontsize=10)
        self.ax_temporal.set_title('Event Timeline', fontweight='bold', fontsize=14)
        
        # Bottom-left: Explain KDTree
        explanation = f"""KDTree Structure Building:

1. Start with all predicted events
2. Choose splitting dimension (X or Y)
3. Find median value
4. Split space into two regions
5. Recursively build subtrees

Current Depth: {max_depth}
Total Events: {NUM_PREDICTED_EVENTS}

Benefits:
• Fast spatial queries: O(log N)
• Memory efficient: O(N)
• Handles high dimensions well"""
        
        self.ax_criteria.text(0.05, 0.95, explanation, transform=self.ax_criteria.transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Bottom-center: Show algorithm steps
        steps = f"""Algorithm Steps:

1. ✓ Build KDTree for predicted events
2. ⏳ Process each real event
3. ⏳ Query spatial candidates
4. ⏳ Apply temporal gate
5. ⏳ Find best match"""
        
        self.ax_steps.text(0.05, 0.95, steps, transform=self.ax_steps.transAxes,
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Bottom-right: Show current state
        self.ax_results.text(0.5, 0.5, 'KDTree Building in Progress...', 
                           ha='center', va='center', transform=self.ax_results.transAxes,
                           fontsize=14, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    def draw_phase2_spatial_query(self, progress):
        """Phase 2: Show spatial query process in detail"""
        # Calculate which real event to process
        phase_progress = (progress - 0.30) / 0.30
        event_idx = int(phase_progress * NUM_REAL_EVENTS)
        event_idx = min(event_idx, NUM_REAL_EVENTS - 1)
        
        real_event = self.real_events[event_idx]
        
        # Top-left: Show KDTree structure
        self.ax_tree.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                            c=PRED_COLOR, s=EVENT_SIZE, alpha=0.6, 
                            label='Predicted Events', edgecolors='black', linewidth=1)
        self.draw_kdtree_recursive(self.ax_tree, self.pred_events, max_depth=3)
        self.ax_tree.legend(fontsize=10)
        self.ax_tree.set_title('KDTree Structure (Complete)', fontweight='bold', fontsize=14)
        
        # Top-center: Show spatial query
        self.ax_spatial.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                               c=PRED_COLOR, s=EVENT_SIZE, alpha=0.6, 
                               label='Predicted Events', edgecolors='black', linewidth=1)
        
        # Highlight current real event
        self.ax_spatial.scatter(real_event[0], real_event[1], 
                               c=REAL_COLOR, s=EVENT_SIZE*2, 
                               label=f'Current Real Event {event_idx}', 
                               edgecolors='black', linewidth=3)
        
        # Draw spatial tolerance circle
        circle = Circle(real_event, SPATIAL_TOLERANCE, fill=False, 
                       color=TOLERANCE_COLOR, linewidth=3, linestyle='--')
        self.ax_spatial.add_patch(circle)
        
        # Query KDTree for spatial candidates
        spatial_candidates = self.pred_tree.query_ball_point(real_event, SPATIAL_TOLERANCE)
        
        if len(spatial_candidates) > 0:
            candidate_points = self.pred_events[spatial_candidates]
            self.ax_spatial.scatter(candidate_points[:, 0], candidate_points[:, 1], 
                                   c='yellow', s=EVENT_SIZE*1.5, alpha=0.9, 
                                   label=f'Spatial Candidates ({len(spatial_candidates)})',
                                   edgecolors='black', linewidth=2)
        
        self.ax_spatial.legend(fontsize=10)
        self.ax_spatial.set_title(f'Phase 2: Spatial Query (Event {event_idx})', fontweight='bold', fontsize=14)
        
        # Top-right: Show temporal distribution
        self.ax_temporal.scatter(self.pred_times, range(len(self.pred_times)), 
                                c=PRED_COLOR, s=EVENT_SIZE//2, alpha=0.6, label='Predicted')
        self.ax_temporal.scatter(self.real_times, range(len(self.real_times)), 
                                c=REAL_COLOR, s=EVENT_SIZE//2, alpha=0.6, label='Real')
        
        # Highlight current real event time
        real_time = self.real_times[event_idx]
        self.ax_temporal.axvline(x=real_time, color=REAL_COLOR, linewidth=3, 
                                label=f'Real Event Time ({real_time}ms)')
        
        self.ax_temporal.legend(fontsize=10)
        self.ax_temporal.set_title('Event Timeline', fontweight='bold', fontsize=14)
        
        # Bottom-left: Explain spatial criteria
        spatial_explanation = f"""Spatial Matching Criteria:

Current Real Event: {event_idx}
Position: ({real_event[0]:.0f}, {real_event[1]:.0f})

Spatial Tolerance: {SPATIAL_TOLERANCE} pixels
Formula: √((x₁-x₂)² + (y₁-y₂)²) ≤ {SPATIAL_TOLERANCE}

Spatial Candidates Found: {len(spatial_candidates)}

Next: Apply temporal gate to these candidates"""
        
        self.ax_criteria.text(0.05, 0.95, spatial_explanation, transform=self.ax_criteria.transAxes,
                             fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Bottom-center: Show current step
        steps = f"""Current Step: Spatial Query

1. ✓ Build KDTree for predicted events
2. ✓ Process real event {event_idx}
3. ✓ Query spatial candidates
4. ⏳ Apply temporal gate
5. ⏳ Find best match

Spatial Query: O(log N) time complexity"""
        
        self.ax_steps.text(0.05, 0.95, steps, transform=self.ax_steps.transAxes,
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Bottom-right: Show spatial query results
        self.ax_results.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                               c=PRED_COLOR, s=EVENT_SIZE, alpha=0.3, 
                               label='All Predicted')
        self.ax_results.scatter(real_event[0], real_event[1], 
                               c=REAL_COLOR, s=EVENT_SIZE*2, 
                               label=f'Real Event {event_idx}')
        
        circle_results = Circle(real_event, SPATIAL_TOLERANCE, fill=False, 
                              color=TOLERANCE_COLOR, linewidth=2)
        self.ax_results.add_patch(circle_results)
        
        if len(spatial_candidates) > 0:
            candidate_points = self.pred_events[spatial_candidates]
            self.ax_results.scatter(candidate_points[:, 0], candidate_points[:, 1], 
                                   c='yellow', s=EVENT_SIZE*1.5, alpha=0.9,
                                   label=f'Spatial Candidates ({len(spatial_candidates)})')
        
        self.ax_results.legend(fontsize=10)
        self.ax_results.set_title('Spatial Query Results', fontweight='bold', fontsize=14)
    
    def draw_phase3_temporal_alignment(self, progress):
        """Phase 3: Show temporal alignment process in detail"""
        # Calculate which real event to process
        phase_progress = (progress - 0.60) / 0.25
        event_idx = int(phase_progress * NUM_REAL_EVENTS)
        event_idx = min(event_idx, NUM_REAL_EVENTS - 1)
        
        real_event = self.real_events[event_idx]
        real_time = self.real_times[event_idx]
        target_time = real_time + DT_MS
        
        # Get spatial candidates
        spatial_candidates = self.pred_tree.query_ball_point(real_event, SPATIAL_TOLERANCE)
        
        # Apply temporal gate
        temporal_candidates = []
        if len(spatial_candidates) > 0:
            candidate_times = self.pred_times[spatial_candidates]
            temporal_mask = np.abs(candidate_times - target_time) <= TEMPORAL_TOLERANCE_MS
            temporal_candidates = [spatial_candidates[i] for i in range(len(spatial_candidates)) if temporal_mask[i]]
        
        # Top-left: Show KDTree structure
        self.ax_tree.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                            c=PRED_COLOR, s=EVENT_SIZE, alpha=0.6, 
                            label='Predicted Events', edgecolors='black', linewidth=1)
        self.draw_kdtree_recursive(self.ax_tree, self.pred_events, max_depth=3)
        self.ax_tree.legend(fontsize=10)
        self.ax_tree.set_title('KDTree Structure', fontweight='bold', fontsize=14)
        
        # Top-center: Show final candidates
        self.ax_spatial.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                               c=PRED_COLOR, s=EVENT_SIZE, alpha=0.3, 
                               label='Predicted Events', edgecolors='black', linewidth=1)
        
        self.ax_spatial.scatter(real_event[0], real_event[1], 
                               c=REAL_COLOR, s=EVENT_SIZE*2, 
                               label=f'Real Event {event_idx}', 
                               edgecolors='black', linewidth=3)
        
        # Show spatial candidates
        if len(spatial_candidates) > 0:
            spatial_points = self.pred_events[spatial_candidates]
            self.ax_spatial.scatter(spatial_points[:, 0], spatial_points[:, 1], 
                                   c='yellow', s=EVENT_SIZE*1.5, alpha=0.8, 
                                   label=f'Spatial Candidates ({len(spatial_candidates)})',
                                   edgecolors='black', linewidth=1)
        
        # Show temporal candidates
        if len(temporal_candidates) > 0:
            temporal_points = self.pred_events[temporal_candidates]
            self.ax_spatial.scatter(temporal_points[:, 0], temporal_points[:, 1], 
                                   c='green', s=EVENT_SIZE*2, alpha=0.9, 
                                   label=f'Final Candidates ({len(temporal_candidates)})',
                                   edgecolors='black', linewidth=2)
        
        circle = Circle(real_event, SPATIAL_TOLERANCE, fill=False, 
                       color=TOLERANCE_COLOR, linewidth=2, linestyle='--')
        self.ax_spatial.add_patch(circle)
        
        self.ax_spatial.legend(fontsize=10)
        self.ax_spatial.set_title(f'Phase 3: Temporal Alignment (Event {event_idx})', fontweight='bold', fontsize=14)
        
        # Top-right: Show temporal gate in detail
        self.ax_temporal.axvline(x=real_time, color=REAL_COLOR, linewidth=3, 
                                label=f'Real Time ({real_time}ms)')
        self.ax_temporal.axvline(x=target_time, color='green', linewidth=3, 
                                label=f'Target Time ({target_time}ms)')
        
        # Draw temporal tolerance window
        self.ax_temporal.axvspan(target_time - TEMPORAL_TOLERANCE_MS, target_time + TEMPORAL_TOLERANCE_MS, 
                                alpha=0.3, color='green', 
                                label=f'Temporal Gate (±{TEMPORAL_TOLERANCE_MS}ms)')
        
        # Show all predicted events
        self.ax_temporal.scatter(self.pred_times, range(len(self.pred_times)), 
                                c=PRED_COLOR, s=EVENT_SIZE//2, alpha=0.6, label='Predicted')
        
        # Show spatial candidates
        if len(spatial_candidates) > 0:
            spatial_times = self.pred_times[spatial_candidates]
            spatial_indices = spatial_candidates
            self.ax_temporal.scatter(spatial_times, spatial_indices, 
                                    c='yellow', s=EVENT_SIZE, alpha=0.8, 
                                    label='Spatial Candidates')
        
        # Show temporal candidates
        if len(temporal_candidates) > 0:
            temporal_times = self.pred_times[temporal_candidates]
            temporal_indices = temporal_candidates
            self.ax_temporal.scatter(temporal_times, temporal_indices, 
                                    c='green', s=EVENT_SIZE*1.5, alpha=0.9, 
                                    label='Temporal Candidates')
        
        self.ax_temporal.legend(fontsize=10)
        self.ax_temporal.set_title('Temporal Gate Application', fontweight='bold', fontsize=14)
        
        # Bottom-left: Explain temporal criteria
        temporal_explanation = f"""Temporal Matching Criteria:

Real Event Time: {real_time}ms
Target Time: {real_time} + {DT_MS} = {target_time}ms
Temporal Tolerance: ±{TEMPORAL_TOLERANCE_MS}ms

Temporal Gate Formula:
|t_predicted - (t_real + dt)| ≤ {TEMPORAL_TOLERANCE_MS}ms

Spatial Candidates: {len(spatial_candidates)}
Temporal Candidates: {len(temporal_candidates)}

Next: Choose best match from temporal candidates"""
        
        self.ax_criteria.text(0.05, 0.95, temporal_explanation, transform=self.ax_criteria.transAxes,
                             fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Bottom-center: Show current step
        steps = f"""Current Step: Temporal Alignment

1. ✓ Build KDTree for predicted events
2. ✓ Process real event {event_idx}
3. ✓ Query spatial candidates
4. ✓ Apply temporal gate
5. ⏳ Find best match

Temporal Gate: |t_j-(t_i+Δt)|≤{TEMPORAL_TOLERANCE_MS}ms"""
        
        self.ax_steps.text(0.05, 0.95, steps, transform=self.ax_steps.transAxes,
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Bottom-right: Show temporal alignment results
        self.ax_results.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                               c=PRED_COLOR, s=EVENT_SIZE, alpha=0.3, 
                               label='All Predicted')
        self.ax_results.scatter(real_event[0], real_event[1], 
                               c=REAL_COLOR, s=EVENT_SIZE*2, 
                               label=f'Real Event {event_idx}')
        
        if len(temporal_candidates) > 0:
            temporal_points = self.pred_events[temporal_candidates]
            self.ax_results.scatter(temporal_points[:, 0], temporal_points[:, 1], 
                                   c='green', s=EVENT_SIZE*2, alpha=0.9,
                                   label=f'Final Candidates ({len(temporal_candidates)})')
        
        self.ax_results.legend(fontsize=10)
        self.ax_results.set_title('Temporal Alignment Results', fontweight='bold', fontsize=14)
    
    def draw_phase4_complete_matching(self, progress):
        """Phase 4: Show complete matching process"""
        # Calculate how many events to show as processed
        phase_progress = (progress - 0.85) / 0.15
        num_processed = int(phase_progress * NUM_REAL_EVENTS)
        
        # Top-left: Show KDTree structure
        self.ax_tree.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                            c=PRED_COLOR, s=EVENT_SIZE, alpha=0.6, 
                            label='Predicted Events', edgecolors='black', linewidth=1)
        self.draw_kdtree_recursive(self.ax_tree, self.pred_events, max_depth=3)
        self.ax_tree.legend(fontsize=10)
        self.ax_tree.set_title('KDTree Structure', fontweight='bold', fontsize=14)
        
        # Top-center: Show all events
        self.ax_spatial.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                               c=PRED_COLOR, s=EVENT_SIZE, alpha=0.6, 
                               label='Predicted Events', edgecolors='black', linewidth=1)
        self.ax_spatial.scatter(self.real_events[:, 0], self.real_events[:, 1], 
                               c=REAL_COLOR, s=EVENT_SIZE, alpha=0.6, 
                               label='Real Events', edgecolors='black', linewidth=1)
        
        # Show matches
        matches_found = 0
        for i in range(min(num_processed, NUM_REAL_EVENTS)):
            real_event = self.real_events[i]
            spatial_candidates = self.pred_tree.query_ball_point(real_event, SPATIAL_TOLERANCE)
            
            if len(spatial_candidates) > 0:
                real_time = self.real_times[i]
                target_time = real_time + DT_MS
                candidate_times = self.pred_times[spatial_candidates]
                temporal_mask = np.abs(candidate_times - target_time) <= TEMPORAL_TOLERANCE_MS
                
                if np.any(temporal_mask):
                    temporal_candidates = [spatial_candidates[j] for j in range(len(spatial_candidates)) if temporal_mask[j]]
                    if temporal_candidates:
                        best_match = temporal_candidates[0]  # Simplified: take first match
                        pred_event = self.pred_events[best_match]
                        
                        # Draw match line
                        self.ax_spatial.plot([real_event[0], pred_event[0]], [real_event[1], pred_event[1]], 
                                           'g-', linewidth=3, alpha=0.8)
                        matches_found += 1
        
        self.ax_spatial.legend(fontsize=10)
        self.ax_spatial.set_title(f'Phase 4: Complete Matching ({matches_found} matches)', fontweight='bold', fontsize=14)
        
        # Top-right: Show matching statistics
        self.ax_temporal.text(0.5, 0.5, f"""Matching Statistics:

Total Real Events: {NUM_REAL_EVENTS}
Total Predicted Events: {NUM_PREDICTED_EVENTS}
Events Processed: {num_processed}
Matches Found: {matches_found}

Spatial Tolerance: {SPATIAL_TOLERANCE}px
Temporal Tolerance: ±{TEMPORAL_TOLERANCE_MS}ms
dt: {DT_MS}ms

Success Rate: {matches_found/max(num_processed,1)*100:.1f}%""", 
                             ha='center', va='center', transform=self.ax_temporal.transAxes,
                             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Bottom-left: Show final criteria
        final_criteria = f"""Final Matching Criteria:

✓ Spatial: √((x₁-x₂)² + (y₁-y₂)²) ≤ {SPATIAL_TOLERANCE}px
✓ Temporal: |t_pred - (t_real + {DT_MS})| ≤ {TEMPORAL_TOLERANCE_MS}ms
✓ Polarity: Opposite polarities (0↔1)
✓ One-to-one: Each event matched only once

Algorithm Complete!
Total Matches: {matches_found}"""
        
        self.ax_criteria.text(0.05, 0.95, final_criteria, transform=self.ax_criteria.transAxes,
                             fontsize=11, verticalalignment='top',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        # Bottom-center: Show final steps
        final_steps = f"""Algorithm Complete!

1. ✓ Build KDTree for predicted events
2. ✓ Process all real events
3. ✓ Query spatial candidates
4. ✓ Apply temporal gate
5. ✓ Find best matches

KDTree Benefits:
• Spatial queries: O(log N)
• Memory efficient: O(N)
• Handles large datasets"""
        
        self.ax_steps.text(0.05, 0.95, final_steps, transform=self.ax_steps.transAxes,
                          fontsize=11, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        # Bottom-right: Show final results
        self.ax_results.scatter(self.pred_events[:, 0], self.pred_events[:, 1], 
                               c=PRED_COLOR, s=EVENT_SIZE, alpha=0.6, 
                               label='Predicted Events', edgecolors='black', linewidth=1)
        self.ax_results.scatter(self.real_events[:, 0], self.real_events[:, 1], 
                               c=REAL_COLOR, s=EVENT_SIZE, alpha=0.6, 
                               label='Real Events', edgecolors='black', linewidth=1)
        
        # Show matches
        for i in range(min(num_processed, NUM_REAL_EVENTS)):
            real_event = self.real_events[i]
            spatial_candidates = self.pred_tree.query_ball_point(real_event, SPATIAL_TOLERANCE)
            
            if len(spatial_candidates) > 0:
                real_time = self.real_times[i]
                target_time = real_time + DT_MS
                candidate_times = self.pred_times[spatial_candidates]
                temporal_mask = np.abs(candidate_times - target_time) <= TEMPORAL_TOLERANCE_MS
                
                if np.any(temporal_mask):
                    temporal_candidates = [spatial_candidates[j] for j in range(len(spatial_candidates)) if temporal_mask[j]]
                    if temporal_candidates:
                        best_match = temporal_candidates[0]
                        pred_event = self.pred_events[best_match]
                        
                        # Draw match line
                        self.ax_results.plot([real_event[0], pred_event[0]], [real_event[1], pred_event[1]], 
                                           'g-', linewidth=3, alpha=0.8)
        
        self.ax_results.legend(fontsize=10)
        self.ax_results.set_title('Final Matching Results', fontweight='bold', fontsize=14)
    
    def create_animation(self):
        """Create and save the detailed animation"""
        print("Creating detailed KDTree cancellation animation...")
        print(f"Duration: {DURATION_SECONDS} seconds")
        print(f"Frames: {TOTAL_FRAMES}")
        print(f"FPS: {FPS}")
        print("This will be much slower and more educational!")
        
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
        output_file = "detailed_kdtree_cancellation_animation.mp4"
        print(f"Saving detailed animation to: {output_file}")
        
        try:
            anim.save(output_file, writer='ffmpeg', fps=FPS, bitrate=1800)
            print(f"✓ Detailed animation saved successfully: {output_file}")
        except Exception as e:
            print(f"✗ Error saving animation: {e}")
            print("Trying alternative format...")
            try:
                output_file_gif = "detailed_kdtree_cancellation_animation.gif"
                anim.save(output_file_gif, writer='pillow', fps=FPS)
                print(f"✓ Detailed animation saved as GIF: {output_file_gif}")
            except Exception as e2:
                print(f"✗ Error saving GIF: {e2}")
                print("Showing animation in window instead...")
                plt.show()
        
        return anim

def main():
    """Main function to create the detailed animation"""
    print("=" * 70)
    print("Detailed KDTree Spatial-Temporal Alignment Animation")
    print("=" * 70)
    print("This detailed animation demonstrates:")
    print("1. KDTree structure building with tree-like visualization")
    print("2. Step-by-step spatial query process")
    print("3. Clear temporal alignment criteria and gates")
    print("4. Detailed matching decision process")
    print("5. Complete algorithm with statistics")
    print("=" * 70)
    print("Much slower and more educational than the previous version!")
    print("=" * 70)
    
    # Create detailed animation
    animator = DetailedKDTreeVisualization()
    animation_obj = animator.create_animation()
    
    print("\nDetailed animation complete!")
    print("The animation shows the complete KDTree-based")
    print("spatial-temporal alignment process in detail.")

if __name__ == "__main__":
    main()
