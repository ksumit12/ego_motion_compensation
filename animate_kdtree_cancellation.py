#!/usr/bin/env python3
"""
Animated Visualization of KDTree Spatial-Temporal Matching for Ego-Motion Cancellation

This script creates a 5-10 second animation showing:
1. How events are organized in a spatial grid
2. How KDTree queries find spatial neighbors efficiently  
3. How temporal sliding window works
4. How spatial + temporal + polarity constraints are applied
5. How matched events are cancelled/removed

The animation demonstrates the core concepts of the ego-motion cancellation algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches
from scipy.spatial import cKDTree
import time
import os

# Configuration
FIG_SIZE = (16, 10)
DPI = 100
FPS = 10
DURATION_SECONDS = 8
TOTAL_FRAMES = FPS * DURATION_SECONDS

# Event camera parameters (simplified)
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
DISC_CENTER_X = 665
DISC_CENTER_Y = 337
DISC_RADIUS = 264

# Matching parameters
SPATIAL_TOLERANCE = 3.0  # pixels
TEMPORAL_TOLERANCE = 2.0  # milliseconds
DT_ESTIMATE = 2.0  # milliseconds

# Animation parameters
GRID_CELL_SIZE = SPATIAL_TOLERANCE * 2  # Visual grid size
NUM_EVENTS = 200  # Number of events to show
TIME_WINDOW_MS = 50  # Temporal window in milliseconds

class EventData:
    """Simple event data structure"""
    def __init__(self, x, y, polarity, timestamp):
        self.x = x
        self.y = y
        self.polarity = polarity  # +1 or -1
        self.timestamp = timestamp
        self.matched = False
        self.id = id(self)

def generate_synthetic_events(num_events, time_range_ms):
    """Generate synthetic event data for visualization"""
    events = []
    
    # Generate events in a circular pattern (like a spinning disc)
    center_x, center_y = DISC_CENTER_X, DISC_CENTER_Y
    radius_range = (50, DISC_RADIUS - 50)
    
    for i in range(num_events):
        # Random position in circular region
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(*radius_range)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        # Ensure within image bounds
        x = np.clip(x, 0, IMAGE_WIDTH - 1)
        y = np.clip(y, 0, IMAGE_HEIGHT - 1)
        
        # Random polarity
        polarity = np.random.choice([-1, 1])
        
        # Timestamp (events arrive over time)
        timestamp = np.random.uniform(0, time_range_ms)
        
        events.append(EventData(x, y, polarity, timestamp))
    
    return events

def create_spatial_grid(events, cell_size):
    """Create spatial grid visualization"""
    grid_cells = {}
    
    for event in events:
        gx = int(event.x // cell_size)
        gy = int(event.y // cell_size)
        key = (gx, gy)
        
        if key not in grid_cells:
            grid_cells[key] = []
        grid_cells[key].append(event)
    
    return grid_cells

def find_spatial_neighbors_kdtree(events, query_x, query_y, radius):
    """Find spatial neighbors using KDTree"""
    if not events:
        return []
    
    # Build KDTree from event positions
    positions = np.array([[e.x, e.y] for e in events])
    tree = cKDTree(positions)
    
    # Query for neighbors within radius
    neighbor_indices = tree.query_ball_point([query_x, query_y], radius)
    
    return [events[i] for i in neighbor_indices]

def animate_kdtree_cancellation():
    """Create the main animation"""
    
    # Generate synthetic events
    print("Generating synthetic event data...")
    real_events = generate_synthetic_events(NUM_EVENTS, TIME_WINDOW_MS)
    pred_events = generate_synthetic_events(NUM_EVENTS, TIME_WINDOW_MS)
    
    # Sort events by timestamp
    real_events.sort(key=lambda e: e.timestamp)
    pred_events.sort(key=lambda e: e.timestamp)
    
    # Create figure and subplots
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    
    # Main visualization area
    ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax_main.set_xlim(0, IMAGE_WIDTH)
    ax_main.set_ylim(0, IMAGE_HEIGHT)
    ax_main.set_aspect('equal')
    ax_main.set_title('KDTree Spatial-Temporal Matching for Ego-Motion Cancellation', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('X (pixels)')
    ax_main.set_ylabel('Y (pixels)')
    
    # Grid visualization
    ax_grid = plt.subplot2grid((3, 3), (0, 2))
    ax_grid.set_title('Spatial Grid Structure', fontsize=12)
    ax_grid.set_xlabel('Grid X')
    ax_grid.set_ylabel('Grid Y')
    
    # Temporal window visualization
    ax_time = plt.subplot2grid((3, 3), (1, 2))
    ax_time.set_title('Temporal Window', fontsize=12)
    ax_time.set_xlabel('Time (ms)')
    ax_time.set_ylabel('Events')
    
    # Matching statistics
    ax_stats = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax_stats.set_title('Matching Statistics', fontsize=12)
    ax_stats.axis('off')
    
    # Draw ROI circle
    roi_circle = Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, 
                       fill=False, color='red', linewidth=2, linestyle='--', alpha=0.7)
    ax_main.add_patch(roi_circle)
    ax_main.text(DISC_CENTER_X, DISC_CENTER_Y + DISC_RADIUS + 20, 'ROI', 
                ha='center', va='bottom', color='red', fontweight='bold')
    
    # Animation variables
    current_time = 0
    matched_pairs = []
    spatial_grid = create_spatial_grid(pred_events, GRID_CELL_SIZE)
    
    # Text elements for statistics
    stats_text = ax_stats.text(0.1, 0.5, '', fontsize=10, verticalalignment='center',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def animate(frame):
        nonlocal current_time, matched_pairs
        
        # Clear previous frame
        ax_main.clear()
        ax_grid.clear()
        ax_time.clear()
        
        # Set up axes again
        ax_main.set_xlim(0, IMAGE_WIDTH)
        ax_main.set_ylim(0, IMAGE_HEIGHT)
        ax_main.set_aspect('equal')
        ax_main.set_title(f'KDTree Spatial-Temporal Matching (t={current_time:.1f}ms)', fontsize=14, fontweight='bold')
        ax_main.set_xlabel('X (pixels)')
        ax_main.set_ylabel('Y (pixels)')
        
        ax_grid.set_title('Spatial Grid Structure', fontsize=12)
        ax_grid.set_xlabel('Grid X')
        ax_grid.set_ylabel('Grid Y')
        
        ax_time.set_title('Temporal Window', fontsize=12)
        ax_time.set_xlabel('Time (ms)')
        ax_time.set_ylabel('Events')
        
        # Draw ROI circle again
        roi_circle = Circle((DISC_CENTER_X, DISC_CENTER_Y), DISC_RADIUS, 
                           fill=False, color='red', linewidth=2, linestyle='--', alpha=0.7)
        ax_main.add_patch(roi_circle)
        ax_main.text(DISC_CENTER_X, DISC_CENTER_Y + DISC_RADIUS + 20, 'ROI', 
                    ha='center', va='bottom', color='red', fontweight='bold')
        
        # Update current time
        current_time = (frame / TOTAL_FRAMES) * TIME_WINDOW_MS
        
        # Find events in current temporal window
        temporal_window_start = current_time - TEMPORAL_TOLERANCE
        temporal_window_end = current_time + TEMPORAL_TOLERANCE
        
        # Get real events in temporal window
        active_real_events = [e for e in real_events 
                             if temporal_window_start <= e.timestamp <= temporal_window_end]
        
        # Get predicted events in temporal window (shifted by dt)
        active_pred_events = [e for e in pred_events 
                             if temporal_window_start <= (e.timestamp - DT_ESTIMATE) <= temporal_window_end]
        
        # Draw spatial grid
        grid_cells = create_spatial_grid(active_pred_events, GRID_CELL_SIZE)
        
        # Draw grid cells
        for (gx, gy), events_in_cell in grid_cells.items():
            x_min = gx * GRID_CELL_SIZE
            y_min = gy * GRID_CELL_SIZE
            x_max = (gx + 1) * GRID_CELL_SIZE
            y_max = (gy + 1) * GRID_CELL_SIZE
            
            # Draw grid cell
            rect = Rectangle((x_min, y_min), GRID_CELL_SIZE, GRID_CELL_SIZE,
                           fill=False, edgecolor='lightgray', linewidth=0.5, alpha=0.5)
            ax_main.add_patch(rect)
            
            # Draw grid cell in grid subplot
            ax_grid.add_patch(Rectangle((gx, gy), 1, 1, fill=True, 
                                       facecolor='lightblue', alpha=0.3))
            ax_grid.text(gx + 0.5, gy + 0.5, str(len(events_in_cell)), 
                        ha='center', va='center', fontsize=8)
        
        # Draw predicted events
        for event in active_pred_events:
            color = 'blue' if event.polarity > 0 else 'cyan'
            marker = '+' if event.polarity > 0 else 'x'
            ax_main.scatter(event.x, event.y, c=color, marker=marker, s=20, alpha=0.7)
        
        # Draw real events and perform matching
        matches_this_frame = 0
        for real_event in active_real_events:
            if real_event.matched:
                continue
                
            # Draw real event
            color = 'red' if real_event.polarity > 0 else 'orange'
            marker = '+' if real_event.polarity > 0 else 'x'
            ax_main.scatter(real_event.x, real_event.y, c=color, marker=marker, s=30, alpha=0.8)
            
            # Find spatial neighbors using KDTree
            neighbors = find_spatial_neighbors_kdtree(active_pred_events, 
                                                   real_event.x, real_event.y, 
                                                   SPATIAL_TOLERANCE)
            
            # Draw spatial tolerance circle
            tolerance_circle = Circle((real_event.x, real_event.y), SPATIAL_TOLERANCE,
                                    fill=False, color='green', linewidth=1, alpha=0.5)
            ax_main.add_patch(tolerance_circle)
            
            # Check temporal and polarity constraints
            valid_matches = []
            for neighbor in neighbors:
                # Temporal constraint: |t_pred - (t_real + dt)| <= temporal_tolerance
                time_diff = abs(neighbor.timestamp - DT_ESTIMATE - real_event.timestamp)
                temporal_ok = time_diff <= TEMPORAL_TOLERANCE
                
                # Polarity constraint (opposite polarity for cancellation)
                polarity_ok = (neighbor.polarity != real_event.polarity)
                
                if temporal_ok and polarity_ok and not neighbor.matched:
                    valid_matches.append(neighbor)
            
            # Find best match (closest spatially)
            if valid_matches:
                best_match = min(valid_matches, 
                               key=lambda e: np.sqrt((e.x - real_event.x)**2 + (e.y - real_event.y)**2))
                
                # Draw match line
                ax_main.plot([real_event.x, best_match.x], [real_event.y, best_match.y], 
                           'g-', linewidth=2, alpha=0.8)
                
                # Mark as matched
                real_event.matched = True
                best_match.matched = True
                matched_pairs.append((real_event, best_match))
                matches_this_frame += 1
        
        # Draw temporal window visualization
        ax_time.axvline(current_time, color='red', linewidth=2, alpha=0.7, label='Current Time')
        ax_time.axvspan(temporal_window_start, temporal_window_end, alpha=0.2, color='blue', 
                       label=f'Temporal Window (±{TEMPORAL_TOLERANCE}ms)')
        
        # Plot event timestamps
        real_times = [e.timestamp for e in active_real_events]
        pred_times = [e.timestamp - DT_ESTIMATE for e in active_pred_events]
        
        if real_times:
            ax_time.scatter(real_times, [1] * len(real_times), c='red', marker='+', s=20, alpha=0.7, label='Real Events')
        if pred_times:
            ax_time.scatter(pred_times, [0] * len(pred_times), c='blue', marker='o', s=20, alpha=0.7, label='Predicted Events')
        
        ax_time.set_ylim(-0.5, 1.5)
        ax_time.legend(fontsize=8)
        
        # Update statistics
        total_real = len([e for e in real_events if e.timestamp <= current_time])
        total_matched = len(matched_pairs)
        cancellation_rate = (total_matched / total_real * 100) if total_real > 0 else 0
        
        stats_text.set_text(f"""
        Current Time: {current_time:.1f} ms
        Active Real Events: {len(active_real_events)}
        Active Predicted Events: {len(active_pred_events)}
        Matches This Frame: {matches_this_frame}
        Total Matches: {total_matched}
        Cancellation Rate: {cancellation_rate:.1f}%
        
        Spatial Tolerance: {SPATIAL_TOLERANCE} px
        Temporal Tolerance: {TEMPORAL_TOLERANCE} ms
        DT Estimate: {DT_ESTIMATE} ms
        """)
        
        # Set grid limits
        ax_grid.set_xlim(-1, int(IMAGE_WIDTH // GRID_CELL_SIZE) + 1)
        ax_grid.set_ylim(-1, int(IMAGE_HEIGHT // GRID_CELL_SIZE) + 1)
        
        return ax_main, ax_grid, ax_time, stats_text
    
    # Create animation
    print(f"Creating animation with {TOTAL_FRAMES} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=TOTAL_FRAMES, 
                                  interval=1000//FPS, blit=False, repeat=True)
    
    # Save animation
    output_path = "kdtree_cancellation_animation.mp4"
    print(f"Saving animation to {output_path}...")
    
    try:
        anim.save(output_path, writer='ffmpeg', fps=FPS, bitrate=1800)
        print(f"Animation saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying with different writer...")
        try:
            anim.save(output_path.replace('.mp4', '.gif'), writer='pillow', fps=FPS)
            print(f"Animation saved as GIF instead")
        except Exception as e2:
            print(f"Error saving GIF: {e2}")
            print("Animation created but not saved. You can view it interactively.")
    
    # Show the animation
    plt.tight_layout()
    plt.show()
    
    return anim

def create_static_diagram():
    """Create a static diagram showing the KDTree matching process"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Spatial Grid Structure
    ax1.set_title('1. Spatial Grid Organization', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(0, 200)
    
    # Draw grid cells
    cell_size = 40
    for i in range(0, 200, cell_size):
        for j in range(0, 200, cell_size):
            rect = Rectangle((i, j), cell_size, cell_size, fill=False, 
                           edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
    
    # Add some events in different cells
    events = [(30, 30, 1), (70, 30, -1), (110, 70, 1), (150, 150, -1)]
    for x, y, pol in events:
        color = 'red' if pol > 0 else 'blue'
        marker = '+' if pol > 0 else 'x'
        ax1.scatter(x, y, c=color, marker=marker, s=100)
    
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.text(100, 180, 'Events organized in spatial cells\nfor O(1) neighbor lookup', 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. KDTree Query Process
    ax2.set_title('2. KDTree Spatial Query', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 200)
    
    # Draw query point and radius
    query_x, query_y = 100, 100
    ax2.scatter(query_x, query_y, c='green', marker='o', s=200, label='Query Point')
    
    # Draw search radius
    radius = 30
    circle = Circle((query_x, query_y), radius, fill=False, color='green', linewidth=2)
    ax2.add_patch(circle)
    
    # Draw candidate events
    candidates = [(85, 95, 1), (110, 105, -1), (95, 120, 1)]
    for x, y, pol in candidates:
        color = 'red' if pol > 0 else 'blue'
        marker = '+' if pol > 0 else 'x'
        ax2.scatter(x, y, c=color, marker=marker, s=100)
        # Draw line to query point
        ax2.plot([query_x, x], [query_y, y], 'g--', alpha=0.5)
    
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.legend()
    ax2.text(100, 20, 'KDTree finds all events within\nspatial tolerance efficiently', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 3. Temporal Window
    ax3.set_title('3. Temporal Sliding Window', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.set_ylim(-1, 3)
    
    # Draw time axis
    ax3.axhline(y=1, color='black', linewidth=2)
    ax3.axhline(y=2, color='black', linewidth=2)
    
    # Draw events
    real_times = [20, 30, 45, 60, 75]
    pred_times = [22, 32, 47, 62, 77]  # shifted by dt
    
    ax3.scatter(real_times, [1] * len(real_times), c='red', marker='+', s=100, label='Real Events')
    ax3.scatter(pred_times, [2] * len(pred_times), c='blue', marker='o', s=100, label='Predicted Events')
    
    # Draw temporal window
    current_time = 50
    temporal_tol = 10
    ax3.axvline(current_time, color='green', linewidth=2, alpha=0.7, label='Current Time')
    ax3.axvspan(current_time - temporal_tol, current_time + temporal_tol, 
               alpha=0.2, color='green', label=f'Temporal Window (±{temporal_tol}ms)')
    
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Event Stream')
    ax3.set_yticks([1, 2])
    ax3.set_yticklabels(['Real', 'Predicted'])
    ax3.legend()
    ax3.text(50, 0, 'Events must be within temporal tolerance\n|t_pred - (t_real + dt)| ≤ ε_t', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    # 4. Matching Constraints
    ax4.set_title('4. Matching Constraints & Cancellation', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 200)
    ax4.set_ylim(0, 200)
    
    # Draw matched pair
    real_x, real_y = 80, 100
    pred_x, pred_y = 90, 110
    
    ax4.scatter(real_x, real_y, c='red', marker='+', s=200, label='Real Event')
    ax4.scatter(pred_x, pred_y, c='blue', marker='x', s=200, label='Predicted Event')
    
    # Draw match line
    ax4.plot([real_x, pred_x], [real_y, pred_y], 'g-', linewidth=3, label='Match')
    
    # Draw constraints
    ax4.text(140, 150, 'Constraints:', fontsize=12, fontweight='bold')
    ax4.text(140, 130, '✓ Spatial: d ≤ 3px', fontsize=10)
    ax4.text(140, 115, '✓ Temporal: |Δt| ≤ 2ms', fontsize=10)
    ax4.text(140, 100, '✓ Polarity: opposite', fontsize=10)
    ax4.text(140, 85, '✓ Greedy: closest first', fontsize=10)
    
    ax4.text(100, 50, 'Matched events are cancelled\n(removed from residual)', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('kdtree_cancellation_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Static diagram saved as 'kdtree_cancellation_diagram.png'")

if __name__ == "__main__":
    print("KDTree Spatial-Temporal Matching Animation")
    print("=" * 50)
    
    # Create static diagram first
    print("\n1. Creating static diagram...")
    create_static_diagram()
    
    # Create animation
    print("\n2. Creating animation...")
    anim = animate_kdtree_cancellation()
    
    print("\nAnimation complete!")
    print("\nKey Concepts Demonstrated:")
    print("- Spatial Grid: Events organized in cells for O(1) neighbor lookup")
    print("- KDTree Query: Efficient spatial neighbor search within tolerance")
    print("- Temporal Window: Sliding window maintains temporal constraints")
    print("- Matching Constraints: Spatial + Temporal + Polarity filtering")
    print("- Greedy Matching: Closest spatial match wins")
    print("- Cancellation: Matched events removed from residual")

