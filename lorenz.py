import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation

# Multiple chaotic systems with different parameters
systems = [
    # [sigma, rho, beta, dt, x0, y0, z0, colormap]
    [10.0, 28.0, 8.0/3, 0.005, 0.1, 0.1, 0.1, plt.cm.plasma],
    [15.0, 45.0, 3.5, 0.003, 0.2, 0.2, 0.2, plt.cm.inferno],
    [12.0, 35.0, 4.0, 0.004, -0.1, -0.1, -0.1, plt.cm.viridis],
    [18.0, 50.0, 5.0, 0.002, 0.05, -0.05, 0.15, plt.cm.magma],
]

# Animation parameters
animation_speed = 10
update_points = 6
trail_length = 4000
fade_start_ratio = 0.75
fade_curve = 9.0
min_alpha = 0.05

# Static plot parameters
static_iterations = 50000  # Number of iterations for static plot
static_trail_length = 10000  # How many points to keep for static plot

class Enhanced3DChaoticLorenz:
    def __init__(self, mode='animation'):
        """
        Initialize the Lorenz system
        
        Parameters:
        mode: 'animation' for live animation, 'static' for complete pattern plot
        """
        self.mode = mode
        
        # Initialize multiple systems
        self.systems_data = []
        for i, (a, b, c, dt, x0, y0, z0, cmap) in enumerate(systems):
            system = {
                'params': (a, b, c, dt),
                'trajectory': {'x': [x0], 'y': [y0], 'z': [z0]},
                'colormap': cmap,
                'line_collection': Line3DCollection([], linewidth=0.8 - i*0.1)
            }
            self.systems_data.append(system)
        
        # Set up the figure with better 3D perspective
        self.fig = plt.figure(figsize=(14, 10), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Enhanced plot limits for better 3D depth
        self.ax.set_xlim(-40, 40)
        self.ax.set_ylim(-50, 50)
        self.ax.set_zlim(-10, 80)
        self.ax.set_axis_off()
        
        # KEY ENHANCEMENT: Better 3D viewing angle
        self.ax.view_init(elev=20, azim=45)
        
        # Animation-specific initialization
        if self.mode == 'animation':
            self.camera_angle = 0
            self.camera_speed = 0.5
            
            # Add all line collections to the plot
            for system in self.systems_data:
                self.ax.add_collection3d(system['line_collection'])
            
            # Add title
            self.ax.text2D(0.02, 0.98, f"Enhanced 3D Multi-Chaos: {len(systems)} Systems (Live Animation)", 
                          transform=self.ax.transAxes, color='white', fontsize=12, 
                          verticalalignment='top')
        
        plt.tight_layout()
    
    def lorenz_step(self, x, y, z, params):
        a, b, c, dt = params
        dx = a * (y - x)
        dy = x * (b - z) - y
        dz = x * y - c * z
        
        z_amplification = 1.2  
        
        return x + dx * dt, y + dy * dt, z + dz * dt * z_amplification
    
    def generate_complete_trajectory(self, system_index):
        """Generate complete trajectory for static plotting"""
        system = self.systems_data[system_index]
        
        print(f"Generating trajectory for system {system_index + 1}/{len(systems)}...")
        
        # Generate full trajectory
        x, y, z = system['trajectory']['x'][0], system['trajectory']['y'][0], system['trajectory']['z'][0]
        
        for i in range(static_iterations):
            # Show progress
            if i % 10000 == 0:
                print(f"  Progress: {i}/{static_iterations} iterations")
            
            x, y, z = self.lorenz_step(x, y, z, system['params'])
            
            # Add periodic perturbations for complexity
            if i % 600 == 0:
                x += np.random.normal(0, 0.5)
                y += np.random.normal(0, 0.5)
                z += np.random.normal(0, 0.8)
            
            # Add spiral effect
            if i % 1000 == 0:
                spiral_factor = 2.0
                x += spiral_factor * np.cos(i * 0.1)
                y += spiral_factor * np.sin(i * 0.1)
            
            system['trajectory']['x'].append(x)
            system['trajectory']['y'].append(y)
            system['trajectory']['z'].append(z)
            
            # Keep only recent points to manage memory
            if len(system['trajectory']['x']) > static_trail_length:
                system['trajectory']['x'] = system['trajectory']['x'][-static_trail_length:]
                system['trajectory']['y'] = system['trajectory']['y'][-static_trail_length:]
                system['trajectory']['z'] = system['trajectory']['z'][-static_trail_length:]
    
    def create_system_colors(self, num_segments, colormap, system_index):
        fade_start_idx = int(num_segments * fade_start_ratio)
        alphas = np.ones(num_segments)
        
        # Keep full brightness for the first part
        alphas[:fade_start_idx] = 1.0
        
        # Apply fade curve
        if fade_start_idx < num_segments:
            fade_length = num_segments - fade_start_idx
            fade_positions = np.linspace(0, 1, fade_length)
            fade_values = (1 - fade_positions) ** fade_curve
            alphas[fade_start_idx:] = fade_values * (1 - min_alpha) + min_alpha
        
        # Dynamic colors based on z-position for depth effect
        colors = colormap(np.linspace(0.3, 1.0, num_segments))
        colors[:, 3] = alphas * (0.9 - system_index * 0.1)
        
        return colors
    
    def plot_static_pattern(self):
        """Generate and plot the complete static pattern"""
        print("Generating complete Lorenz attractor patterns...")
        
        # Generate complete trajectories for all systems
        for i in range(len(self.systems_data)):
            self.generate_complete_trajectory(i)
        
        print("Plotting complete patterns...")
        
        # Plot all systems
        for i, system in enumerate(self.systems_data):
            if len(system['trajectory']['x']) > 1:
                # Create segments
                points = np.array([
                    system['trajectory']['x'], 
                    system['trajectory']['y'], 
                    system['trajectory']['z']
                ]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Create colors
                colors = self.create_system_colors(len(segments), system['colormap'], i)
                
                # Create line collection
                line_collection = Line3DCollection(segments, colors=colors, linewidths=0.8 - i*0.1)
                self.ax.add_collection3d(line_collection)
        
        # Add title
        self.ax.text2D(0.02, 0.98, f"Complete 3D Multi-Chaos Pattern: {len(systems)} Systems", 
                      transform=self.ax.transAxes, color='white', fontsize=12, 
                      verticalalignment='top')
        
        # Add some stats
        total_points = sum(len(system['trajectory']['x']) for system in self.systems_data)
        self.ax.text2D(0.02, 0.02, f"Total Points: {total_points:,}", 
                      transform=self.ax.transAxes, color='white', fontsize=10)
        
        print("Static plot complete! Close the window to exit.")
        plt.show()
    
    def animate(self, frame):
        """Animation function for all systems with enhanced 3D effects"""
        updated_collections = []
        
        # Dynamic camera rotation for better 3D perspective
        self.camera_angle += self.camera_speed
        if frame % 200 == 0:
            self.camera_speed *= -1
        
        # Update camera view for dynamic 3D effect
        elev = 20 + 15 * np.sin(self.camera_angle * 0.01)
        azim = 45 + self.camera_angle * 0.3
        self.ax.view_init(elev=elev, azim=azim)
        
        for i, system in enumerate(self.systems_data):
            # Add new points to this system's trajectory
            for _ in range(update_points):
                traj = system['trajectory']
                x_new, y_new, z_new = self.lorenz_step(
                    traj['x'][-1], traj['y'][-1], traj['z'][-1], 
                    system['params']
                )
                
                # Add periodic perturbations for more complex 3D patterns
                if frame % 600 == 0:
                    x_new += np.random.normal(0, 0.5)
                    y_new += np.random.normal(0, 0.5)
                    z_new += np.random.normal(0, 0.8)
                
                # Add spiral effect for more interesting 3D trajectories
                if frame % 1000 == 0:
                    spiral_factor = 2.0
                    x_new += spiral_factor * np.cos(frame * 0.1)
                    y_new += spiral_factor * np.sin(frame * 0.1)
                
                traj['x'].append(x_new)
                traj['y'].append(y_new)
                traj['z'].append(z_new)
            
            # Keep only the last 'trail_length' points
            if len(system['trajectory']['x']) > trail_length:
                system['trajectory']['x'] = system['trajectory']['x'][-trail_length:]
                system['trajectory']['y'] = system['trajectory']['y'][-trail_length:]
                system['trajectory']['z'] = system['trajectory']['z'][-trail_length:]
            
            # Create segments for this system
            if len(system['trajectory']['x']) > 1:
                points = np.array([
                    system['trajectory']['x'], 
                    system['trajectory']['y'], 
                    system['trajectory']['z']
                ]).T.reshape(-1, 1, 3)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # Depth-based line width variation
                z_values = np.array(system['trajectory']['z'])
                if len(z_values) > 1:
                    z_normalized = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-10)
                    line_widths = 0.3 + z_normalized[-len(segments):] * 1.5
                    system['line_collection'].set_linewidths(line_widths)
                
                # Create colors for this system
                colors = self.create_system_colors(len(segments), system['colormap'], i)
                
                # Update the line collection
                system['line_collection'].set_segments(segments)
                system['line_collection'].set_colors(colors)
                
            updated_collections.append(system['line_collection'])
        
        return updated_collections
    
    def start_animation(self):
        """Start the enhanced 3D chaotic animation"""
        if self.mode != 'animation':
            print("Error: Cannot start animation in static mode")
            return None
            
        anim = animation.FuncAnimation(
            self.fig, self.animate, interval=animation_speed, blit=False, cache_frame_data=False
        )
        plt.show()
        return anim

# USER CONFIGURATION - CHANGE THIS TO SWITCH MODES
MODE = 'animation'  # Change to 'animation' for live animation, 'static' for complete pattern

def main():
    print("Enhanced 3D Lorenz Attractor Visualization")
    print("=" * 50)
    
    if MODE == 'animation':
        print("Mode: Live Animation")
        print("Starting live animation... Close window to exit.")
        animator = Enhanced3DChaoticLorenz(mode='animation')
        anim = animator.start_animation()
        
    elif MODE == 'static':
        print("Mode: Static Complete Pattern")
        print("This will generate the complete pattern (may take a few moments)...")
        animator = Enhanced3DChaoticLorenz(mode='static')
        animator.plot_static_pattern()
        
    else:
        print("Error: Invalid mode. Use 'animation' or 'static'")

if __name__ == "__main__":
    main()

# USAGE INSTRUCTIONS:
# 
# 1. FOR LIVE ANIMATION:
#    - Set MODE = 'animation'
#    - Run the script
#    - Watch the live, evolving patterns with camera rotation
#
# 2. FOR STATIC COMPLETE PATTERN:
#    - Set MODE = 'static'
#    - Run the script
#    - Wait for generation (shows progress)
#    - View the complete, final pattern as a static image
#
# 3. CUSTOMIZATION:
#    - Adjust 'static_iterations' for longer/shorter patterns
#    - Modify 'static_trail_length' for memory management
#    - Change 'systems' parameters for different attractors