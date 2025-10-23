#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

def load_theta_data(filepath):
    """Load theta data from AEB tracker CSV file."""
    print(f"Loading data from: {filepath}")
    
    try:
        # The data has a complex format, let's parse it manually
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        print(f"Total lines in file: {len(lines)}")
        
        # Parse the data manually since it has irregular formatting
        parsed_data = []
        line_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Split by comma and clean up
            parts = [part.strip() for part in line.split(',')]
            
            # Extract numeric values from each part
            numeric_parts = []
            for part in parts:
                # Split by whitespace to handle multiple values in one part
                sub_parts = part.split()
                for sub_part in sub_parts:
                    try:
                        numeric_parts.append(float(sub_part))
                    except ValueError:
                        continue
            
            # We need at least 11 columns: [ts, x, y, vx, vy, L1, L2, theta, theta_dot, D1, D2]
            if len(numeric_parts) >= 11:
                parsed_data.append(numeric_parts[:11])
                line_count += 1
                
                # Progress indicator for large files
                if line_count % 10000 == 0:
                    print(f"  Processed {line_count} valid data lines...")
        
        if parsed_data:
            data = np.array(parsed_data)
            print(f"Successfully parsed {len(data)} data lines")
            print(f"Data shape: {data.shape}")
            
            # Verify the data structure
            print(f"Columns: timestamp, x, y, vx, vy, L1, L2, theta, theta_dot, D1, D2")
            print(f"First row sample: {data[0]}")
            print(f"Last row sample: {data[-1]}")
            
            return data
        else:
            print("Could not parse any valid data from the file")
            return None
            
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Trying alternative parsing method...")
        
        # Alternative method: try to read as space-separated values
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Replace commas with spaces and split
            content = content.replace(',', ' ')
            values = content.split()
            
            # Convert to float and reshape
            numeric_values = []
            for val in values:
                try:
                    numeric_values.append(float(val))
                except ValueError:
                    continue
            
            # Reshape into rows of 11 columns
            if len(numeric_values) >= 11:
                num_rows = len(numeric_values) // 11
                data = np.array(numeric_values[:num_rows * 11]).reshape(num_rows, 11)
                print(f"Alternative parsing successful: {data.shape}")
                return data
            else:
                print("Alternative parsing failed: insufficient data")
                return None
                
        except Exception as e2:
            print(f"Alternative parsing also failed: {e2}")
            return None

def analyze_omega_from_theta(data, filename):
    """Analyze omega (angular velocity) from theta_dot values over time."""
    if data is None or len(data) == 0:
        print("No data to analyze")
        return
    
    # Extract columns: [ts, x, y, vx, vy, L1, L2, theta, theta_dot, D1, D2]
    timestamps = data[:, 0]
    theta_values = data[:, 7]  # theta column (angular position)
    omega_values = data[:, 8]  # theta_dot column (angular velocity = omega)
    
    print(f"\nOmega (Angular Velocity) Analysis for {filename}:")
    print(f"Total data points: {len(data)}")
    print(f"Time range: {timestamps.min():.6f}s to {timestamps.max():.6f}s")
    print(f"Duration: {timestamps.max() - timestamps.min():.6f}s")
    
    # Convert to degrees for easier interpretation
    theta_deg = np.degrees(theta_values)
    omega_deg_s = np.degrees(omega_values)
    
    print(f"Theta range: {theta_deg.min():.2f}° to {theta_deg.max():.2f}°")
    print(f"Omega range: {omega_deg_s.min():.2f}°/s to {omega_deg_s.max():.2f}°/s")
    
    # Calculate statistics
    omega_mean = np.mean(omega_deg_s)
    omega_std = np.std(omega_deg_s)
    omega_max = np.max(omega_deg_s)
    omega_min = np.min(omega_deg_s)
    
    print(f"\nOmega Statistics:")
    print(f"Mean: {omega_mean:.2f}°/s ± {omega_std:.2f}°/s")
    print(f"Max: {omega_max:.2f}°/s")
    print(f"Min: {omega_min:.2f}°/s")
    print(f"Range: {omega_max - omega_min:.2f}°/s")
    
    # Calculate cumulative rotation
    cumulative_rotation = np.cumsum(omega_deg_s * np.diff(timestamps, prepend=timestamps[0]))
    total_rotation = cumulative_rotation[-1]
    print(f"Total cumulative rotation: {total_rotation:.2f}°")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Omega (Angular Velocity) Analysis: {filename}', fontsize=16)
    
    # Plot 1: Omega over time
    axes[0, 0].plot(timestamps, omega_deg_s, 'r-', linewidth=0.8, alpha=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Omega (degrees/s)')
    axes[0, 0].set_title('Angular Velocity vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Theta over time
    axes[0, 1].plot(timestamps, theta_deg, 'b-', linewidth=0.8, alpha=0.8)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Theta (degrees)')
    axes[0, 1].set_title('Angular Position vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cumulative rotation over time
    axes[0, 2].plot(timestamps, cumulative_rotation, 'g-', linewidth=0.8, alpha=0.8)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Cumulative Rotation (degrees)')
    axes[0, 2].set_title('Cumulative Rotation vs Time')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Omega histogram
    axes[1, 0].hist(omega_deg_s, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Omega (degrees/s)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Omega Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=omega_mean, color='blue', linestyle='--', label=f'Mean: {omega_mean:.1f}°/s')
    axes[1, 0].legend()
    
    # Plot 5: Theta histogram
    axes[1, 1].hist(theta_deg, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].set_xlabel('Theta (degrees)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Theta Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Omega vs Theta (phase plot)
    axes[1, 2].scatter(theta_deg, omega_deg_s, c=timestamps, cmap='viridis', alpha=0.6, s=1)
    axes[1, 2].set_xlabel('Theta (degrees)')
    axes[1, 2].set_ylabel('Omega (degrees/s)')
    axes[1, 2].set_title('Omega vs Theta (Phase Plot)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"omega_analysis_{Path(filename).stem}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    # Show the plot
    plt.show()
    
    return {
        'timestamps': timestamps,
        'theta': theta_values,
        'omega': omega_values,
        'theta_deg': theta_deg,
        'omega_deg_s': omega_deg_s,
        'cumulative_rotation': cumulative_rotation,
        'filename': filename
    }

def main():
    """Main function to analyze omega from a single AEB tracker file."""
    print("Omega (Angular Velocity) Analysis Script for AEB Tracker Data")
    print("=" * 60)
    
    # Check if filename is provided as command line argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not filepath.startswith('AEB_tracker/'):
            filepath = f"AEB_tracker/{filepath}"
    else:
        # Default to first available file
        aeb_files = [f for f in os.listdir('AEB_tracker') if f.endswith('.csv')]
        if not aeb_files:
            print("No AEB tracker CSV files found in AEB_tracker/ directory!")
            return
        
        print("Available files:")
        for i, f in enumerate(aeb_files):
            print(f"  {i+1}. {f}")
        
        # Use the first file as default
        filepath = f"AEB_tracker/{aeb_files[0]}"
        print(f"\nUsing default file: {filepath}")
        print("To analyze a specific file, run: python analyze_theta.py <filename>")
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return
    
    # Analyze the single file
    data = load_theta_data(filepath)
    if data is not None:
        result = analyze_omega_from_theta(data, os.path.basename(filepath))
        if result:
            print(f"\nAnalysis complete for: {result['filename']}")
            print(f"Total rotation tracked: {result['cumulative_rotation'][-1]:.2f}°")

if __name__ == "__main__":
    main() 