#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# CSV file paths
REAL_CSV = "real_events_5.0s_to_5.005s.csv"
PRED_CSV = "predicted_events_5.0s_to_5.005s.csv"

def load_csv_events(csv_path):
    """Load events from CSV file"""
    print(f"Loading: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df):,} events")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes.tolist()}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found - {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def verify_csv_data(real_df, pred_df):
    """Verify the loaded CSV data"""
    print("\n" + "="*50)
    print("CSV DATA VERIFICATION")
    print("="*50)
    
    if real_df is not None:
        print(f"Real events: {len(real_df):,}")
        print(f"  X range: {real_df['x'].min():.1f} to {real_df['x'].max():.1f}")
        print(f"  Y range: {real_df['y'].min():.1f} to {real_df['y'].max():.1f}")
        print(f"  Time range: {real_df['timestamp'].min():.6f}s to {real_df['timestamp'].max():.6f}s")
        print(f"  Polarity values: {sorted(real_df['polarity'].unique())}")
        print(f"  Event type values: {sorted(real_df['event_type'].unique())}")
    
    if pred_df is not None:
        print(f"Predicted events: {len(pred_df):,}")
        print(f"  X range: {pred_df['x'].min():.1f} to {pred_df['x'].max():.1f}")
        print(f"  Y range: {pred_df['y'].min():.1f} to {pred_df['y'].max():.1f}")
        print(f"  Time range: {pred_df['timestamp'].min():.6f}s to {pred_df['timestamp'].max():.6f}s")
        print(f"  Polarity values: {sorted(pred_df['polarity'].unique())}")
        print(f"  Event type values: {sorted(pred_df['event_type'].unique())}")

def create_comparison_plot(real_df, pred_df):
    """Create side-by-side comparison plot"""
    print("\nCreating comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot real events
    if real_df is not None and len(real_df) > 0:
        scatter1 = ax1.scatter(real_df['x'], real_df['y'], 
                              c=real_df['timestamp'], cmap='Blues', 
                              s=5, alpha=0.6)
        ax1.set_title(f'Real Events - {len(real_df):,} events')
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Timestamp (s)')
        
        # Set axis properties
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, 'No real events data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Real Events - No Data')
    
    # Plot predicted events
    if pred_df is not None and len(pred_df) > 0:
        scatter2 = ax2.scatter(pred_df['x'], pred_df['y'], 
                              c=pred_df['timestamp'], cmap='Reds', 
                              s=5, alpha=0.6)
        ax2.set_title(f'Predicted Events - {len(pred_df):,} events')
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Timestamp (s)')
        
        # Set axis properties
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, 'No predicted events data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Predicted Events - No Data')
    
    # Set consistent axis limits if both datasets exist
    if real_df is not None and pred_df is not None and len(real_df) > 0 and len(pred_df) > 0:
        all_x = np.concatenate([real_df['x'], pred_df['x']])
        all_y = np.concatenate([real_df['y'], pred_df['y']])
        
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        # Add some padding
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        
        ax1.set_xlim(x_min - x_pad, x_max + x_pad)
        ax1.set_ylim(y_min - y_pad, y_max + y_pad)
        ax2.set_xlim(x_min - x_pad, x_max + x_pad)
        ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    
    fig.suptitle('Event Comparison: Real vs Predicted (from CSV files)', fontsize=14)
    plt.tight_layout()
    
    return fig

def analyze_spatial_overlap(real_df, pred_df):
    """Analyze spatial overlap between real and predicted events"""
    if real_df is None or pred_df is None or len(real_df) == 0 or len(pred_df) == 0:
        print("Cannot analyze overlap - missing data")
        return
    
    print("\n" + "="*50)
    print("SPATIAL OVERLAP ANALYSIS")
    print("="*50)
    
    # Calculate spatial statistics
    real_x_center = real_df['x'].mean()
    real_y_center = real_df['y'].mean()
    pred_x_center = pred_df['x'].mean()
    pred_y_center = pred_df['y'].mean()
    
    print(f"Real events center: ({real_x_center:.1f}, {real_y_center:.1f})")
    print(f"Predicted events center: ({pred_x_center:.1f}, {pred_y_center:.1f})")
    
    # Center offset
    center_offset = np.sqrt((real_x_center - pred_x_center)**2 + (real_y_center - pred_y_center)**2)
    print(f"Center offset: {center_offset:.2f} pixels")
    
    # Spatial spread
    real_x_std = real_df['x'].std()
    real_y_std = real_df['y'].std()
    pred_x_std = pred_df['x'].std()
    pred_y_std = pred_df['y'].std()
    
    print(f"Real events spread: X={real_x_std:.1f}, Y={real_y_std:.1f}")
    print(f"Predicted events spread: X={pred_x_std:.1f}, Y={pred_y_std:.1f}")
    
    # Count events in similar regions (within 10 pixels of each other)
    overlap_threshold = 10.0
    overlap_count = 0
    
    for _, real_row in real_df.iterrows():
        real_x, real_y = real_row['x'], real_row['y']
        for _, pred_row in pred_df.iterrows():
            pred_x, pred_y = pred_row['x'], pred_row['y']
            distance = np.sqrt((real_x - pred_x)**2 + (real_y - pred_y)**2)
            if distance <= overlap_threshold:
                overlap_count += 1
                break
    
    overlap_ratio = overlap_count / len(real_df) if len(real_df) > 0 else 0
    print(f"\nSpatial overlap (within {overlap_threshold}px): {overlap_count}/{len(real_df)} ({overlap_ratio:.1%})")

def main():
    print("CSV Event Comparison Script")
    print("="*50)
    
    # Load CSV files
    real_df = load_csv_events(REAL_CSV)
    pred_df = load_csv_events(PRED_CSV)
    
    # Verify data
    verify_csv_data(real_df, pred_df)
    
    # Create comparison plot
    fig = create_comparison_plot(real_df, pred_df)
    
    # Save plot
    output_file = "csv_events_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {output_file}")
    
    # Show plot
    plt.show()
    
    # Analyze spatial overlap
    analyze_spatial_overlap(real_df, pred_df)
    
    print(f"\nComparison complete!")
    print(f"Real events: {len(real_df):,} (from {REAL_CSV})")
    print(f"Predicted events: {len(pred_df):,} (from {PRED_CSV})")

if __name__ == "__main__":
    main()
