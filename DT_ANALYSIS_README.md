# DT Analysis Pipeline

This pipeline analyzes the relationship between prediction time step (dt) and cancellation rates within a circular region of interest (ROI).

## Overview

The analysis consists of two main steps:

1. **Generate Window Predictions** (`generate_window_predictions.py`)
   - Creates predictions for specific time windows with different dt values (1ms to 20ms)
   - Only processes the time windows used in visualization (5.000-5.010s, 8.200-8.210s, 9.000-9.010s)
   - Saves predictions in organized folders to avoid memory issues

2. **Analyze Cancellation Rates** (`analyze_dt_cancellation.py`)
   - Loads the window predictions and calculates cancellation rates for each dt value
   - Focuses specifically on the circular ROI (center: 665.27, 337.47, radius: 250px)
   - Generates plots and CSV data showing dt vs cancellation rate

## Quick Start

### Option 1: Run Complete DT Analysis Pipeline
```bash
python run_dt_analysis.py
```

### Option 2: Run Tolerance Analysis (requires existing predictions)
```bash
python run_tolerance_analysis.py
```

### Option 3: Run Steps Individually
```bash
# Step 1: Generate predictions
python generate_window_predictions.py

# Step 2: Analyze cancellation rates for different dt values
python analyze_dt_cancellation.py

# Step 3: Analyze tolerance combinations (optional)
python analyze_tolerance_combinations.py

# Step 4: Comprehensive analysis of both dt and tolerances (optional)
python analyze_dt_and_tolerance.py
```

## Configuration

### Time Windows
The analysis focuses on three specific time windows:
- 5.000s to 5.010s
- 8.200s to 8.210s  
- 9.000s to 9.010s

### DT Values
Tests dt values from 1ms to 20ms in 1ms steps (20 total values).

### ROI Parameters
- Center: (665.27, 337.47) pixels
- Radius: 250 pixels
- Scale factor: 1.05 (slightly larger than exact radius)

### Cancellation Parameters
- Temporal tolerance: 5.0ms
- Spatial tolerance: 2.0 pixels
- Polarity mode: "opposite" (real and predicted events must have opposite polarities)

## Output Structure

```
window_predictions/
├── window_1_5.000s_to_5.010s/
│   ├── combined_events_dt_01ms.npy
│   ├── combined_events_dt_02ms.npy
│   └── ... (up to dt_20ms.npy)
├── window_2_8.200s_to_8.210s/
│   └── ... (same structure)
└── window_3_9.000s_to_9.010s/
    └── ... (same structure)

dt_analysis_results/
├── dt_cancellation_analysis.png  # Main plot showing dt vs cancellation rate
└── dt_cancellation_data.csv      # Raw data in CSV format
```

## Memory Efficiency

This approach is memory-efficient because:
- Only processes small time windows (10ms each) instead of the full dataset
- Generates predictions only for the specific windows being analyzed
- Saves intermediate results to disk to avoid keeping everything in memory
- Focuses analysis on the circular ROI to reduce computational load

## Additional Analysis Scripts

### 3. **Analyze Tolerance Combinations** (`analyze_tolerance_combinations.py`)
   - Tests different spatial and temporal tolerance combinations
   - Uses existing window prediction data (much faster)
   - Creates heatmaps showing optimal tolerance parameters
   - Focuses on a specific dt value (configurable)

### 4. **Comprehensive Analysis** (`analyze_dt_and_tolerance.py`)
   - Tests ALL combinations of dt values AND tolerance parameters
   - Most comprehensive but slowest analysis
   - Creates 3D surface plots and comprehensive visualizations
   - Finds optimal parameters across all dimensions

## Expected Results

The analysis will show:
- How cancellation rate varies with dt values
- Which dt values give the best cancellation performance
- Optimal spatial and temporal tolerance combinations
- Differences between time windows
- Event counts within the ROI for each parameter combination
- Heatmaps and 3D visualizations of parameter space

## Troubleshooting

1. **File not found errors**: Ensure the input files exist:
   - Real events: `/home/sumit/anu_research/recording/new_data/perlin_1280hz_hand_outframe.csv`
   - Tracker data: `/home/sumit/anu_research/ego_motion/results_csv/perlin_1280hz_hand_outframe_combined.csv`

2. **Memory issues**: The scripts are designed to be memory-efficient, but if you still encounter issues, you can:
   - Reduce the number of time windows in the `WINDOWS` list
   - Reduce the dt range in `DT_RANGE_MS`
   - Increase the step size in `DT_STEP_MS`

3. **No events in window**: If a time window has no events, it will be skipped with a warning message.

## Customization

To modify the analysis:

1. **Change time windows**: Edit the `WINDOWS` list in both scripts
2. **Change dt range**: Modify `DT_RANGE_MS` and `DT_STEP_MS` in both scripts
3. **Change ROI**: Update `DISC_CENTER_X`, `DISC_CENTER_Y`, and `DISC_RADIUS` in both scripts
4. **Change cancellation parameters**: Modify `BIN_MS`, `R_PIX`, and `POLARITY_MODE` in the analysis script
