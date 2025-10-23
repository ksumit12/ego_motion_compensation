#!/usr/bin/env python3
"""
Runner script to execute the complete dt analysis pipeline.
This script runs both the prediction generation and cancellation analysis.
"""

import subprocess
import sys
import time
import os

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Completed in {elapsed_time:.1f}s")
        
        if result.stdout:
            print("\nOutput:")
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed after {elapsed_time:.1f}s")
        print(f"Error code: {e.returncode}")
        
        if e.stdout:
            print("\nStdout:")
            print(e.stdout)
        
        if e.stderr:
            print("\nStderr:")
            print(e.stderr)
        
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"✗ Failed after {elapsed_time:.1f}s")
        print(f"Error: {e}")
        return False

def main():
    """Main execution function"""
    print("DT Analysis Pipeline")
    print("===================")
    print("This script will:")
    print("1. Generate predictions for time windows with different dt values")
    print("2. Analyze cancellation rates for each dt value within the circular ROI")
    print()
    
    # Check if required files exist
    required_files = [
        "generate_window_predictions.py",
        "analyze_dt_cancellation.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return 1
    
    # Step 1: Generate window predictions
    success1 = run_script("generate_window_predictions.py", 
                         "Generate predictions for time windows with different dt values")
    
    if not success1:
        print("\n❌ Prediction generation failed. Stopping pipeline.")
        return 1
    
    # Step 2: Analyze cancellation rates
    success2 = run_script("analyze_dt_cancellation.py", 
                         "Analyze cancellation rates for different dt values")
    
    if not success2:
        print("\n❌ Cancellation analysis failed.")
        return 1
    
    print(f"\n{'='*60}")
    print("✅ Pipeline completed successfully!")
    print("Check the following directories for results:")
    print("  - ./window_predictions/     (prediction data)")
    print("  - ./dt_analysis_results/    (analysis results and plots)")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())




