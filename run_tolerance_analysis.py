#!/usr/bin/env python3
"""
Runner script for tolerance analysis.
This script runs the tolerance combination analysis using existing window prediction data.
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
    print("Tolerance Analysis Pipeline")
    print("==========================")
    print("This script will analyze different combinations of:")
    print("- Spatial tolerances (0.5 to 5.0 pixels)")
    print("- Temporal tolerances (1.0 to 10.0 ms)")
    print("- Using existing window prediction data")
    print()
    
    # Check if required files exist
    required_files = [
        "analyze_tolerance_combinations.py",
        "analyze_dt_and_tolerance.py"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return 1
    
    # Check if prediction data exists
    if not os.path.exists("./window_predictions"):
        print("Error: Window predictions directory not found.")
        print("Please run generate_window_predictions.py first.")
        return 1
    
    print("Choose analysis type:")
    print("1. Tolerance combinations only (faster)")
    print("2. Comprehensive DT + tolerance analysis (slower)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        success = run_script("analyze_tolerance_combinations.py", 
                           "Analyze tolerance combinations using existing prediction data")
    elif choice == "2":
        success = run_script("analyze_dt_and_tolerance.py", 
                           "Comprehensive analysis of DT values and tolerance combinations")
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        return 1
    
    if not success:
        print("\n❌ Analysis failed.")
        return 1
    
    print(f"\n{'='*60}")
    print("✅ Analysis completed successfully!")
    print("Check the following directories for results:")
    print("  - ./tolerance_analysis_results/     (tolerance combinations only)")
    print("  - ./dt_tolerance_analysis_results/  (comprehensive analysis)")
    print(f"{'='*60}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())




