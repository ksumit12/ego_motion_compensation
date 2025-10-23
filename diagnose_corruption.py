#!/usr/bin/env python3

import numpy as np
import os

def diagnose_numpy_file(filename):
    """Diagnose what's wrong with a numpy file."""
    print(f"DIAGNOSING: {filename}")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"✗ File does not exist: {filename}")
        return
    
    # Check file size
    file_size = os.path.getsize(filename)
    print(f"File size: {file_size:,} bytes ({file_size/1e6:.1f} MB)")
    
    # Try to read file in different ways
    methods = [
        ("Standard numpy.load", lambda f: np.load(f)),
        ("With allow_pickle=True", lambda f: np.load(f, allow_pickle=True)),
        ("As memmap", lambda f: np.load(f, mmap_mode='r')),
        ("As raw bytes", lambda f: np.frombuffer(open(f, 'rb').read(), dtype=np.uint8)),
    ]
    
    for method_name, method_func in methods:
        print(f"\nTrying: {method_name}")
        try:
            data = method_func(filename)
            print(f"✓ Success: shape={getattr(data, 'shape', 'N/A')}, dtype={getattr(data, 'dtype', 'N/A')}")
            
            if hasattr(data, 'shape') and len(data.shape) == 2:
                print(f"  Data preview (first 3 rows):")
                print(f"  {data[:3]}")
            elif hasattr(data, 'shape'):
                print(f"  Raw bytes length: {len(data)}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    # Check file header (numpy files start with specific magic bytes)
    print(f"\nFile header analysis:")
    try:
        with open(filename, 'rb') as f:
            header = f.read(16)
            print(f"First 16 bytes: {header}")
            print(f"As hex: {header.hex()}")
            
            # Check for numpy magic number
            if header.startswith(b'\x93NUMPY'):
                print("✓ Valid numpy magic number")
            else:
                print("✗ Invalid numpy magic number")
                
    except Exception as e:
        print(f"✗ Could not read header: {e}")

def compare_working_vs_broken():
    """Compare working and broken files."""
    print("\nCOMPARING FILES:")
    print("=" * 50)
    
    files_to_check = [
        "predicted_events.npy",
        "combined_events_with_predictions.npy"
    ]
    
    for filename in files_to_check:
        print(f"\n{filename}:")
        if os.path.exists(filename):
            diagnose_numpy_file(filename)
        else:
            print(f"File not found: {filename}")

def main():
    print("NUMPY FILE CORRUPTION DIAGNOSIS")
    print("=" * 60)
    
    # Check the corrupted file
    diagnose_numpy_file("predicted_events.npy")
    
    # Compare with working file
    compare_working_vs_broken()
    
    print(f"\n" + "=" * 60)
    print("LIKELY CAUSES OF CORRUPTION:")
    print("1. Memory mapping issue - file not properly flushed")
    print("2. Interrupted write operation")
    print("3. Disk space issues during write")
    print("4. Process killed before file closure")
    print("5. Memory pressure causing partial writes")
    print("=" * 60)

if __name__ == "__main__":
    main()
