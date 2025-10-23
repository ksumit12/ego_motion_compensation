#!/usr/bin/env python3
"""
Install GPU dependencies for the ego-motion analysis
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("Installing GPU dependencies for ego-motion analysis...")
    
    # Required packages
    packages = [
        "psutil",  # Memory monitoring
        "cupy-cuda12x",  # GPU acceleration (adjust for your CUDA version)
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\nInstallation complete: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("🎉 All dependencies installed! You can now run the analysis with GPU support.")
    else:
        print("⚠️  Some packages failed to install. The analysis will fall back to CPU processing.")

if __name__ == "__main__":
    main()
