#!/usr/bin/env python3
"""
Batch generate all contour combinations from test images.
Generates contours for all smoothing modes and image variants.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

def generate_contours(
    script_dir: str,
    image_path: str,
    smoothing_mode: str,
    smoothing_factor: float = 0.0,
    python_exe: str = "python3",
) -> bool:
    """
    Run contour generator for a single image.
    
    Args:
        script_dir: Directory containing contour-generator.py
        image_path: Path to image file
        smoothing_mode: 'none', 'gaussian', or 'block'
        smoothing_factor: Smoothing factor (ignored if mode is 'none')
        python_exe: Python executable to use
    
    Returns:
        True if successful, False otherwise
    """
    generator = os.path.join(script_dir, "contour-generator.py")
    
    if not os.path.exists(generator):
        print(f"❌ Error: contour-generator.py not found at {generator}")
        return False
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    # Build command
    cmd = [
        python_exe,
        generator,
        image_path,
        "--levels", "-1", "0", "1",
        "--format", "binary",
    ]
    
    # Add smoothing if not 'none'
    if smoothing_mode != "none":
        cmd.extend(["--smoothing", smoothing_mode, str(smoothing_factor)])
    
    # Run command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"   Error: {result.stderr[:200]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Generate all contour combinations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(script_dir, "contour-images")
    
    # Find Python executable (prefer venv)
    repo_root = os.path.dirname(script_dir)
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        python_exe = venv_python
    else:
        python_exe = "python3"
    
    # Verify image directory exists
    if not os.path.isdir(image_dir):
        print(f"❌ Error: contour-images directory not found at {image_dir}")
        sys.exit(1)
    
    print("🚀 Starting batch contour generation...")
    print(f"📁 Image directory: {image_dir}")
    print(f"🔧 Generator: {script_dir}/contour-generator.py")
    print(f"🐍 Python: {python_exe}")
    print("")
    
    # Define tasks: (image_filename, smoothing_mode, smoothing_factor)
    tasks: List[Tuple[str, str, float]] = [
        # 500x500.fits - no smoothing, gaussian, block
        ("500x500.fits", "none", 0.0),
        ("500x500.fits", "gaussian", 1.5),
        ("500x500.fits", "block", 4.0),
        
        # 500x500.hdf5 - no smoothing, gaussian, block
        # ("500x500.hdf5", "none", 0.0),
        # ("500x500.hdf5", "gaussian", 1.5),
        # ("500x500.hdf5", "block", 4.0),
        
        # 500x500_nans.fits - no smoothing, gaussian, block
        ("500x500_nans.fits", "none", 0.0),
        ("500x500_nans.fits", "gaussian", 1.5),
        ("500x500_nans.fits", "block", 4.0),
        
        # 500x500_nans.hdf5 - no smoothing, gaussian, block
        # ("500x500_nans.hdf5", "none", 0.0),
        # ("500x500_nans.hdf5", "gaussian", 1.5),
        # ("500x500_nans.hdf5", "block", 4.0),
    ]
    
    total = len(tasks)
    succeeded = 0
    failed = 0
    
    for idx, (image_file, mode, factor) in enumerate(tasks, 1):
        image_path = os.path.join(image_dir, image_file)
        
        # Display progress
        mode_display = f"{mode}" if mode == "none" else f"{mode} ({factor})"
        print(f"📊 [{idx}/{total}] Processing: {image_file} ({mode_display})")
        
        # Generate contours
        if generate_contours(script_dir, image_path, mode, factor, python_exe):
            print(f"✅ [{idx}/{total}] SUCCESS: {image_file} ({mode})")
            succeeded += 1
        else:
            print(f"❌ [{idx}/{total}] FAILED: {image_file} ({mode})")
            failed += 1
        
        print("")
    
    # Summary
    print("=" * 70)
    print("📈 Batch Generation Summary")
    print("=" * 70)
    print(f"Total tasks:    {total}")
    print(f"✅ Succeeded:   {succeeded}")
    print(f"❌ Failed:      {failed}")
    print("")
    
    if failed == 0:
        print("🎉 All contours generated successfully!")
        return 0
    else:
        print("⚠️  Some tasks failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
