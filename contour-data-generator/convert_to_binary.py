#!/usr/bin/env python3
"""
Utility script to convert existing text contour files to binary format.
Usage: python convert_to_binary.py [input_dir] [output_dir]
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path


def convert_text_to_binary(text_file: str, output_file: str = None) -> bool:
    """
    Convert a single text contour file to binary format with C++ header.
    
    Binary format:
    - Header (16 bytes):
      - Magic: 0x4354524E ('CTRN')
      - Version: 1 (uint32, little-endian)
      - Num coords: number of (x,y) pairs (uint32, little-endian)
      - Reserved: 4 bytes
    - Data: float32 array (little-endian) [x1, y1, x2, y2, ...]
    
    Args:
        text_file: Path to text contour file
        output_file: Optional custom output path (default: same dir, .bin extension)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if output_file is None:
            output_file = text_file.replace('.txt', '.bin')
        
        # Read text file - skip empty lines and comments
        with open(text_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not lines:
            print(f"⚠️  Empty file: {text_file}")
            return False
        
        # Parse coordinates
        coords = []
        for line in lines:
            # Handle both space-separated and comma-separated formats
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
            else:
                parts = line.split()
            
            # Extract x, y (first two values)
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append(x)
                    coords.append(y)
                except ValueError:
                    print(f"⚠️  Skipping invalid line in {text_file}: {line}")
                    continue
        
        if not coords:
            print(f"⚠️  No valid coordinates found in {text_file}")
            return False
        
        # Convert to float32 and write binary with header
        data = np.array(coords, dtype=np.float32)
        num_coord_pairs = len(coords) // 2
        
        with open(output_file, 'wb') as f:
            # Write header
            f.write(b'CTRN')  # Magic number (4 bytes)
            f.write(np.uint32(1).tobytes())  # Version 1 (4 bytes, little-endian)
            f.write(np.uint32(num_coord_pairs).tobytes())  # Num coordinate pairs (4 bytes)
            f.write(b'\x00\x00\x00\x00')  # Reserved (4 bytes)
            
            # Write data (little-endian float32)
            f.write(data.astype(np.float32).tobytes())
        
        # Report
        text_size = os.path.getsize(text_file)
        binary_size = os.path.getsize(output_file)
        reduction = (1 - binary_size / text_size) * 100 if text_size > 0 else 0
        
        print(f"✅ {os.path.basename(text_file):40s} → {os.path.basename(output_file):40s} "
              f"({text_size:>10,} → {binary_size:>10,} bytes, {reduction:>5.1f}% reduction)")
        
        return True
    
    except Exception as e:
        print(f"❌ Error converting {text_file}: {e}")
        return False


def batch_convert(input_dir: str, output_dir: str = None, pattern: str = "*.txt") -> None:
    """
    Convert all matching text files in a directory to binary format.
    
    Args:
        input_dir: Directory containing text contour files
        output_dir: Directory for binary output (default: same as input_dir)
        pattern: File pattern to match (default: "*.txt")
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    
    # Find all matching files
    search_path = os.path.join(input_dir, '**', pattern)
    text_files = sorted(glob.glob(search_path, recursive=True))
    
    if not text_files:
        print(f"❌ No files matching '{pattern}' found in {input_dir}")
        return
    
    print(f"📊 Found {len(text_files)} file(s) to convert\n")
    print("-" * 110)
    
    successful = 0
    failed = 0
    
    for text_file in text_files:
        # Preserve directory structure
        rel_path = os.path.relpath(text_file, input_dir)
        output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(output_subdir, exist_ok=True)
        
        output_file = os.path.join(output_subdir, os.path.basename(text_file).replace('.txt', '.bin'))
        
        if convert_text_to_binary(text_file, output_file):
            successful += 1
        else:
            failed += 1
    
    print("-" * 110)
    print(f"\n📈 Conversion Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed:     {failed}")
    print(f"   📦 Total:      {len(text_files)}")


def verify_binary_file(binary_file: str) -> bool:
    """
    Verify a binary contour file is readable and contains valid data.
    
    Args:
        binary_file: Path to binary file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(binary_file, 'rb') as f:
            # Read and validate header
            magic = f.read(4)
            if magic != b'CTRN':
                print(f"❌ Invalid magic number in {binary_file}: expected 'CTRN', got {magic}")
                return False
            
            version_bytes = f.read(4)
            version = np.frombuffer(version_bytes, dtype=np.uint32, count=1)[0]
            if version != 1:
                print(f"❌ Unsupported version in {binary_file}: {version}")
                return False
            
            num_coords_bytes = f.read(4)
            num_coords = np.frombuffer(num_coords_bytes, dtype=np.uint32, count=1)[0]
            
            reserved = f.read(4)
            
            # Read coordinate data
            data = np.frombuffer(f.read(), dtype=np.float32)
        
        if len(data) != num_coords * 2:
            print(f"❌ Data size mismatch in {binary_file}: header says {num_coords} coords "
                  f"({num_coords * 2} floats), but found {len(data)} floats")
            return False
        
        # Check for reasonable coordinate ranges
        coords = data.reshape(-1, 2)
        print(f"✅ {os.path.basename(binary_file):40s} - {len(coords):>6,} vertices, "
              f"X: [{coords[:, 0].min():>8.2f}, {coords[:, 0].max():>8.2f}], "
              f"Y: [{coords[:, 1].min():>8.2f}, {coords[:, 1].max():>8.2f}]")
        
        return True
    
    except Exception as e:
        print(f"❌ Error reading {binary_file}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert text contour files to binary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all .txt files in a directory
  python convert_to_binary.py ./contours
  
  # Convert and save to different directory
  python convert_to_binary.py ./contours ./binary_contours
  
  # Convert specific file
  python convert_to_binary.py ./contours/level_0.txt
  
  # Verify binary files
  python convert_to_binary.py --verify ./binary_contours
        """
    )
    
    parser.add_argument("input_path", help="Input file or directory")
    parser.add_argument("output_path", nargs="?", help="Output directory (default: same as input)")
    parser.add_argument("--verify", action="store_true", help="Verify binary files instead of converting")
    parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path or input_path
    
    if args.verify:
        # Verify mode
        if os.path.isfile(input_path):
            verify_binary_file(input_path)
        else:
            binary_files = sorted(glob.glob(os.path.join(input_path, "**", "*.bin"), recursive=True))
            if binary_files:
                print(f"🔍 Verifying {len(binary_files)} binary file(s)...\n")
                for bf in binary_files:
                    verify_binary_file(bf)
            else:
                print(f"❌ No binary files found in {input_path}")
    else:
        # Convert mode
        if os.path.isfile(input_path):
            # Single file conversion
            print(f"Converting single file: {input_path}")
            convert_text_to_binary(input_path, output_path)
        else:
            # Batch conversion
            batch_convert(input_path, output_path, args.pattern)


if __name__ == "__main__":
    main()
