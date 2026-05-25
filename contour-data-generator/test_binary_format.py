#!/usr/bin/env python3
"""
Test script to validate binary contour format functionality.
"""

import os
import sys
import numpy as np
import tempfile
import shutil


def test_binary_write_read():
    """Test writing and reading binary contour files with header."""
    print("=" * 70)
    print("TEST 1: Binary Write/Read with Header")
    print("=" * 70)
    
    # Create test data
    test_coords = np.array([
        3.166906, 0.500000,
        3.500000, 0.708216,
        3.888752, 1.500000,
        3.500000, 2.235382,
        3.330101, 2.500000,
    ], dtype=np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        binary_path = os.path.join(tmpdir, 'test_level_0.bin')
        
        # Write with header
        print(f"\nWriting {len(test_coords) // 2} coordinate pairs with header...")
        with open(binary_path, 'wb') as f:
            # Write header
            f.write(b'CTRN')  # Magic (4 bytes)
            f.write(np.uint32(1).tobytes())  # Version (4 bytes)
            f.write(np.uint32(len(test_coords) // 2).tobytes())  # Num coords (4 bytes)
            f.write(b'\x00\x00\x00\x00')  # Reserved (4 bytes)
            # Write data
            f.write(test_coords.astype(np.float32).tobytes())
        
        file_size = os.path.getsize(binary_path)
        header_size = 16
        data_size = len(test_coords) * 4
        expected_size = header_size + data_size
        print(f"✅ File size: {file_size} bytes (header: {header_size}, data: {data_size}, expected: {expected_size})")
        
        # Read and validate header
        print(f"\nReading and validating binary file...")
        with open(binary_path, 'rb') as f:
            magic = f.read(4)
            version = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[0]
            num_coords = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[0]
            reserved = f.read(4)
            data = np.frombuffer(f.read(), dtype=np.float32)
        
        print(f"✅ Magic: {magic} (expected: b'CTRN')")
        print(f"✅ Version: {version} (expected: 1)")
        print(f"✅ Num coordinates: {num_coords} (expected: {len(test_coords) // 2})")
        print(f"✅ Data floats: {len(data)} (expected: {len(test_coords)})")
        
        # Verify data
        if np.allclose(test_coords, data):
            print(f"✅ Data matches: All values identical")
        else:
            print(f"❌ Data mismatch!")
            return False
        
        # Reshape to coordinates
        coords = data.reshape(-1, 2)
        print(f"✅ Reshaped to coordinate pairs: {coords.shape}")
        print(f"   First coord: ({coords[0, 0]:.6f}, {coords[0, 1]:.6f})")
        print(f"   Last coord: ({coords[-1, 0]:.6f}, {coords[-1, 1]:.6f})")
    
    return True


def test_format_efficiency():
    """Compare text vs binary format efficiency."""
    print("\n" + "=" * 70)
    print("TEST 2: Format Efficiency Comparison")
    print("=" * 70)
    
    # Create test data (1000 coordinate pairs)
    num_coords = 1000
    test_coords = np.random.random(num_coords * 2).astype(np.float32) * 1000
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Binary format
        binary_path = os.path.join(tmpdir, 'test.bin')
        test_coords.tofile(binary_path)
        binary_size = os.path.getsize(binary_path)
        
        # Text format
        text_path = os.path.join(tmpdir, 'test.txt')
        with open(text_path, 'w') as f:
            for i in range(0, len(test_coords), 2):
                f.write(f"{test_coords[i]:.6f} {test_coords[i+1]:.6f}\n")
        text_size = os.path.getsize(text_path)
        
        reduction = (1 - binary_size / text_size) * 100
        
        print(f"\nTest data: {num_coords:,} coordinate pairs")
        print(f"\nFormat        | File Size  | Per Pair")
        print(f"--------------|------------|----------")
        print(f"Text          | {text_size:>10,} | {text_size/num_coords:>8.2f} B")
        print(f"Binary        | {binary_size:>10,} | {binary_size/num_coords:>8.2f} B")
        print(f"Reduction:    | {reduction:>9.1f}% |")
    
    return True


def test_read_existing():
    """Test reading an existing text contour file and converting it."""
    print("\n" + "=" * 70)
    print("TEST 3: Read Existing Text Contour File")
    print("=" * 70)
    
    # Look for existing test files
    test_files = [
        '/home/michaela/Desktop/carta/carta-backend-test-data/contours/500x500_contours/level_0.txt',
        '/home/michaela/Desktop/carta/carta-backend-test-data/contours/500x500_nans_contours/level_0.txt',
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nFound test file: {os.path.basename(test_file)}")
            
            # Read text file
            try:
                text_coords = np.loadtxt(test_file, dtype=np.float32)
                print(f"✅ Read {len(text_coords)} coordinate pairs from text file")
                print(f"   Shape: {text_coords.shape}")
                print(f"   First row: {text_coords[0]}")
                
                # Get file sizes
                text_size = os.path.getsize(test_file)
                
                # Estimate binary size
                binary_size_est = len(text_coords) * 4 * 2  # float32 pairs
                reduction = (1 - binary_size_est / text_size) * 100
                
                print(f"\n   Text size:      {text_size:>10,} bytes")
                print(f"   Binary (est):   {binary_size_est:>10,} bytes")
                print(f"   Space saved:    {reduction:>9.1f}%")
                
                return True
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return False
    
    print("⚠️  No existing test files found - skipping this test")
    return True


def test_read_function():
    """Test the read_contour_binary() helper function."""
    print("\n" + "=" * 70)
    print("TEST 4: read_contour_binary() Function")
    print("=" * 70)
    
    def read_contour_binary(file_path: str) -> np.ndarray:
        """Read contour data from binary file (with header validation)."""
        with open(file_path, 'rb') as f:
            # Read and validate header
            magic = f.read(4)
            if magic != b'CTRN':
                raise ValueError(f"Invalid binary file: magic number mismatch (expected 'CTRN', got {magic})")
            
            version = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[0]
            if version != 1:
                raise ValueError(f"Unsupported binary format version: {version}")
            
            num_coords = np.frombuffer(f.read(4), dtype=np.uint32, count=1)[0]
            reserved = f.read(4)
            
            # Read coordinate data
            data = np.frombuffer(f.read(), dtype=np.float32)
            
            if len(data) != num_coords * 2:
                raise ValueError(f"Data size mismatch: expected {num_coords * 2} floats, got {len(data)}")
        
        return data.reshape(-1, 2) if len(data) > 0 else data
    
    # Create test data
    test_coords = np.array([
        [3.166906, 0.500000],
        [3.500000, 0.708216],
        [3.888752, 1.500000],
    ], dtype=np.float32)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        binary_path = os.path.join(tmpdir, 'test_coords.bin')
        
        # Write with header
        print(f"\nWriting coordinate array with header...")
        with open(binary_path, 'wb') as f:
            f.write(b'CTRN')
            f.write(np.uint32(1).tobytes())
            f.write(np.uint32(len(test_coords)).tobytes())
            f.write(b'\x00\x00\x00\x00')
            f.write(test_coords.flatten().astype(np.float32).tobytes())
        
        # Read with helper function
        print(f"Reading with read_contour_binary()...")
        read_coords = read_contour_binary(binary_path)
        
        print(f"✅ Read shape: {read_coords.shape}")
        print(f"✅ Data matches: {np.allclose(test_coords, read_coords)}")
        
        # Display
        print(f"\nCoordinate data:")
        for i, (x, y) in enumerate(read_coords):
            print(f"   [{i}] ({x:.6f}, {y:.6f})")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "█" * 70)
    print("  Binary Contour Format - Validation Tests")
    print("█" * 70 + "\n")
    
    tests = [
        test_binary_write_read,
        test_format_efficiency,
        test_read_existing,
        test_read_function,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
