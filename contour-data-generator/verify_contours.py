#!/usr/bin/env python3
"""
Validate generated contour vertices against the reference binary files.

This script walks a list of image/reference pairs, regenerates contours from the
source image, and compares them against the expected .bin files under the
contours/ directory.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


DEFAULT_TASKS: List[dict] = [
    {
        "image": "10x10.fits",
        "reference": "10x10_contours",
        "levels": [-1, 0, 1],
        "smoothing": "none",
        "smoothing_factor": 0.0,
    },
    # {
    #     "image": "500x500.fits",
    #     "reference": "500x500_contours",
    #     "levels": [-1, 0, 1],
    #     "smoothing": "none",
    #     "smoothing_factor": 0.0,
    # },
    # {
    #     "image": "500x500.fits",
    #     "reference": "500x500_block_contours",
    #     "levels": [-1, 0, 1],
    #     "smoothing": "block",
    #     "smoothing_factor": 4.0,
    # },
    # {
    #     "image": "500x500.fits",
    #     "reference": "500x500_gaussian_contours",
    #     "levels": [-1, 0, 1],
    #     "smoothing": "gaussian",
    #     "smoothing_factor": 4.0,
    # },
    # {
    #     "image": "500x500_nans.fits",
    #     "reference": "500x500_nans_contours",
    #     "levels": [-1, 0, 1],
    #     "smoothing": "none",
    #     "smoothing_factor": 0.0,
    # },
    # {
    #     "image": "500x500_nans.fits",
    #     "reference": "500x500_nans_block_contours",
    #     "levels": [-1, 0, 1],
    #     "smoothing": "block",
    #     "smoothing_factor": 4.0,
    # },
    # {
    #     "image": "500x500_nans.fits",
    #     "reference": "500x500_nans_gaussian_contours",
    #     "levels": [-1, 0, 1],
    #     "smoothing": "gaussian",
    #     "smoothing_factor": 4.0,
    # },
]


def load_generator_module(script_dir: Path):
    generator_path = script_dir / "contour-generator.py"
    if not generator_path.exists():
        raise FileNotFoundError(f"contour-generator.py not found at {generator_path}")

    spec = importlib.util.spec_from_file_location("contour_generator", generator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load contour generator from {generator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_contours(
    module,
    image_path: Path,
    reference_dir: Path,
    levels: List[float],
    smoothing_mode: str = "none",
    smoothing_factor: float = 0.0,
    debug: bool = False,
) -> Tuple[int, int]:
    """Compare generated contours against reference binary files."""
    image, _ = module.load_image(str(image_path))

    if smoothing_mode != "none":
        image = module.apply_smoothing(image, smoothing_mode, smoothing_factor)

    passed = 0
    failed = 0

    for level in levels:
        vertices, _ = module.generate_contours(image, smoothing_mode, smoothing_factor, level)

        expected_path = reference_dir / f"level_{int(level)}.bin"
        if not expected_path.exists():
            print(f"❌ Missing reference file: {expected_path}")
            failed += 1
            continue

        actual = np.array(vertices, dtype=np.float32).reshape(-1, 2)
        expected = module.read_contour_binary(str(expected_path))

        if actual.shape != expected.shape:
            print(
                f"❌ Level {level}: shape mismatch. generated={actual.shape}, expected={expected.shape}"
            )
            failed += 1
            continue

        if np.allclose(actual, expected, atol=0.1, rtol=0.1):
            print(f"✅ Level {level}: matched {len(actual)} vertices")
            passed += 1
        else:
            max_diff = np.max(np.abs(actual - expected))
            print(f"❌ Level {level}: values differ (max diff={max_diff:.6g})")
            if debug:
                print("   Generated first 8 vertices:")
                print(actual[:8].tolist())
                print("   Reference first 8 vertices:")
                print(expected[:8].tolist())

                # mismatched_actual = actual[actual != expected]
                # mismatched_expected = expected[actual != expected]
                
                # print(f"actual: {mismatched_actual} , expected: {mismatched_expected}")
                    
                for i in range(len(actual)):
                    print(f"expected: {expected[i]}")
                    # print(f"actual: {actual[i]} , expected: {expected[i]}")
            failed += 1

    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generated contour vertices against reference binary files.")
    parser.add_argument("--debug", action="store_true", help="Print sample generated and reference vertices for mismatches")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    image_dir = script_dir / "contour-images"

    module = load_generator_module(script_dir)
    tasks = DEFAULT_TASKS

    print("🔎 Verifying contour generation against reference binary files...")
    print(f"📁 Image directory: {image_dir}")
    print(f"📁 Reference directory: {repo_root / 'contours'}")
    print("")

    total_passed = 0
    total_failed = 0

    for task in tasks:
        image_file = task["image"]
        reference_folder = task["reference"]
        image_path = image_dir / image_file
        reference_dir = repo_root / "contours" / reference_folder

        if not image_path.exists():
            print(f"❌ Image not found: {image_path}")
            total_failed += len(task["levels"])
            continue

        if not reference_dir.exists():
            print(f"❌ Reference directory not found: {reference_dir}")
            total_failed += len(task["levels"])
            continue

        print(f"📊 Checking {image_file} -> {reference_folder}")
        passed, failed = validate_contours(
            module,
            image_path,
            reference_dir,
            levels=task["levels"],
            smoothing_mode=task["smoothing"],
            smoothing_factor=task["smoothing_factor"],
            debug=args.debug,
        )
        total_passed += passed
        total_failed += failed
        print("")

    print("=" * 70)
    print("📈 Verification Summary")
    print("=" * 70)
    print(f"✅ Passed level checks: {total_passed}")
    print(f"❌ Failed level checks: {total_failed}")

    if total_failed == 0:
        print("🎉 All contour comparisons matched the reference binary files.")
        return 0

    print("⚠️  Some contour comparisons failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
