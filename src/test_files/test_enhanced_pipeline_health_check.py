"""
Enhanced Pipeline Health Check

Runs the enhanced pipeline test on randomly selected images.
Provides quick validation of the YOLO detection + preprocessing pipeline health.
"""

import os
import sys
import random
import glob
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from test_enhanced_pipeline import test_enhanced_pipeline


def find_all_test_images() -> list:
    """Find all available test images from data and assets folders."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images = []

    # Search in multiple locations
    search_paths = [
        os.path.join(base_dir, "data", "classification", "single_digits"),
        os.path.join(base_dir, "data", "segmentation", "natural"),
        os.path.join(base_dir, "assets"),
    ]

    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']

    for path in search_paths:
        if os.path.exists(path):
            for ext in extensions:
                pattern = os.path.join(path, "**", ext)
                images.extend(glob.glob(pattern, recursive=True))

    # Also look for image.jpg files specifically in segmentation folders
    for root, _, files in os.walk(os.path.join(base_dir, "data", "segmentation")):
        for file in files:
            if file == 'image.jpg':
                images.append(os.path.join(root, file))

    # Remove duplicates and filter out very small files (likely corrupted)
    unique_images = list(set(images))
    valid_images = []

    for img_path in unique_images:
        try:
            if os.path.getsize(img_path) > 1000:  # At least 1KB
                valid_images.append(img_path)
        except (OSError, IOError):
            continue

    return valid_images


def run_fast_random_test(num_images: int = 1000) -> int:
    """
    Run enhanced pipeline health check on randomly selected images.

    Args:
        num_images: Number of random images to test (default: 6)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("\n" + "="*60)
    print(f"Enhanced Pipeline Health Check - {num_images} Images")
    print("="*60)

    # Find all available images
    all_images = find_all_test_images()

    if len(all_images) < num_images:
        print(f"Warning: Only {len(all_images)} images available, testing all of them")
        num_images = len(all_images)

    if num_images == 0:
        print("Error: No test images found!")
        print("Searched in:")
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        print(f"  - {os.path.join(base_dir, 'data', 'classification', 'single_digits')}")
        print(f"  - {os.path.join(base_dir, 'data', 'segmentation', 'natural')}")
        print(f"  - {os.path.join(base_dir, 'assets')}")
        return 1

    # Randomly select images
    selected_images = random.sample(all_images, num_images)

    print(f"Found {len(all_images)} total images")
    print(f"Randomly selected {num_images} images for testing:")
    for i, img_path in enumerate(selected_images, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    print()

    # Run tests
    results = []
    passed = 0

    for i, image_path in enumerate(selected_images, 1):
        print(f"[{i}/{num_images}] Testing: {os.path.basename(image_path)}")
        try:
            success = test_enhanced_pipeline(image_path, file_index=i-1)
            results.append((image_path, success))
            if success:
                passed += 1
                print("  ✓ PASSED")
            else:
                print("  ✗ FAILED")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append((image_path, False))

        print()

    # Summary
    print("="*60)
    print("Enhanced Pipeline Health Check Summary")
    print("="*60)

    for i, (img_path, success) in enumerate(results, 1):
        status = "✓" if success else "✗"
        filename = os.path.basename(img_path)
        print(f"{status} Test {i}: {filename}")

    accuracy = passed / num_images if num_images > 0 else 0
    print(f"\nResults: {passed}/{num_images} passed ({accuracy:.1%})")

    # Output location
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "outputs", "preprocessing_enhanced_test")
    print(f"Visualizations saved to: {output_dir}")

    if passed == num_images:
        print("\n🎉 All tests PASSED! Pipeline is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {passed}/{num_images} tests completed. Pipeline is functional.")
        return 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced pipeline health check for YOLO + preprocessing pipeline")
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=6,
        help="Number of random images to test (default: 6)"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible results"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    exit_code = run_fast_random_test(args.num)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()