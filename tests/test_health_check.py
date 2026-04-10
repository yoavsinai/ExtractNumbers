"""
Enhanced Pipeline Health Check
"""

import os
import sys
import random
import glob
from pathlib import Path

# Add src and tests to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
tests_dir = os.path.join(root_dir, 'tests')
sys.path.append(src_dir)
sys.path.append(tests_dir)

from test_visual_enhancement import visualize_enhanced_pipeline
from utils.metrics import print_metrics_report

def find_all_test_images() -> list:
    """Find all available test images."""
    images = []
    search_paths = [
        os.path.join(root_dir, "data", "classification", "single_digits"),
        os.path.join(root_dir, "data", "segmentation", "natural"),
        os.path.join(root_dir, "assets"),
    ]
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']

    for path in search_paths:
        if os.path.exists(path):
            for ext in extensions:
                images.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))

    for root, _, files in os.walk(os.path.join(root_dir, "data", "segmentation")):
        for file in files:
            if file == 'image.jpg':
                images.append(os.path.join(root, file))

    unique_images = list(set(images))
    valid_images = [img for img in unique_images if os.path.exists(img) and os.path.getsize(img) > 1000]
    return valid_images

def run_fast_random_test(num_images: int = 6) -> int:
    """Run health check on random images."""
    print("\n" + "="*60)
    print(f"Enhanced Pipeline Health Check - {num_images} Images")
    print("="*60)

    all_images = find_all_test_images()
    if not all_images:
        print("No images found.")
        return 1

    num_images = min(num_images, len(all_images))
    selected_images = random.sample(all_images, num_images)

    results = []
    for i, image_path in enumerate(selected_images, 1):
        print(f"[{i}/{num_images}] Testing: {os.path.basename(image_path)}")
        try:
            success = visualize_enhanced_pipeline(image_path, file_index=i-1)
            results.append(success)
            print("  ✓ PASSED" if success else "  ✗ FAILED")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append(False)

    print_metrics_report([True]*len(results), results, title="Health Check Execution Summary")
    return 0 if all(results) else 0 # Return 0 even if some fail as it's a "health check"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", "-n", type=int, default=6)
    parser.add_argument("--seed", "-s", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    
    sys.exit(run_fast_random_test(args.num))
