#!/usr/bin/env python3
"""
Test script for individual image enhancement models.

This script allows testing a single enhancer on a test image,
outputting timing information and visual comparison.

Usage:
    python test_single_enhancer.py --enhancement edsr --image-path <path_to_image>
    python test_single_enhancer.py --enhancement lapsrn --image-path <path_to_image>
    python test_single_enhancer.py --enhancement swiniR --image-path <path_to_image>
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from src.image_preprocessing import get_enhancer


def load_test_image(image_path):
    """Load an image for testing."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def save_comparison(original, enhanced, output_path):
    """Save side-by-side comparison of original and enhanced images."""
    # Resize to same height for comparison
    h = min(original.shape[0], enhanced.shape[0])
    original_resized = cv2.resize(original, (original.shape[1] * h // original.shape[0], h))
    enhanced_resized = cv2.resize(enhanced, (enhanced.shape[1] * h // enhanced.shape[0], h))
    
    # Create side-by-side comparison
    comparison = np.hstack([original_resized, enhanced_resized])
    
    # Add labels
    cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Enhanced", (original_resized.shape[1] + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, comparison)
    print(f"Comparison saved to: {output_path}")


def test_enhancer(enhancement_method, image_path, output_dir="outputs/enhancement_tests"):
    """Test a single enhancement method."""
    
    # Load image
    print(f"\n{'='*60}")
    print(f"Testing Enhancement Method: {enhancement_method.upper()}")
    print(f"{'='*60}")
    
    print(f"Loading image from: {image_path}")
    original = load_test_image(image_path)
    print(f"Original image shape: {original.shape}")
    
    # Create enhancer
    print(f"Initializing {enhancement_method} enhancer...")
    try:
        enhancer = get_enhancer(enhancement_method, scale_factor=2.0)
        print(f"✓ Enhancer initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing enhancer: {e}")
        return False
    
    # Enhance image
    print(f"Processing image...")
    start_time = time.time()
    try:
        enhanced = enhancer.enhance(original)
        elapsed_time = time.time() - start_time
        print(f"✓ Enhancement completed in {elapsed_time:.2f} seconds")
    except Exception as e:
        print(f"✗ Error during enhancement: {e}")
        return False
    
    print(f"Enhanced image shape: {enhanced.shape}")
    
    # Calculate metrics
    scale_factor = enhanced.shape[0] / original.shape[0]
    pixels_per_second = (enhanced.shape[0] * enhanced.shape[1]) / elapsed_time
    
    print(f"\nMetrics:")
    print(f"  Scale factor: {scale_factor:.2f}x")
    print(f"  Processing speed: {pixels_per_second:,.0f} pixels/sec")
    print(f"  Output size: {enhanced.shape}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enhanced_path = output_dir / f"{enhancement_method}_enhanced.png"
    cv2.imwrite(str(enhanced_path), enhanced)
    print(f"\nEnhanced image saved to: {enhanced_path}")
    
    # Save comparison
    comparison_path = output_dir / f"{enhancement_method}_comparison.png"
    save_comparison(original, enhanced, str(comparison_path))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test individual image enhancement models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_single_enhancer.py --enhancement edsr --image-path test_image.jpg
  python test_single_enhancer.py --enhancement swiniR --image-path test_image.jpg
  python test_single_enhancer.py --enhancement diffusion --image-path test_image.jpg
        """
    )
    
    parser.add_argument("--enhancement", type=str, required=True,
                       choices=["none", "unsharp_mask", "clahe", "esrgan", "edsr", 
                               "lapsrn", "realcugan", "bsrgan", "swiniR", "diffusion"],
                       help="Enhancement method to test")
    
    parser.add_argument("--image-path", type=str, required=True,
                       help="Path to the test image")
    
    parser.add_argument("--output-dir", type=str, default="outputs/enhancement_tests",
                       help="Directory to save test results")
    
    args = parser.parse_args()
    
    # Validate image path
    if not Path(args.image_path).exists():
        print(f"Error: Image file not found at {args.image_path}")
        return False
    
    # Run test
    success = test_enhancer(args.enhancement, args.image_path, args.output_dir)
    
    if success:
        print(f"\n{'='*60}")
        print(f"✓ Test completed successfully!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"✗ Test failed")
        print(f"{'='*60}")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
