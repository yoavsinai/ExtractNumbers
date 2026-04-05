"""
Preprocessing Module Validation Tests

Tests for the digit preprocessing module.
Validates the preprocessing pipeline by loading sample images, processing them,
and saving intermediate outputs for visual inspection.
"""

import os
import sys
import glob
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))

from ImagePreprocessing.digit_preprocessor import preprocess_digit
import cv2
import numpy as np


def test_preprocessing(image_path: str, output_dir: str = None) -> bool:
    """
    Test preprocessing module by loading sample image, processing, and saving results.
    
    Applies the full preprocessing pipeline and saves all intermediate steps for validation.
    
    Args:
        image_path: Path to input image to test
        output_dir: Directory to save test output images (defaults to outputs/preprocessing_test_output)
    
    Returns:
        True if test passed, False otherwise
    
    Example:
        >>> success = test_preprocessing("path/to/digit.png", output_dir="outputs/test_results")
        >>> if success:
        >>>     print("Preprocessing validation passed!")
    """
    # Set default output dir if not provided
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_dir, "outputs", "preprocessing_test_output")
    
    # Validate input
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing preprocessing on: {image_path}")
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image: {image_path}")
            return False
        
        print(f"✓ Loaded image: shape {img.shape}")
        
        # Run preprocessing with intermediate steps
        processed, steps = preprocess_digit(
            img,
            target_size=128,
            return_intermediate=True
        )
        
        print(f"✓ Preprocessing completed successfully")
        print(f"  - Original shape: {steps['original'].shape}")
        print(f"  - Processed shape: {processed.shape}")
        print(f"  - Otsu threshold value: {steps['threshold_value']:.2f}")
        
        # Save output images
        cv2.imwrite(os.path.join(output_dir, "01_original.png"), steps['original'])
        cv2.imwrite(os.path.join(output_dir, "02_upscaled.png"), steps['upscaled'])
        cv2.imwrite(os.path.join(output_dir, "03_denoised.png"), steps['denoised'])
        cv2.imwrite(os.path.join(output_dir, "04_sharpened.png"), steps['sharpened'])
        cv2.imwrite(os.path.join(output_dir, "05_grayscale.png"), steps['grayscale'])
        cv2.imwrite(os.path.join(output_dir, "06_binary.png"), processed)
        
        print(f"✓ Saved test images to: {os.path.abspath(output_dir)}")
        print("\nGenerated files:")
        print("  - 01_original.png")
        print("  - 02_upscaled.png")
        print("  - 03_denoised.png")
        print("  - 04_sharpened.png")
        print("  - 05_grayscale.png")
        print("  - 06_binary.png")
        
        return True
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_image(image_path: str, output_base: str = None):
    """Test preprocessing on a single image."""
    if output_base is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_base = os.path.join(base_dir, "outputs", "preprocessing_test_output")
    
    print("\n" + "="*60)
    print(f"Testing image: {image_path}")
    print("="*60)
    
    output_dir = os.path.join(output_base, Path(image_path).stem)
    success = test_preprocessing(image_path, output_dir)
    
    return success


def find_sample_images(search_paths: list) -> list:
    """Find sample digit images in common locations."""
    images = []
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    for path in search_paths:
        if os.path.isfile(path):
            images.append(path)
        elif os.path.isdir(path):
            for ext in extensions:
                images.extend(glob.glob(os.path.join(path, f"**/{ext}"), recursive=True))
    
    return list(set(images))  # Remove duplicates


def main():
    """Run preprocessing tests."""
    print("\n" + "="*60)
    print("Digit Preprocessing Validation Test")
    print("="*60)
    
    # Default search locations for sample images
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    search_paths = [
        os.path.join(base_dir, "data", "classification", "single_digits", "0"),
        os.path.join(base_dir, "data", "classification", "single_digits", "1"),
        os.path.join(base_dir, "assets"),
    ]
    
    # Find sample images
    images = find_sample_images(search_paths)
    
    if not images:
        print("\nNo sample images found in default locations.")
        print("\nUsage: python test_preprocessing.py [image_path] [image_path] ...")
        print("\nExample:")
        print("  python test_preprocessing.py path/to/digit1.png path/to/digit2.png")
        print(f"\nSearched in:")
        for path in search_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    # Limit to first 3 images if many found
    if len(images) > 3:
        print(f"\nFound {len(images)} images. Testing first 3...")
        images = images[:3]
    else:
        print(f"\nFound {len(images)} sample image(s)")
    
    # Run tests
    results = []
    for img_path in images:
        success = test_single_image(img_path, os.path.join(base_dir, "outputs", "preprocessing_test_output"))
        results.append((img_path, success))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for img_path, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {os.path.basename(img_path)}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} passed")
    
    if passed == total:
        print("\n✓ All preprocessing tests PASSED!")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    # If arguments provided, use them as image paths
    if len(sys.argv) > 1:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_base = os.path.join(base_dir, "outputs", "preprocessing_test_output")
        
        print("\n" + "="*60)
        print("Digit Preprocessing Validation Test")
        print("="*60)
        
        results = []
        for image_path in sys.argv[1:]:
            success = test_single_image(image_path, output_base)
            results.append((image_path, success))
        
        # Summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        
        for img_path, success in results:
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"{status}: {os.path.basename(img_path)}")
        
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        print(f"\nResults: {passed}/{total} passed")
        sys.exit(0 if passed == total else 1)
    
    else:
        exit_code = main()
        sys.exit(exit_code)
