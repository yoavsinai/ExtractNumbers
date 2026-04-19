"""
Preprocessing Module Validation Tests
"""

import os
import sys
import glob
from pathlib import Path
import cv2
import numpy as np

# Add src to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
sys.path.append(src_dir)

from image_preprocessing.digit_preprocessor import preprocess_digit
from utils.metrics import print_metrics_report

def test_preprocessing(image_path: str, output_dir: str = None) -> bool:
    """
    Test preprocessing module by loading sample image, processing, and saving results.
    """
    if output_dir is None:
        output_dir = os.path.join(root_dir, "outputs", "preprocessing_test_output")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Testing preprocessing on: {image_path}")
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image: {image_path}")
            return False
        
        processed, steps = preprocess_digit(
            img,
            target_size=128,
            return_intermediate=True
        )
        
        # Save output images
        cv2.imwrite(os.path.join(output_dir, "01_original.png"), steps['original'])
        cv2.imwrite(os.path.join(output_dir, "02_upscaled.png"), steps['upscaled'])
        cv2.imwrite(os.path.join(output_dir, "03_denoised.png"), steps['denoised'])
        cv2.imwrite(os.path.join(output_dir, "04_grayscale.png"), steps['grayscale'])
        cv2.imwrite(os.path.join(output_dir, "05_binary.png"), processed)
        
        return True
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False

def main():
    """Run preprocessing tests."""
    print("\n" + "="*60)
    print("Digit Preprocessing Validation Test")
    print("="*60)
    
    search_paths = [
        os.path.join(root_dir, "data", "classification", "single_digits", "0"),
        os.path.join(root_dir, "data", "classification", "single_digits", "1"),
        os.path.join(root_dir, "assets"),
    ]
    
    images = []
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    for path in search_paths:
        if os.path.isdir(path):
            for ext in extensions:
                images.extend(glob.glob(os.path.join(path, f"**/{ext}"), recursive=True))
    
    images = list(set(images))
    if not images:
        print("No sample images found.")
        return 1
    
    if len(images) > 3:
        images = images[:3]
    
    all_results = []
    for img_path in images:
        output_dir = os.path.join(root_dir, "outputs", "preprocessing_test_output", Path(img_path).stem)
        success = test_preprocessing(img_path, output_dir)
        all_results.append(success)
    
    # Use metrics report for the summary
    # Here we treat "True" as the expected success and all_results as predictions
    print_metrics_report([True]*len(all_results), all_results, title="Preprocessing Execution Summary")
    
    return 0 if all(all_results) else 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results = []
        for path in sys.argv[1:]:
            success = test_preprocessing(path)
            results.append(success)
        print_metrics_report([True]*len(results), results, title="Preprocessing Execution Summary")
        sys.exit(0 if all(results) else 1)
    else:
        sys.exit(main())
