"""
Mini Test for YOLO Detection + Preprocessing Enhancement Pipeline

Tests the sub-pipeline that:
1. Uses YOLO to detect digit bounding boxes
2. Applies preprocessing enhancement to cropped digits
3. Visualizes: original with boxes, original crops, enhanced crops
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from full_pipelines.single_photo_full_pipeline_not_up_to_date import load_yolo_model, run_yolo_on_image
from ImagePreprocessing.digit_preprocessor import preprocess_digit


def visualize_enhanced_pipeline(image_path: str, output_dir: str = None, file_index: int = 0):
    """
    Visualize the YOLO detection + preprocessing enhancement pipeline.

    Creates a 3-panel visualization showing:
    1. Original image with YOLO bounding boxes
    2. Original cropped digits
    3. Enhanced cropped digits (after preprocessing)

    Args:
        image_path: Path to input image
        output_dir: Output directory (defaults to outputs/preprocessing_enhanced_test)
        file_index: Index for output filename
    """
    if output_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_dir, "outputs", "preprocessing_enhanced_test")

    # Load YOLO model
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yolo_weights = os.path.join(base_dir, "outputs", "bbox_comparison", "yolo_run", "weights", "best.pt")

    if not os.path.exists(yolo_weights):
        print(f"YOLO weights not found: {yolo_weights}")
        return False

    yolo_model = load_yolo_model(yolo_weights)

    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO detection
    bboxes, _ = run_yolo_on_image(yolo_model, image_path)

    if len(bboxes) == 0:
        print("No digits detected in image - creating visualization anyway")
        # Still create visualization showing no detections found
        bboxes = []  # Empty list for visualization
        original_crops = []
        enhanced_crops = []
    else:
        print(f"Detected {len(bboxes)} digits")
        # Sort bboxes left to right
        bboxes = sorted(bboxes, key=lambda b: b[0])

        # Extract original and enhanced digit crops
        original_crops = []
        enhanced_crops = []

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)

            # Extract crop
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            original_crops.append(crop)

            # Apply preprocessing enhancement
            enhanced = preprocess_digit(crop, target_size=64)
            enhanced_crops.append(enhanced)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original image with bounding boxes
    ax = axes[0]
    ax.imshow(img_rgb, interpolation='nearest')

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax.set_title("1. Original Image with YOLO Boxes")
    ax.text(0.5, -0.05, f"Digits detected: {len(bboxes)}",
            size=12, ha="center", transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    ax.axis('off')

    # Panel 2: Original cropped digits
    ax = axes[1]
    if original_crops:
        # Pad images to same height for concatenation
        max_h = max(c.shape[0] for c in original_crops) if original_crops else 64
        padded_crops = []

        for crop in original_crops:
            pad_h = max_h - crop.shape[0]
            padded = cv2.copyMakeBorder(crop, 0, pad_h, 5, 5, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            padded_crops.append(padded)

        if padded_crops:
            concatenated = np.concatenate(padded_crops, axis=1)
            ax.imshow(cv2.cvtColor(concatenated, cv2.COLOR_BGR2RGB))
            ax.text(0.5, -0.05, "Original Crops",
                    size=12, ha="center", transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    else:
        ax.text(0.5, 0.5, "No crops", ha='center', va='center')

    ax.set_title("2. Original Digit Crops")
    ax.axis('off')

    # Panel 3: Enhanced cropped digits
    ax = axes[2]
    if enhanced_crops:
        # Enhanced crops are already grayscale binary images
        max_h = max(c.shape[0] for c in enhanced_crops) if enhanced_crops else 64
        padded_crops = []

        for crop in enhanced_crops:
            pad_h = max_h - crop.shape[0]
            # Convert binary to 3-channel for consistent display
            crop_3ch = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            padded = cv2.copyMakeBorder(crop_3ch, 0, pad_h, 5, 5, cv2.BORDER_CONSTANT, value=[128, 128, 128])
            padded_crops.append(padded)

        if padded_crops:
            concatenated = np.concatenate(padded_crops, axis=1)
            ax.imshow(concatenated, cmap='gray')
            ax.text(0.5, -0.05, "Enhanced Crops (Preprocessed)",
                    size=12, ha="center", transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    else:
        ax.text(0.5, 0.5, "No enhanced crops", ha='center', va='center')

    ax.set_title("3. Enhanced Digit Crops")
    ax.axis('off')

    # Overall title
    filename = os.path.basename(image_path)
    fig.suptitle(f"YOLO Detection + Preprocessing Enhancement | {filename}", fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"enhanced_pipeline_{file_index}_{os.path.splitext(filename)[0]}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Enhanced pipeline visualization saved: {output_path}")
    print(f"  - Detected {len(bboxes)} digits")
    print(f"  - Applied preprocessing enhancement to each crop")

    return True


def test_enhanced_pipeline(image_path: str, output_dir: str = None, file_index: int = 0) -> bool:
    """
    Test the YOLO detection + preprocessing enhancement pipeline.

    This test validates that:
    1. YOLO model loads and runs successfully
    2. Preprocessing pipeline works on detected digits (if any)
    3. Visualization is created showing the pipeline results

    The test PASSES if the pipeline runs without errors, regardless of whether
    digits are actually detected. "No digits found" is a valid pipeline result.

    Args:
        image_path: Path to test image
        output_dir: Output directory
        file_index: Index for output filename

    Returns:
        True if pipeline ran successfully (always, unless exceptions occur)
    """
    print(f"\nTesting enhanced pipeline on: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return False

    try:
        success = visualize_enhanced_pipeline(image_path, output_dir, file_index)

        if success:
            print("✓ Enhanced pipeline test PASSED")
            return True
        else:
            print("✗ Enhanced pipeline test FAILED")
            return False

    except Exception as e:
        print(f"Error during enhanced pipeline test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run enhanced pipeline tests on sample images."""
    print("\n" + "="*70)
    print("YOLO Detection + Preprocessing Enhancement Pipeline Test")
    print("="*70)

    # Test on sample images from assets or data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_images = []

    # Look for sample images
    search_paths = [
        os.path.join(base_dir, "assets"),
        os.path.join(base_dir, "data", "segmentation", "natural"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'image.jpg' in file:
                        test_images.append(os.path.join(root, file))

    if not test_images:
        print("No sample images found. Please provide image paths as arguments:")
        print("  python test_enhanced_pipeline.py path/to/image1.jpg path/to/image2.png")
        return

    # Limit to first 2 images for testing
    test_images = test_images[:2]
    print(f"Testing on {len(test_images)} sample image(s)")

    # Run tests
    results = []
    for i, img_path in enumerate(test_images):
        print(f"\n--- Test {i+1}/{len(test_images)} ---")
        success = test_enhanced_pipeline(img_path, file_index=i)
        results.append((img_path, success))

    # Summary
    print("\n" + "="*70)
    print("Enhanced Pipeline Test Summary")
    print("="*70)

    for img_path, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {os.path.basename(img_path)}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\nResults: {passed}/{total} passed")

    if passed == total:
        print("\n✓ All enhanced pipeline tests PASSED!")
        output_dir = os.path.join(base_dir, "outputs", "preprocessing_enhanced_test")
        print(f"Visualizations saved to: {output_dir}")
        return 0
    else:
        print("\n✗ Some enhanced pipeline tests FAILED")
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific images provided as arguments
        print("\n" + "="*70)
        print("YOLO Detection + Preprocessing Enhancement Pipeline Test")
        print("="*70)

        results = []
        for image_path in sys.argv[1:]:
            success = test_enhanced_pipeline(image_path)
            results.append((image_path, success))

        # Summary
        print("\n" + "="*70)
        print("Enhanced Pipeline Test Summary")
        print("="*70)

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