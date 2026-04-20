"""
Visual Test for YOLO Detection + Preprocessing Enhancement Pipeline
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(root_dir, 'src')
sys.path.append(src_dir)

from full_pipelines.single_photo_pipeline import load_yolo_model, run_yolo_on_image
from image_preprocessing.digit_preprocessor import sharpen_digit
from utils.metrics import print_metrics_report

def visualize_enhanced_pipeline(image_path: str, output_dir: str = None, file_index: int = 0):
    """
    Visualize YOLO detection and preprocessing enhancement.
    """
    if output_dir is None:
        output_dir = os.path.join(root_dir, "outputs", "preprocessing_enhanced_test")

    yolo_weights = os.path.join(root_dir, "outputs", "bbox_comparison", "yolo_run", "weights", "best.pt")
    if not os.path.exists(yolo_weights):
        print(f"YOLO weights not found: {yolo_weights}")
        return False

    yolo_model = load_yolo_model(yolo_weights)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, _ = run_yolo_on_image(yolo_model, image_path)

    original_crops = []
    enhanced_crops = []

    if len(bboxes) > 0:
        bboxes = sorted(bboxes, key=lambda b: b[0])
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            original_crops.append(crop)
            enhanced = sharpen_digit(crop, target_size=64)
            enhanced_crops.append(enhanced)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Detection
    ax = axes[0]
    ax.imshow(img_rgb, interpolation='nearest')
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    ax.set_title("1. Detection")
    ax.axis('off')

    # Panel 2: Original Crops
    ax = axes[1]
    if original_crops:
        max_h = max(c.shape[0] for c in original_crops)
        padded = [cv2.copyMakeBorder(c, 0, max_h-c.shape[0], 5, 5, cv2.BORDER_CONSTANT, value=[128,128,128]) for c in original_crops]
        ax.imshow(cv2.cvtColor(np.concatenate(padded, axis=1), cv2.COLOR_BGR2RGB))
    ax.set_title("2. Original Crops")
    ax.axis('off')

    # Panel 3: Enhanced Crops
    ax = axes[2]
    if enhanced_crops:
        max_h = max(c.shape[0] for c in enhanced_crops)
        padded = [cv2.copyMakeBorder(cv2.cvtColor(c, cv2.COLOR_GRAY2RGB), 0, max_h-c.shape[0], 5, 5, cv2.BORDER_CONSTANT, value=[128,128,128]) for c in enhanced_crops]
        ax.imshow(np.concatenate(padded, axis=1))
    ax.set_title("3. Enhanced Crops")
    ax.axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"enhanced_{file_index}_{os.path.splitext(filename)[0]}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True

def main():
    print("\nVisual Enhancement Pipeline Test")
    test_images = []
    search_paths = [os.path.join(root_dir, "assets"), os.path.join(root_dir, "data", "segmentation", "natural")]
    
    for path in search_paths:
        if os.path.exists(path):
            for r, _, fs in os.walk(path):
                for f in fs:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'image.jpg' in f:
                        test_images.append(os.path.join(r, f))

    if not test_images:
        print("No images found.")
        return 1

    test_images = test_images[:2]
    results = []
    for i, img_path in enumerate(test_images):
        success = visualize_enhanced_pipeline(img_path, file_index=i)
        results.append(success)

    print_metrics_report([True]*len(results), results, title="Visual Enhancement Pipeline Summary")
    return 0 if all(results) else 1

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results = [visualize_enhanced_pipeline(p) for p in sys.argv[1:]]
        print_metrics_report([True]*len(results), results, title="Visual Enhancement Pipeline Summary")
        sys.exit(0 if all(results) else 1)
    else:
        main()
