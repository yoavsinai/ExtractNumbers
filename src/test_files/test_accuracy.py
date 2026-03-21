import os
import tempfile
import torch
import torchvision.datasets as datasets
import cv2
import numpy as np

# Import functions from the single photo pipeline
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'full_pipelines'))
from single_photo_full_pipeline import load_yolo_model, load_digit_model, run_yolo_on_image, recognize_digits

def main():
    # Paths to models
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yolo_weights = os.path.join(base_dir, "outputs", "bbox_comparison", "yolo_run", "weights", "best.pt")
    digit_weights = os.path.join(base_dir, "outputs", "bbox_comparison", "digit_classifier.pth")

    if not os.path.exists(yolo_weights):
        print(f"YOLO weights not found: {yolo_weights}")
        return

    if not os.path.exists(digit_weights):
        print(f"Digit classifier weights not found: {digit_weights}")
        return

    # Load models
    print("Loading models...")
    yolo_model = load_yolo_model(yolo_weights)
    digit_model = load_digit_model(digit_weights)

    # Load SVHN test dataset
    print("Loading SVHN test dataset...")
    dataset = datasets.SVHN(root=os.path.join(base_dir, 'data', 'raw'), split='test', download=False)
    total_samples = min(2000, len(dataset))
    print(f"Testing on {total_samples} samples...")

    correct = 0
    for i in range(total_samples):
        img, label = dataset[i]
        true_str = str(label)

        # Save image to temp file for processing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img.save(temp_path)

        try:
            # Run detection
            bboxes, _ = run_yolo_on_image(yolo_model, temp_path)
            if len(bboxes) == 0:
                recognized_str = ""
            else:
                # Sort bboxes left to right
                bboxes = sorted(bboxes, key=lambda b: b[0])
                digits = recognize_digits(digit_model, temp_path, bboxes)
                recognized_str = "".join(str(d[0]) for d in digits)

            if recognized_str == true_str:
                correct += 1

        finally:
            # Clean up temp file
            os.unlink(temp_path)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_samples} samples...")

    accuracy = correct / total_samples
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total_samples})")

if __name__ == "__main__":
    main()