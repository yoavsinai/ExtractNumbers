import os
import tempfile
import torch
import torchvision.datasets as datasets
import cv2
import numpy as np

import h5py

# Import functions from the single photo pipeline
import sys

max_samples = 1000

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'full_pipelines'))
from single_photo_full_pipeline import load_yolo_model, load_digit_model, run_yolo_on_image, recognize_digits

# Import metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metrics import print_metrics_report

def get_true_labels_from_mat(mat_file_path):
    """
    Parses the digitStruct.mat file (v7.3) using h5py to get the ground truth sequence of digits.
    """
    try:
        f = h5py.File(mat_file_path, 'r')
    except (FileNotFoundError, OSError):
        print(f"Error: Ground truth file not found or is not a valid HDF5 file at {mat_file_path}")
        print("Please ensure the SVHN dataset is downloaded and in the 'data/raw/svhn' directory.")
        return None

    all_labels = []
    
    # This dataset contains references to the data for each image
    bbox_dataset = f['digitStruct/bbox']

    for i in range(bbox_dataset.shape[0]):
        bbox_ref = bbox_dataset[i, 0]
        bbox_group = f[bbox_ref]
        
        label_data = bbox_group['label']
        
        # The data structure differs for single vs. multi-digit numbers.
        if label_data.shape[0] > 1:
            # For multiple digits, 'label_data' is an array of references.
            labels = [int(f[ref[0]][()][0,0]) for ref in label_data]
        else:
            # For a single digit, 'label_data' just contains the value.
            labels = [int(label_data[0,0])]

        # In the SVHN dataset, the label '10' represents the digit '0'.
        label_str = "".join(str(l % 10) for l in labels)
        all_labels.append(label_str)
        
    f.close()
    return all_labels


def visualize_results(result, output_dir, file_index):
    """
    Visualize the prediction for a single image in three stages.
    Saves an image showing:
    1. The original, untouched input image.
    2. Original image with raw YOLO bounding boxes in red.
    3. Cropped digits sent to the recognizer.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img = result['image']
    digit_images = result.get('digit_images', [])
    digits = result.get('digits', [])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel 1: Original Untouched Image ---
    ax = axes[0]
    ax.imshow(img, interpolation='nearest')
    ax.set_title("1. Original Image")
    ax.axis('off')
    
    # --- Panel 2: Raw YOLO Bounding Box ---
    ax = axes[1]
    ax.imshow(img, interpolation='nearest')
    num_bboxes = len(result.get('bboxes', []))
    for i, bbox in enumerate(result.get('bboxes', [])):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    ax.set_title("2. YOLO Detection")
    ax.text(0.5, -0.05, f"Digits passed to next layer: {num_bboxes}", 
            size=12, ha="center", transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    ax.axis('off')

    # --- Panel 3: Cropped Digits for Recognition ---
    ax = axes[2]
    if digit_images:
        # Pad images to the same height before concatenating
        max_h = max(d.shape[0] for d in digit_images) if digit_images else 28
        padded_digits = []
        for d_img in digit_images:
            pad_h = max_h - d_img.shape[0]
            padded_img = cv2.copyMakeBorder(d_img, 0, pad_h, 5, 5, cv2.BORDER_CONSTANT, value=[128,128,128])
            padded_digits.append(padded_img)

        if padded_digits:
            concatenated_digits = np.concatenate(padded_digits, axis=1)
            ax.imshow(concatenated_digits, cmap='gray')
            pred_str = result.get('pred_str', '')
            ax.text(0.5, -0.05, f"Predicted Number: {pred_str}", 
                    size=12, ha="center", transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
    else:
        ax.text(0.5, 0.5, "No digits found", ha='center', va='center')
    ax.set_title("3. Processed Digits")
    ax.axis('off')
    
    # Add an overall title
    fig.suptitle(f"Image {file_index} | True: '{result['true_str']}' | Pred: '{result['pred_str']}'", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"prediction_{file_index}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

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

    # --- Filter dataset for multi-digit images ---
    print("Loading ground truth labels to find multi-digit images...")
    mat_file_path = os.path.join(base_dir, 'data', 'raw', 'svhn', 'test', 'digitStruct.mat')
    all_true_labels = get_true_labels_from_mat(mat_file_path)
    if all_true_labels is None:
        return # Error handled in helper function

    multi_digit_samples = [
        (i, label) for i, label in enumerate(all_true_labels) if len(label) > 1
    ]
    
    if not multi_digit_samples:
        print("No multi-digit images found in the dataset.")
        return

    # Load models
    print("Loading models...")
    yolo_model = load_yolo_model(yolo_weights)
    digit_model = load_digit_model(digit_weights)

    # Load SVHN test dataset
    print("Loading SVHN test dataset...")
    dataset = datasets.SVHN(root=os.path.join(base_dir, 'data', 'raw'), split='test', download=False)
    
    total_samples_to_test = min(max_samples, len(multi_digit_samples))
    test_samples = multi_digit_samples[:total_samples_to_test]
    
    print(f"Found {len(multi_digit_samples)} multi-digit images. Testing on the first {len(test_samples)}...")

    all_preds = []
    all_true = []
    output_dir = os.path.join(base_dir, "outputs", "fullpipelines_predictions")

    for i, (image_index, true_str) in enumerate(test_samples):
        img, _ = dataset[image_index] # We use our own true_str, so ignore the dataset's label

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img.save(temp_path)

        try:
            # Run detection
            bboxes, _ = run_yolo_on_image(yolo_model, temp_path)
            
            recognized_str = ""
            digits = []

            # --- Visualization data gathering ---
            img_for_vis = cv2.imread(temp_path)
            digit_images_for_vis = []

            if len(bboxes) > 0:
                bboxes = sorted(bboxes, key=lambda b: b[0])
                digits = recognize_digits(digit_model, temp_path, bboxes)
                recognized_str = "".join(str(d[0]) for d in digits)

                for bbox in bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    digit_crop = img_for_vis[y1:y2, x1:x2]
                    if digit_crop.size > 0:
                        resized_digit = cv2.resize(digit_crop, (28, 28))
                        digit_images_for_vis.append(resized_digit)

            all_preds.append(recognized_str)
            all_true.append(true_str)

            if recognized_str == true_str:
                print(f"Correct: Predicted '{recognized_str}', True '{true_str}' (Image index {image_index})")
            else:
                print(f"Mismatch: Predicted '{recognized_str}', True '{true_str}' (Image index {image_index})")
            
            img_np = np.array(img)
            result = {
                'image': img_np,
                'true_str': true_str,
                'pred_str': recognized_str,
                'bboxes': bboxes,
                'digits': digits,
                'digit_images': digit_images_for_vis
            }
            visualize_results(result, output_dir, image_index)

        finally:
            os.unlink(temp_path)

        if (i + 1) % 100 == 0:
            print(f"Processed and visualized {i + 1}/{len(test_samples)} samples...")

    print_metrics_report(all_true, all_preds, title="SVHN Multi-Digit Pipeline Evaluation")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()