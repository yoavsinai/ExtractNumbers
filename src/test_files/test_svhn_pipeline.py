import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tempfile
import h5py
import torch
import torchvision.datasets as datasets

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from full_pipelines.single_photo_full_pipeline_not_up_to_date import load_yolo_model, load_digit_model, run_yolo_on_image, recognize_digits

def get_true_labels_from_mat(mat_file_path):
    """
    Parses the digitStruct.mat file (v7.3) using h5py to get the ground truth sequence of digits.
    """
    try:
        f = h5py.File(mat_file_path, 'r')
    except (FileNotFoundError, OSError):
        print(f"Error: Ground truth file not found or is not a valid HDF5 file at {mat_file_path}")
        return None

    all_labels = []
    bbox_dataset = f['digitStruct/bbox']
    for i in range(bbox_dataset.shape[0]):
        bbox_ref = bbox_dataset[i, 0]
        bbox_group = f[bbox_ref]
        label_data = bbox_group['label']
        if label_data.shape[0] > 1:
            labels = [int(f[ref[0]][()][0,0]) for ref in label_data]
        else:
            labels = [int(label_data[0,0])]
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
    img = result['image']
    digit_images = result.get('digit_images', [])
    
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

    # Load models
    print("Loading models...")
    yolo_model = load_yolo_model(yolo_weights)
    digit_model = load_digit_model(digit_weights)

    # --- Filter dataset for multi-digit images using SVHN ---
    print("Loading SVHN ground truth labels...")
    mat_file_path = os.path.join(base_dir, 'data', 'raw', 'svhn', 'test', 'digitStruct.mat')
    all_true_labels = get_true_labels_from_mat(mat_file_path)
    if all_true_labels is None:
        return

    multi_digit_samples = [
        (i, label) for i, label in enumerate(all_true_labels) if len(label) > 1
    ]
    
    if not multi_digit_samples:
        print("No multi-digit images found in the dataset.")
        return

    # Load SVHN test dataset
    print("Loading SVHN test dataset...")
    dataset = datasets.SVHN(root=os.path.join(base_dir, 'data', 'raw'), split='test', download=False)
    
    max_samples = 1000
    total_samples_to_test = min(max_samples, len(multi_digit_samples))
    test_samples = multi_digit_samples[:total_samples_to_test]
    
    print(f"Testing on {len(test_samples)} multi-digit SVHN samples...")
    output_dir = os.path.join(base_dir, "outputs", "fullpipelines_predictions")

    correct = 0
    for i, (image_index, true_str) in enumerate(test_samples):
        img, _ = dataset[image_index]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img.save(temp_path)

        try:
            # Run detection
            bboxes, _ = run_yolo_on_image(yolo_model, temp_path)
            
            recognized_str = ""
            digit_images_for_vis = []
            digits = []
            image_for_vis = cv2.imread(temp_path)

            if len(bboxes) > 0:
                bboxes = sorted(bboxes, key=lambda b: b[0])
                digits = recognize_digits(digit_model, temp_path, bboxes)
                recognized_str = "".join(str(d[0]) for d in digits)

                for bbox in bboxes:
                    x1, y1, x2, y2 = map(int, bbox)
                    digit_crop = image_for_vis[y1:y2, x1:x2]
                    if digit_crop.size > 0:
                        resized_digit = cv2.resize(digit_crop, (28, 28))
                        digit_images_for_vis.append(resized_digit)

            if recognized_str == true_str:
                correct += 1
            else:
                print(f"Mismatch: Predicted '{recognized_str}', True '{true_str}' (Image index {image_index})")
            
            result = {
                'image': cv2.cvtColor(image_for_vis, cv2.COLOR_BGR2RGB),
                'true_str': true_str,
                'pred_str': recognized_str,
                'bboxes': bboxes,
                'digit_images': digit_images_for_vis,
                'digits': digits
            }
            visualize_results(result, output_dir, image_index)

        except Exception as e:
            import traceback
            print(f"Error processing image {image_index}:")
            traceback.print_exc()
        finally:
            os.unlink(temp_path)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(test_samples)} samples...")

    accuracy = correct / len(test_samples) if len(test_samples) > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(test_samples)})")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()