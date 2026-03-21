import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from full_pipelines.single_photo_full_pipeline import load_yolo_model, load_digit_model, run_yolo_on_image, recognize_digits

def get_ground_truth_from_mask(mask_path):
    """
    This is a placeholder. We need to figure out how to get the ground truth number.
    Let's assume for now that the folder name contains the number.
    e.g. data/segmentation/natural/123/mask.png -> '123'
    """
    try:
        # The number is the name of the parent folder of the mask
        return os.path.basename(os.path.dirname(mask_path))
    except Exception:
        return None

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
        if i < len(digits):
            pred, conf = digits[i]
            label = f"{pred} ({conf:.2f})"
            ax.text(x1, y2 + 15, label, color='white', fontsize=10, bbox=dict(facecolor='red', alpha=0.5))

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

    data_dir = os.path.join(base_dir, 'data', 'segmentation', 'natural')
    output_dir = os.path.join(base_dir, "outputs", "fullpipelines_predictions")

    # Find all mask files
    mask_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file == 'mask.png':
                mask_files.append(os.path.join(root, file))

    total_samples = len(mask_files)
    print(f"Testing on {total_samples} samples from {data_dir}...")

    correct = 0
    for i, mask_path in enumerate(mask_files):
        true_str = get_ground_truth_from_mask(mask_path)
        image_path = os.path.join(os.path.dirname(mask_path), 'image.jpg')

        if not os.path.exists(image_path):
            continue
        
        image_for_vis = cv2.imread(image_path)
        if image_for_vis is None:
            print(f"Could not read image: {image_path}")
            continue

        try:
            # Run detection
            bboxes, _ = run_yolo_on_image(yolo_model, image_path)
            
            recognized_str = ""
            digit_images_for_vis = []
            digits = []

            if len(bboxes) > 0:
                bboxes = sorted(bboxes, key=lambda b: b[0])
                digits = recognize_digits(digit_model, image_path, bboxes)
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
                print(f"Mismatch: Predicted '{recognized_str}', True '{true_str}' (image {i}: {image_path})")

            result = {
                'image': cv2.cvtColor(image_for_vis, cv2.COLOR_BGR2RGB),
                'true_str': true_str,
                'pred_str': recognized_str,
                'bboxes': bboxes,
                'digit_images': digit_images_for_vis,
                'digits': digits
            }
            visualize_results(result, output_dir, i)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total_samples} samples...")

    accuracy = correct / total_samples if total_samples > 0 else 0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total_samples})")
    print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
