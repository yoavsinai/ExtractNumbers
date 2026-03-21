import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Path configuration
BASE_DIR = r"C:\Users\user\OneDrive - Bar-Ilan University - Students\Bar Ilan\C\ExtractNumbers\ExtractNumbers"
CSV_PATH = os.path.join(BASE_DIR, "outputs", "bbox_comparison", "yolo_predictions.csv")
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "outputs", "bbox_comparison", "yolo_dataset", "images", "val")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "outputs", "bbox_comparison", "yolo_comparison_summary.png")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find CSV at {CSV_PATH}")
        return
    
    if not os.path.exists(VAL_IMAGES_DIR):
        print(f"Error: Val directory not found. Please run training first.")
        return

    # 1. Load prediction results.
    df = pd.read_csv(CSV_PATH)
    
    # 2. Keep only samples that exist in the validation split.
    # Validation filenames follow: category_id.jpg (e.g., natural_10.jpg).
    val_filenames = set(os.listdir(VAL_IMAGES_DIR))
    
    def is_test_image(row):
        # Extract the sample index (the part after '/').
        idx = row['sample_id'].split('/')[-1]
        target_filename = f"{row['category']}_{idx}.jpg"
        return target_filename in val_filenames

    df['is_test'] = df.apply(is_test_image, axis=1)
    test_df = df[df['is_test'] == True].copy()

    print(f"Filtering complete: Found {test_df['image_path'].nunique()} unique test images.")

    categories = ['natural', 'synthetic', 'handwritten']
    samples_per_cat = 2
    
    fig, axes = plt.subplots(len(categories), samples_per_cat, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.4)

    for i, cat in enumerate(categories):
        cat_df = test_df[test_df['category'] == cat]
        unique_images = cat_df['image_path'].unique()
        
        if len(unique_images) < samples_per_cat:
            print(f"Warning: Not enough test images for {cat}")
            continue
            
        selected_img_paths = np.random.choice(unique_images, samples_per_cat, replace=False)
        
        for j, img_path in enumerate(selected_img_paths):
            mask_path = img_path.replace("image.jpg", "mask.png")
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = axes[i, j]
            ax.imshow(img)
            
            # Draw ground-truth boxes (green).
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for lbl_idx, cnt in enumerate(contours):
                    x_g, y_g, w_g, h_g = cv2.boundingRect(cnt)
                    area_g = w_g * h_g
                    if area_g > 10:
                        ax.add_patch(plt.Rectangle((x_g, y_g), w_g, h_g, fill=False, edgecolor='lime', linewidth=3))
            
            # Draw YOLO predicted boxes (red).
            predictions = cat_df[cat_df['image_path'] == img_path]
            for _, pred in predictions.iterrows():
                if pd.notna(pred['pred_x1']):
                    p_x1, p_y1, p_x2, p_y2 = pred['pred_x1'], pred['pred_y1'], pred['pred_x2'], pred['pred_y2']
                    ax.add_patch(plt.Rectangle((p_x1, p_y1), p_x2-p_x1, p_y2-p_y1, fill=False, edgecolor='red', linewidth=2, linestyle='--'))
                    ax.text(p_x1, p_y1-5, f"{pred['pred_conf']:.2f}", color='red', fontsize=8, weight='bold')

            ax.set_title(f"TEST IMAGE: {cat}")
            ax.axis('off')

    plt.suptitle("YOLO Final Test Results (Unseen Data)\nGreen = Truth | Red = Prediction", fontsize=16)
    plt.savefig(OUTPUT_IMAGE, bbox_inches='tight')
    print(f"Test visualization saved to: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    main()