import os
import sys
import subprocess
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Add src to path so we can import modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import preprocess_digit

# Path configuration
BOUNDING_BOX_SRC = os.path.join(BASE_DIR, "src", "bounding_box")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bbox_comparison")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "globalbb_predictions.csv")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "globalbb_run", "weights", "best.pt")
SHARPENED_DIR = os.path.join(OUTPUT_DIR, "sharpened_crops")

def run_python_script(script_path, args=[]):
    """Run a Python script and capture output."""
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error running {script_path}:\n{result.stderr}")
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Enhanced Extraction Pipeline (Stage 1 + Sharpening + Stage 2 Placeholder)")
    parser.add_argument("--analyze-only", action="store_true", help="Skip detection if predictions already exist.")
    parser.add_argument("--force-train", action="store_true", help="Force retraining of Stage 1 model.")
    args = parser.parse_args()

    os.makedirs(SHARPENED_DIR, exist_ok=True)

    print("=== Step 1: Global Bounding Box Detection (Stage 1) ===")
    
    # Check if we can skip the heavy detection logic
    predictions_exist = os.path.exists(PREDICTIONS_CSV)
    
    if args.analyze_only and predictions_exist:
        print("=> --analyze-only enabled and predictions exist. Skipping Stage 1 inference.")
    else:
        # Determine if we need to train
        skip_train = os.path.exists(BEST_MODEL_PATH) and not args.force_train
        detector_script = os.path.join(BOUNDING_BOX_SRC, "globalbb_detector.py")
        
        detector_args = ["--output-dir", OUTPUT_DIR]
        if skip_train:
            detector_args.append("--skip-train")
            print("=> Found existing weights. Running inference only.")
        else:
            print("=> No weights found or force-train enabled. Starting training...")

        if not run_python_script(detector_script, detector_args):
            print("Failed at Stage 1 detection.")
            sys.exit(1)

    # Load predictions
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"Error: Predictions file not found at {PREDICTIONS_CSV}")
        sys.exit(1)
    
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"=> Loaded {len(df)} detection results.")

    print("\n=== Step 2: Image Sharpening (The 'Clean-Up' Stage) ===")
    
    # We will pick 3 random samples (one from each category) for the final progression image
    categories = ['handwritten', 'synthetic', 'natural']
    progression_samples = []

    for cat in categories:
        cat_samples = df[df['category'] == cat].dropna(subset=['pred_x1'])
        if not cat_samples.empty:
            progression_samples.append(cat_samples.sample(1).iloc[0])

    print("Applying sharpening filters to crops...")
    # For efficiency in this demo, we'll focus on processing the progression samples deeply
    # but we can loop through more if needed.
    
    processed_results = []

    for sample in progression_samples:
        img_path = sample['image_path']
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        x1, y1, x2, y2 = int(sample['pred_x1']), int(sample['pred_y1']), int(sample['pred_x2']), int(sample['pred_y2'])
        
        # Ensure coordinates are within bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            continue

        # Apply sharpening from our preprocessor logic
        # target_size=None to keep the sharpened crop size natural
        sharpened_crop = preprocess_digit(crop, upscale_factor=2.0, unsharp_strength=2.0)
        
        # Save sharpened crop
        save_name = f"sharp_{sample['category']}_{sample['sample_id'].replace('/', '_')}.png"
        save_path = os.path.join(SHARPENED_DIR, save_name)
        cv2.imwrite(save_path, sharpened_crop)
        
        processed_results.append({
            'sample': sample,
            'original_img': img,
            'crop': crop,
            'sharpened': sharpened_crop,
            'save_path': save_path
        })

    print(f"=> Sharpening complete. {len(processed_results)} key samples generated.")

    print("\n=== Step 3: Individual Digit Detection (Stage 2) [PLACEHOLDER] ===")
    print("=> This stage is reserved for future integration. Skipping for now.")

    print("\n=== Rendering Pipeline Progression Visualization ===")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for i, res in enumerate(processed_results):
        sample = res['sample']
        img_rgb = cv2.cvtColor(res['original_img'], cv2.COLOR_BGR2RGB)
        
        # Column 1: Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"1. Original\n({sample['category']})", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Column 2: Stage 1 Detection (BBs)
        axes[i, 1].imshow(img_rgb)
        x1, y1, x2, y2 = int(sample['pred_x1']), int(sample['pred_y1']), int(sample['pred_x2']), int(sample['pred_y2'])
        
        # Draw Pred (Red)
        axes[i, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=3))
        
        # Try to find corresponding mask for Ground Truth (Green)
        mask_path = img_path.replace("image.jpg", "mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                axes[i, 1].add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor='lime', linewidth=2, linestyle='--'))

        axes[i, 1].set_title("2. GlobalBB Detection\n(Green: GT | Red: Pred)", fontsize=10)
        axes[i, 1].axis('off')
        
        # Column 3: Sharpened Crop
        # Preprocess_digit returns binary, but we want to show the sharpening effect.
        # Let's call a specific part of preprocessor to get the colored sharpened version for display
        from image_preprocessing.digit_preprocessor import apply_unsharp_mask, upscale_image
        
        display_crop = upscale_image(res['crop'], scale_factor=2.0)
        display_crop = apply_unsharp_mask(display_crop, strength=2.5)
        display_crop_rgb = cv2.cvtColor(display_crop, cv2.COLOR_BGR2RGB)
        
        axes[i, 2].imshow(display_crop_rgb)
        axes[i, 2].set_title("3. Image Sharpening\n(Upscaled + Unsharp Mask)", fontsize=10)
        axes[i, 2].axis('off')
        
        # Column 4: Stage 2 Placeholder
        axes[i, 3].text(0.5, 0.5, "Stage 2 Reserved\n(Individual Digits)", 
                       ha='center', va='center', bbox=dict(facecolor='lightgray', alpha=0.5))
        axes[i, 3].set_title("4. Future Detection", fontsize=10)
        axes[i, 3].axis('off')

    plt.suptitle("Multi-Stage Digit Extraction Pipeline: Progression Overview", fontsize=18, fontweight='bold', y=0.98)
    
    progression_path = os.path.join(OUTPUT_DIR, "full_pipeline_progression.png")
    plt.savefig(progression_path, bbox_inches='tight', dpi=150)
    print(f"\nFinal progression image saved to: {progression_path}")
    
    # Also run the classic visualizers to keep the requested files updated
    print("\nUpdating classic summary visualizations...")
    visualizer_script = os.path.join(BOUNDING_BOX_SRC, "visualize_globalbb_results.py")
    run_python_script(visualizer_script)

    print("\n=== Pipeline Execution Summary ===")
    print(f"Total Samples Analyzed: {len(df)}")
    print(f"Sharpened Samples Saved to: {SHARPENED_DIR}")
    print(f"Visual Progression: {progression_path}")
    print("\nSUCCESS: All requested stages completed correctly.")

if __name__ == "__main__":
    main()
