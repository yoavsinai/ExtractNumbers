import os
import sys
import subprocess
import pandas as pd
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Add src to path so we can import modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import preprocess_digit, upscale_image, apply_bilateral_filter, apply_unsharp_mask

# Path configuration
BOUNDING_BOX_SRC = os.path.join(BASE_DIR, "src", "bounding_box")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bbox_comparison")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "globalbb_predictions.csv")
BEST_GLOBAL_PATH = os.path.join(OUTPUT_DIR, "globalbb_run", "weights", "best.pt")
BEST_INDIVIDUAL_PATH = os.path.join(OUTPUT_DIR, "individualbb_run", "weights", "best.pt")
SHARPENED_DIR = os.path.join(OUTPUT_DIR, "sharpened_crops")
PROG_IMAGE_PATH = os.path.join(OUTPUT_DIR, "full_pipeline_progression.png")

def run_python_script(script_path, args=[], capture=True):
    """Run a Python script. If capture=True, output is hidden until completion."""
    cmd = [sys.executable, script_path] + args
    if not capture:
        # Standard output/error will go directly to the user's console
        result = subprocess.run(cmd)
        return result.returncode == 0
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error running {script_path}:\n{result.stderr}")
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Enhanced Extraction Pipeline (Stage 1 + Sharpening + Stage 4 Individual Detection)")
    parser.add_argument("--analyze-only", action="store_true", help="Skip detection if predictions already exist.")
    parser.add_argument("--force-train", action="store_true", help="Force retraining of models.")
    parser.add_argument("--viz-only", action="store_true", help="Just regenerate the progression visualization using 3 random samples.")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(SHARPENED_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    if args.viz_only:
        print("=> --viz-only enabled. Skipping Stage 1-3 checks and jumping to fresh visualization.")
    else:
        print("=== Step 1: Global Bounding Box Detection (GlobalBB) ===")
        print("Goal: Detect the entire number sequence as a single entity (GlobalBB).")
        
        # Check if we can skip the heavy detection logic
        predictions_exist = os.path.exists(PREDICTIONS_CSV)
        
        if args.analyze_only and predictions_exist:
            print("=> --analyze-only enabled and predictions exist. Skipping Stage 1.")
        else:
            # Determine if we need to train GlobalBB
            skip_train_global = os.path.exists(BEST_GLOBAL_PATH) and not args.force_train
            detector_script = os.path.join(BOUNDING_BOX_SRC, "globalbb_detector.py")
            
            detector_args = ["--output-dir", OUTPUT_DIR, "--epochs", str(args.epochs)]
            if skip_train_global:
                detector_args.append("--skip-train")
                print("=> Found existing GlobalBB weights. Running inference only.")
            else:
                print("=> No GlobalBB weights found or force-train enabled. Starting training...")

            if not run_python_script(detector_script, detector_args):
                print("Failed at Stage 1 detection.")
                sys.exit(1)

        # Generate the Label Preview diagnostic
        print("\nGenerating label preview (diagnostic)...")
        from bounding_box.globalbb_detector import iter_samples, _read_mask_grayscale
        categories = ["natural", "synthetic", "handwritten"]
        dataset_root = os.path.join(BASE_DIR, "data", "segmentation")
        all_samples = iter_samples(dataset_root, categories)
        
        preview_grid = []
        category_counts = {cat: 0 for cat in categories}
        for s in all_samples:
            cat = s["category"]
            if category_counts[cat] < 3:
                img = cv2.imread(s["image_path"])
                mask = _read_mask_grayscale(s["mask_path"])
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
                dilated = cv2.dilate(mask, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (0, 255, 0), 2)
                cv2.putText(img, cat, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                preview_grid.append(img)
                category_counts[cat] += 1
        
        if preview_grid:
            fig, axes = plt.subplots(3, 3, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                if i < len(preview_grid):
                    ax.imshow(cv2.cvtColor(preview_grid[i], cv2.COLOR_BGR2RGB))
                ax.axis('off')
            preview_path = os.path.join(OUTPUT_DIR, "preview_labels_before_training.png")
            plt.tight_layout()
            plt.savefig(preview_path)
            plt.close()
            print(f"=> Saved label preview to: {preview_path}")

        print("\n=== Step 2: Image Enhancement (Sharpening) ===")
        print("Goal: Apply upscaling and unsharp masking to prepare the dataset for individual digit detection.")
        
        indiv_script = os.path.join(BOUNDING_BOX_SRC, "individualbb_detector.py")
        
        # Check if we need to regenerate the sharpened dataset
        individual_out_root = os.path.join(OUTPUT_DIR, "individualbb_dataset")
        dataset_exists = os.path.exists(individual_out_root)
        
        if dataset_exists and not args.force_train:
            print("=> Sharpened dataset already exists. Skipping enhancement phase.")
        else:
            print("=> Generating sharpened crops for training (This is the Enhancement Phase)...")
            # Use capture=False so user sees the tqdm progress bar
            if not run_python_script(indiv_script, ["--output-dir", OUTPUT_DIR, "--prepare-only"], capture=False):
                print("Failed at Stage 2 Image Enhancement.")
                sys.exit(1)

        print("\n=== Step 3: Individual Digit Detection (IndividualBB) ===")
        print("Explanation: This stage trains/loads a second YOLO model using the sharpened crops from Step 2.")
        
        # Check if IndividualBB weights exist
        skip_train_indiv = os.path.exists(BEST_INDIVIDUAL_PATH) and not args.force_train
        
        indiv_args = ["--output-dir", OUTPUT_DIR, "--epochs", str(args.epochs), "--train-only"]
        if skip_train_indiv:
            indiv_args.append("--skip-train")
            print("=> Found existing IndividualBB weights. Loading model...")
        else:
            print("=> Starting IndividualBB model training...")

        # Use capture=False so user sees the YOLO training bars
        if not run_python_script(indiv_script, indiv_args, capture=False):
            print("Failed at IndividualBB training.")
            sys.exit(1)

    # Load Stage 1 predictions
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"Error: Predictions file not found at {PREDICTIONS_CSV}")
        sys.exit(1)
    
    df = pd.read_csv(PREDICTIONS_CSV)
    df_valid = df.dropna(subset=['pred_x1'])
    print(f"\n=> Loaded {len(df)} Stage 1 records ({len(df_valid)} valid detections).")

    # Load IndividualBB Model for Stage 4 inference
    from ultralytics import YOLO
    indiv_model = YOLO(BEST_INDIVIDUAL_PATH)

    print(f"\n=== Step 4: Batch Processing & Enhancement ===")
    print(f"Goal: Cropping, Sharpening, and Individual Detection for ALL {len(df_valid)} images.")
    
    # We will store results for visualization separately
    viz_samples = []
    categories = ['handwritten', 'synthetic', 'natural']
    
    # Process detections - Shuffle for truly random visualization samples
    shuffled_df = df_valid.sample(frac=1).reset_index(drop=True)
    
    print(f"\n=> Processing images (Early exit enabled for --viz-only)")
    for _, row in tqdm(shuffled_df.iterrows(), total=len(shuffled_df), desc="Sharpening & Detecting"):
        img_path = row['image_path']
        cat = row['category']
        img = cv2.imread(img_path)
        if img is None: continue
        
        x1, y1, x2, y2 = int(row['pred_x1']), int(row['pred_y1']), int(row['pred_x2']), int(row['pred_y2'])
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        # 1. Enhancement (Sharpening)
        sharp = upscale_image(crop, scale_factor=2.0)
        sharp = apply_bilateral_filter(sharp)
        sharp = apply_unsharp_mask(sharp, strength=2.0)
        
        # Save sharpened crop to disk
        sample_name = os.path.splitext(os.path.basename(img_path))[0]
        crop_filename = f"{cat}_{sample_name}_sharp.jpg"
        cv2.imwrite(os.path.join(SHARPENED_DIR, crop_filename), sharp)
        
        # 2. Individual Detection
        results = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
        indiv_preds = []
        if results and len(results[0].boxes) > 0:
            indiv_preds = results[0].boxes.xyxy.detach().cpu().numpy()

        # Capture a few random samples for the final summary image
        if len(viz_samples) < 3 and cat not in [v['cat'] for v in viz_samples]:
             viz_samples.append({
                'cat': cat,
                'original': img,
                'crop_coords': (x1, y1, x2, y2),
                'sharpened': sharp,
                'indiv_preds': indiv_preds,
                'sample_info': row
             })
             
        # Optimization: Stop early if we only want visualization and have enough samples
        if args.viz_only and len(viz_samples) >= 3:
            print("\n=> Collected 3 random samples for visualization. Exiting early.")
            break

    print(f"\n=== Rendering Pipeline Progression Visualization ===")
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    for i, res in enumerate(viz_samples):
        sample = res['sample_info']
        img_rgb = cv2.cvtColor(res['original'], cv2.COLOR_BGR2RGB)
        
        # Panel 1: Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"1. Original\n({res['cat']})", fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Panel 2: Stage 1 Detection (GlobalBB)
        axes[i, 1].imshow(img_rgb)
        x1, y1, x2, y2 = res['crop_coords']
        axes[i, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=3))
        
        # GT (Green) - Match Sanity Check logic (multiple boxes if separated)
        mask_path = sample['image_path'].replace("image.jpg", "mask.png")
        if os.path.exists(mask_path):
            from bounding_box.globalbb_detector import _read_mask_grayscale
            mask = _read_mask_grayscale(mask_path)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw * bh > 10:
                    axes[i, 1].add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, edgecolor='lime', linewidth=2, linestyle='--'))

        axes[i, 1].set_title("2. Global Detection\n(GlobalBB)", fontsize=10)
        axes[i, 1].axis('off')
        
        # Panel 3: Image Enhancement
        sharp_rgb = cv2.cvtColor(res['sharpened'], cv2.COLOR_BGR2RGB)
        axes[i, 2].imshow(sharp_rgb)
        axes[i, 2].set_title("3. Image Enhancement\n(Sharpening)", fontsize=10)
        axes[i, 2].axis('off')
        
        # Panel 4: Individual Digit Detection
        axes[i, 3].imshow(sharp_rgb)
        for p in res['indiv_preds']:
            px1, py1, px2, py2 = p
            axes[i, 3].add_patch(plt.Rectangle((px1, py1), px2-px1, py2-py1, fill=False, edgecolor='red', linewidth=2))
        
        # GT for individual digits (Green)
        if os.path.exists(mask_path):
            from bounding_box.globalbb_detector import extract_digit_bboxes
            digit_boxes = extract_digit_bboxes(mask)
            cx1, cy1, cx2, cy2 = res['crop_coords']
            for (dx1, dy1, dx2, dy2) in digit_boxes:
                nx1 = (dx1 - cx1) * 2.0
                ny1 = (dy1 - cy1) * 2.0
                nx2 = (dx2 - cx1) * 2.0
                ny2 = (dy2 - cy1) * 2.0
                if nx2 > 0 and ny2 > 0:
                   axes[i, 3].add_patch(plt.Rectangle((nx1, ny1), nx2-nx1, ny2-ny1, fill=False, edgecolor='lime', linewidth=2, linestyle=':'))

        axes[i, 3].set_title("4. Individual Detection\n(IndividualBB)", fontsize=10)
        axes[i, 3].axis('off')

    plt.suptitle("FULL EXTRACTION PIPELINE: Multi-Stage Progression", fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(PROG_IMAGE_PATH, bbox_inches='tight', dpi=150)
    
    print(f"\n=> SUCCESS: All stages complete.")
    print(f"=> Process summary image saved to: {PROG_IMAGE_PATH}")

    # Restore the classic GlobalBB summary report
    print("\nUpdating classic globalbb comparison summary...")
    visualizer_script = os.path.join(BOUNDING_BOX_SRC, "visualize_globalbb_results.py")
    if os.path.exists(visualizer_script):
        run_python_script(visualizer_script)

    print("\n--- Training Results Summary (Stage 2) ---")
    results_csv = os.path.join(OUTPUT_DIR, "individualbb_runs", "run1", "results.csv")
    if os.path.exists(results_csv):
        rdf = pd.read_csv(results_csv)
        rdf.columns = rdf.columns.str.strip()
        last = rdf.iloc[-1]
        print(f"Final Individual Detection mAP50: {last['metrics/mAP50(B)']:.2%}")
        print(f"Precision: {last['metrics/precision(B)']:.2%}")
        print(f"Recall: {last['metrics/recall(B)']:.2%}")
    else:
        print("Note: Run training to see full metrics.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
