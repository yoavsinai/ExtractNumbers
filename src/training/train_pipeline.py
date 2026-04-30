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

from image_preprocessing.digit_preprocessor import enhance_digit
from utils.data_utils import iter_new_samples, get_gt_from_anno

# Structured Output Directories
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")
DATASETS_DIR = os.path.join(BASE_DIR, "outputs", "datasets")
PREDS_DIR = os.path.join(BASE_DIR, "outputs", "raw_predictions")

# Specific Paths
BOUNDING_BOX_SRC = os.path.join(BASE_DIR, "src", "bounding_box")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
BEST_GLOBAL_PATH = os.path.join(TRAINED_MODELS_DIR, "globalbb.pt")
BEST_INDIVIDUAL_PATH = os.path.join(TRAINED_MODELS_DIR, "individualbb.pt")
GLOBAL_PREDS_CSV = os.path.join(PREDS_DIR, "globalbb_preds.csv")
INDIV_PREDS_CSV = os.path.join(PREDS_DIR, "individualbb_preds.csv")
PROG_IMAGE_PATH = os.path.join(VIS_DIR, "pipeline_progression.png")
SHARPENED_DIR = os.path.join(DATASETS_DIR, "sharpened_crops")

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
    parser.add_argument("--force-inference", action="store_true", help="Force re-running of all inference stages.")
    parser.add_argument("--viz-only", action="store_true", help="Just regenerate the progression visualization using 3 random samples.")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(SHARPENED_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    if args.viz_only:
        print("=> --viz-only enabled. Skipping Stage 1-3 checks and jumping to fresh visualization.")
    else:
        predictions_exist = os.path.exists(GLOBAL_PREDS_CSV)
        
        if args.analyze_only and predictions_exist:
            print("=> --analyze-only enabled and predictions exist. Skipping Stage 1.")
        else:
            # We still train in the yolo_runs dir but we will copy the result out
            YOLO_OUT = os.path.join(BASE_DIR, "outputs", "yolo_runs")
            skip_train_global = os.path.exists(BEST_GLOBAL_PATH) and not args.force_train
            detector_script = os.path.join(BOUNDING_BOX_SRC, "globalbb_detector.py")
            
            detector_args = ["--output-dir", YOLO_OUT, "--epochs", str(args.epochs)]
            if skip_train_global:
                detector_args.append("--skip-train")
                if not args.force_inference:
                   detector_args.append("--skip-train") # This is redundant but safety for next line
                print("=> Found existing GlobalBB weights. Running inference (unless cached)...")
            else:
                print("=> No GlobalBB weights found or force-train enabled. Starting training...")

            if not run_python_script(detector_script, detector_args, capture=False):
                print("Failed at Stage 1 detection.")
                sys.exit(1)
            
            # Copy final weights if they were just trained
            new_global_weights = os.path.join(YOLO_OUT, "globalbb_run", "weights", "best.pt")
            if os.path.exists(new_global_weights):
                shutil.copy(new_global_weights, BEST_GLOBAL_PATH)
            
            # Copy predictions CSV
            new_preds = os.path.join(YOLO_OUT, "globalbb_predictions.csv")
            if os.path.exists(new_preds):
                shutil.copy(new_preds, GLOBAL_PREDS_CSV)

        dataset_root = os.path.join(BASE_DIR, "data", "digits_data")
        all_samples = iter_new_samples(dataset_root)
        
        preview_grid = []
        category_counts = {} # Dynamic counting
        for s in tqdm(all_samples, desc="Generating label preview"):
            cat = s["category"]
            if category_counts.get(cat, 0) < 3:
                img = cv2.imread(s["image_path"])
                global_boxes, _, _, _ = get_gt_from_anno(s['anno_path'])
                for x1, y1, x2, y2 in global_boxes:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, cat, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                preview_grid.append(img)
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
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

        # Copy IndividualBB weights if they were just trained
        new_indiv_weights = os.path.join(YOLO_OUT, "individualbb_run", "weights", "best.pt")
        if os.path.exists(new_indiv_weights):
            shutil.copy(new_indiv_weights, BEST_INDIVIDUAL_PATH)

    # Load Stage 1 predictions
    df = pd.read_csv(GLOBAL_PREDS_CSV)
    df_valid = df.dropna(subset=['pred_x1'])
    print(f"\n=> Loaded {len(df)} Stage 1 records ({len(df_valid)} valid detections).")

    # Load IndividualBB Model for Stage 4 inference
    from ultralytics import YOLO
    indiv_model = YOLO(BEST_INDIVIDUAL_PATH)

    # Cache check for Stage 4
    if os.path.exists(INDIV_PREDS_CSV) and not args.force_inference and not args.force_train:
        print(f"=> Found existing IndividualBB predictions at {INDIV_PREDS_CSV}. Loading...")
        df_indiv = pd.read_csv(INDIV_PREDS_CSV)
        
        # Pick 3 random samples for visualization from the cached results
        viz_samples = []
        for cat in ['race_numbers', 'handwritten', 'svhn']:
             cat_rows = df_indiv[df_indiv['category'] == cat]
             if not cat_rows.empty:
                 row = cat_rows.sample(1).iloc[0]
                 img = cv2.imread(row['image_path'])
                 
                 # Reconstruct sharpened crop (or load it)
                 x1, y1, x2, y2 = int(row['pred_x1_global']), int(row['pred_y1_global']), int(row['pred_x2_global']), int(row['pred_y2_global'])
                 crop = img[y1:y2, x1:x2]
                 sharp = enhance_digit(crop, upscale_factor=2.0)
                 
                 # Parse individual boxes from string format if needed or handle row columns
                 # For now, let's just use the logic to reconstruct from the loop if we don't store boxes in nested format
                 # To keep it simple, if user wants viz, they might need to run inference or we store boxes.
                 # Let's actually run a mini-loop for viz samples if cached.
                 
                 # Re-detect just for the 3 viz samples to keep code clean
                 results = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
                 indiv_preds = []
                 if results and len(results[0].boxes) > 0:
                     indiv_preds = results[0].boxes.xyxy.detach().cpu().numpy()

                 viz_samples.append({
                    'cat': cat,
                    'original': img,
                    'crop_coords': (x1, y1, x2, y2),
                    'crop': crop,
                    'sharpened': sharp,
                    'indiv_preds': indiv_preds,
                    'sample_info': row
                 })
    else:
        # Process detections - Shuffle for truly random visualization samples
        shuffled_df = df_valid.sample(frac=1).reset_index(drop=True)
        
        viz_samples = []
        all_indiv_results = []
        
        print(f"\n=> Processing Individual Digit Detection (Stage 4)...")
        for _, row in tqdm(shuffled_df.iterrows(), total=len(shuffled_df), desc="Predicting Digits"):
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
            sharp = enhance_digit(crop, upscale_factor=2.0)
            
            # 2. Individual Detection
            results = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
            indiv_preds = []
            if results and len(results[0].boxes) > 0:
                indiv_preds = results[0].boxes.xyxy.detach().cpu().numpy()

            # Store result
            res_entry = row.to_dict()
            res_entry.update({
                'pred_x1_global': x1, 'pred_y1_global': y1, 'pred_x2_global': x2, 'pred_y2_global': y2,
                'num_digits': len(indiv_preds)
            })
            all_indiv_results.append(res_entry)

            # Capture a few random samples for the final summary image
            if len(viz_samples) < 3 and cat not in [v['cat'] for v in viz_samples]:
                viz_samples.append({
                    'cat': cat,
                    'original': img,
                    'crop_coords': (x1, y1, x2, y2),
                    'crop': crop,
                    'sharpened': sharp,
                    'indiv_preds': indiv_preds,
                    'sample_info': row
                })
                 
            # Optimization: Stop early if we only want visualization and have enough samples
            if args.viz_only and len(viz_samples) >= 3:
                print("\n=> Collected 3 random samples for visualization. Exiting early.")
                break
        
        if not args.viz_only:
            pd.DataFrame(all_indiv_results).to_csv(INDIV_PREDS_CSV, index=False)
            print(f"=> Stage 4 predictions saved to {INDIV_PREDS_CSV}")

    print(f"\n=== Rendering Pipeline Progression Visualization ===")
    
    fig, axes = plt.subplots(3, 5, figsize=(22, 12))
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
        
        # GT (Green) from annotations.json
        anno_path = sample['image_path'].replace("original.png", "annotations.json")
        if os.path.exists(anno_path):
            global_boxes, _, _, _ = get_gt_from_anno(anno_path)
            for x1, y1, x2, y2 in global_boxes:
                axes[i, 1].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2, linestyle='--'))

        axes[i, 1].set_title("2. Global Detection\n(GlobalBB)", fontsize=10)
        axes[i, 1].axis('off')

        # Panel 3: Raw Crop (Stage 2 input)
        crop_rgb = cv2.cvtColor(res['crop'], cv2.COLOR_BGR2RGB)
        axes[i, 2].imshow(crop_rgb)
        axes[i, 2].set_title("3. Raw Crop\n(Unsharpened)", fontsize=10)
        axes[i, 2].axis('off')
        
        # Panel 4: Image Enhancement
        sharp_rgb = cv2.cvtColor(res['sharpened'], cv2.COLOR_BGR2RGB)
        axes[i, 3].imshow(sharp_rgb)
        axes[i, 3].set_title("4. Image Enhancement\n(Sharpening)", fontsize=10)
        axes[i, 3].axis('off')
        
        # Panel 5: Individual Digit Detection
        axes[i, 4].imshow(sharp_rgb)
        for p in res['indiv_preds']:
            px1, py1, px2, py2 = p
            axes[i, 4].add_patch(plt.Rectangle((px1, py1), px2-px1, py2-py1, fill=False, edgecolor='red', linewidth=2))
        
        # GT for individual digits (Green) from annotations.json
        if os.path.exists(anno_path):
            _, digit_info, _, _ = get_gt_from_anno(anno_path)
            cx1, cy1, cx2, cy2 = res['crop_coords']
            for digit in digit_info:
                dx1, dy1, dx2, dy2 = digit['bbox']
                nx1 = (dx1 - cx1) * 2.0
                ny1 = (dy1 - cy1) * 2.0
                nx2 = (dx2 - cx1) * 2.0
                ny2 = (dy2 - cy1) * 2.0
                if nx2 > 0 and ny2 > 0:
                   axes[i, 4].add_patch(plt.Rectangle((nx1, ny1), nx2-nx1, ny2-ny1, fill=False, edgecolor='lime', linewidth=2, linestyle=':'))

        axes[i, 4].set_title("5. Individual Detection\n(IndividualBB)", fontsize=10)
        axes[i, 4].axis('off')

    plt.suptitle("FULL EXTRACTION PIPELINE: Multi-Stage Progression", fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(PROG_IMAGE_PATH, bbox_inches='tight', dpi=150)
    
    print(f"\n=> SUCCESS: All stages complete.")
    print(f"=> Process summary image saved to: {PROG_IMAGE_PATH}")

    # Restore the classic GlobalBB summary report
    print("\nUpdating classic globalbb comparison summary...")
    # NOTE: We keep visualizer for backward compatibility but redirect output
    visualizer_script = os.path.join(BOUNDING_BOX_SRC, "visualize_globalbb_results.py")
    if os.path.exists(visualizer_script):
        # We manually copy the summary if it was generated
        run_python_script(visualizer_script, ["--output-dir", YOLO_OUT])
        new_summary = os.path.join(YOLO_OUT, "globalbb_comparison_summary.png")
        if os.path.exists(new_summary):
            shutil.copy(new_summary, os.path.join(VIS_DIR, "globalbb_summary.png"))

    print("\n--- Training Results Summary (Stage 2) ---")
    results_csv = os.path.join(YOLO_OUT, "individualbb_run", "results.csv")
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
