import argparse
import os
import subprocess
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src"))

from utils.data_utils import iter_new_samples, get_gt_from_anno
SRC_DIR = os.path.join(BASE_DIR, "src", "bounding_box")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bbox_comparison")
GlobalBB_RUN_DIR = os.path.join(OUTPUT_DIR, "globalbb_runs", "run1")

def run_quiet_script(script_name, args=[]):
    """Run a Python script and print output only on error."""
    script_path = os.path.join(SRC_DIR, script_name)
    cmd = [sys.executable, script_path] + args
    
    # Use UTF-8 decoding to avoid UnicodeDecodeError on subprocess output.
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

def analyze_epochs(csv_path):
    """Analyze GlobalBB results.csv to track epoch-by-epoch improvement."""
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    map_col = 'metrics/mAP50(B)'
    
    print("\n--- Epoch-by-Epoch Improvement (mAP50) ---")
    map_vals = df[map_col].values
    for i in range(len(map_vals)):
        diff = map_vals[i] - map_vals[i-1] if i > 0 else 0
        trend = "^" if diff > 0.005 else "~"
        print(f"Epoch {i+1:02d}: {map_vals[i]:.4f} (Change: {diff:+.4f}) {trend}")
    
    # Simple recommendation based on the recent mAP trend.
    last_5_avg_diff = np.mean(np.diff(map_vals[-5:]))
    if last_5_avg_diff < 0.002:
        print("\nRecommendation: The model has plateaued. You can likely use 10-15 epochs.")
    else:
        print("\nRecommendation: The model is still improving. 20 epochs is appropriate.")


def preview_ground_truth():
    """Create a preview image to verify labels before training."""
    print("\nGenerating Ground Truth preview...")
    dataset_root = os.path.join(BASE_DIR, "data", "digits_data")
    preview_img_path = os.path.join(OUTPUT_DIR, "preview_labels_before_training.png")
    
    all_samples = iter_new_samples(dataset_root)
    if not all_samples:
        print("Warning: No samples found for preview.")
        return preview_img_path
        
    categories = list(set([s['category'] for s in all_samples]))
    samples_per_cat = 2
    
    fig, axes = plt.subplots(len(categories), samples_per_cat, figsize=(10, 8))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    for i, cat in enumerate(categories):
        cat_samples = [s for s in all_samples if s['category'] == cat]
        if len(cat_samples) < samples_per_cat:
            selected = cat_samples
        else:
            selected = random.sample(cat_samples, samples_per_cat)
            
        for j, s in enumerate(selected):
            img = cv2.imread(s['image_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = axes[i, j] if len(categories) > 1 else axes[j]
            ax.imshow(img)
            
            global_boxes, _, _, _ = get_gt_from_anno(s['anno_path'])
            for x1, y1, x2, y2 in global_boxes:
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=3))
            
            ax.set_title(f"Preview: {cat}")
            ax.axis('off')
            
    plt.suptitle("SANITY CHECK: Ground Truth Labels Before Training", fontsize=16)
    plt.savefig(preview_img_path, bbox_inches='tight')
    plt.close()
    return preview_img_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Skip training and only run inference to generate analysis.")
    parser.add_argument("--analyze-only", action="store_true", help="Skip both training and inference; only summarize existing CSVs and generate the image.")
    args = parser.parse_args()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Preview step for a quick sanity check.
    preview_path = preview_ground_truth()
    print(f"\n=> SANITY CHECK: Opening preview image...")
    # On Windows, this will automatically launch the image viewer
    if os.name == 'nt':
        os.startfile(preview_path)
    else:
        import subprocess
        subprocess.run(['open', preview_path] if sys.platform == 'darwin' else ['xdg-open', preview_path])
        
    # Give the user a moment to look at it before heavy processing starts
    import time
    time.sleep(2)

    if not args.analyze_only:
        # 1. Run GlobalBB training and inference.
        print("Step 1/3: Running GlobalBB Training & Inference...")
        
        globalbb_args = [
            "--dataset-root", os.path.join(BASE_DIR, "data", "digits_data"),
            "--output-dir", OUTPUT_DIR,
            "--epochs", "20",
            "--overwrite-conversion"
        ]
        if args.skip_train:
            globalbb_args.append("--skip-train")

        run_quiet_script("globalbb_detector.py", globalbb_args)
    else:
        print("Step 1/3: Skipping GlobalBB Training & Inference (--analyze-only enabled)...")

    # 2. Print final GlobalBB metrics from results.csv.
    results_csv = os.path.join(GlobalBB_RUN_DIR, "results.csv")
    if os.path.exists(results_csv):
        rdf = pd.read_csv(results_csv)
        rdf.columns = rdf.columns.str.strip()
        last = rdf.iloc[-1]
        print("\n=== GlobalBB Final Performance Summary ===")
        print(f"Overall mAP50:  {last['metrics/mAP50(B)']:.2%}")
        print(f"Precision:      {last['metrics/precision(B)']:.2%}")
        print(f"Recall:         {last['metrics/recall(B)']:.2%}")
        
        analyze_epochs(results_csv)

    # 3. Analyze predictions by category.
    pred_csv = os.path.join(OUTPUT_DIR, "globalbb_predictions.csv")
    if os.path.exists(pred_csv):
        pdf = pd.read_csv(pred_csv)
        print("\n=== Accuracy per Category (Average Confidence) ===")
        stats = pdf.groupby('category')['pred_conf'].agg(['mean', 'count'])
        for cat, row in stats.iterrows():
            print(f"- {cat:12}: {row['mean']:.2%} (Total samples: {int(row['count'])})")

    # 4. Generate visualization output.
    print("\nStep 3/3: Generating Comparison Images...")
    run_quiet_script("visualize_globalbb_results.py")
    print(f"Process Complete. Check: {os.path.join(OUTPUT_DIR, 'globalbb_comparison_summary.png')}")

if __name__ == "__main__":
    main()