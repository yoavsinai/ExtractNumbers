import os
import sys
import pandas as pd
import numpy as np
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils.data_utils import iter_new_samples, get_gt_from_anno
from utils.metrics import calculate_iou
from utils.bbox_utils import merge_global_boxes

def main():
    import argparse
    import random
    from collections import defaultdict
    parser = argparse.ArgumentParser(description="Stage 1: Global Bounding Box Evaluation")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    # Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    GLOBAL_MODEL_PATH = os.path.join(TRAINED_DIR, "globalbb.pt")
    DATA_ROOT = os.path.join(BASE_DIR, "data", "digits_data")
    REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not os.path.exists(GLOBAL_MODEL_PATH):
        print(f"❌ Error: Global model not found at {GLOBAL_MODEL_PATH}")
        sys.exit(1)

    print("\n--- Stage 1: Global Bounding Box Evaluation ---")
    model = YOLO(GLOBAL_MODEL_PATH)

    all_samples = list(iter_new_samples(DATA_ROOT))

    # Balanced sampling across categories (exclude synthetic)
    excluded = ['race_number', 'race_numbers', 'ocr_train', 'ocr_trains']
    samples_by_cat = defaultdict(list)
    for s in all_samples:
        if s['category'] not in excluded:
            samples_by_cat[s['category']].append(s)

    random.seed(42)
    eval_samples = []
    if samples_by_cat:
        total_samples = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            per_cat = max(1, int(round(args.max_samples * (len(samps) / total_samples))))
            eval_samples.extend(samps[:per_cat])
        random.shuffle(eval_samples)

    print(f"Evaluating {len(eval_samples)} samples across categories: {list(samples_by_cat.keys())}")

    results = []

    for s in tqdm(eval_samples, desc="Evaluating Global BB"):
        img = cv2.imread(s['image_path'])
        if img is None:
            continue

        global_boxes, _, _, _ = get_gt_from_anno(s['anno_path'])
        if not global_boxes:
            continue

        res = model.predict(source=img, imgsz=256, verbose=False)
        pred_global = None
        iou = 0.0
        conf = 0.0

        if res and len(res[0].boxes) > 0:
            all_gboxes = res[0].boxes.xyxy.cpu().numpy()
            pred_global = merge_global_boxes(all_gboxes)
            conf = res[0].boxes.conf.max().item()
            if global_boxes:
                iou = calculate_iou(global_boxes[0], pred_global)

        results.append({
            'sample_id':  s['sample_id'],
            'category':   s['category'],
            'iou':        iou,
            'confidence': conf,
            'detected':   pred_global is not None,
            'hit_05':     iou >= 0.5,
            'hit_075':    iou >= 0.75,
        })

    df = pd.DataFrame(results)

    report_lines = []
    def log_print(text=""):
        print(text)
        report_lines.append(str(text))

    log_print("\n" + "="*50)
    log_print("📊 STAGE 1: GLOBAL BOUNDING BOX METRICS")
    log_print("="*50)

    # --- Overall metrics ---
    detected_df = df[df['detected'] == True]
    overall_map50  = df['hit_05'].mean()
    overall_prec   = detected_df['hit_05'].sum() / len(detected_df) if len(detected_df) > 0 else 0.0
    overall_recall = df['detected'].mean()
    overall_iou    = df['iou'].mean()

    log_print(f"Overall mAP@0.5:  {overall_map50:.2%}")
    log_print(f"Overall Precision:{overall_prec:.2%}")
    log_print(f"Overall Recall:   {overall_recall:.2%}")
    log_print(f"Overall Mean IoU: {overall_iou:.4f}")

    log_print("\n📈 PERFORMANCE BY CATEGORY:")
    log_print(f"{'Category':<15} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10} {'Mean IoU':>10} {'Count':>7}")
    log_print("-" * 65)

    for cat in sorted(df['category'].unique()):
        c = df[df['category'] == cat]
        c_det = c[c['detected'] == True]
        map50  = c['hit_05'].mean()
        prec   = c_det['hit_05'].sum() / len(c_det) if len(c_det) > 0 else 0.0
        recall = c['detected'].mean()
        miou   = c['iou'].mean()
        log_print(f"{cat:<15} {map50:>10.2%} {prec:>10.2%} {recall:>10.2%} {miou:>10.4f} {len(c):>7}")

    log_print(f"\n{'OVERALL':<15} {overall_map50:>10.2%} {overall_prec:>10.2%} {overall_recall:>10.2%} {overall_iou:>10.4f} {len(df):>7}")

    # Save reports
    report_txt_path = os.path.join(REPORTS_DIR, "stage1_global_bbox_summary.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📝 Text report saved to: {report_txt_path}")

    csv_path = os.path.join(REPORTS_DIR, "stage1_global_bbox_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Detailed CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()

