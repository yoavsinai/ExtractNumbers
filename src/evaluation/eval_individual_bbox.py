import os
import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import enhance_digit
from utils.data_utils import iter_new_samples, get_gt_from_anno
from utils.metrics import calculate_iou

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 3: Individual Bounding Box Evaluation")
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    # Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    INDIV_MODEL_PATH = os.path.join(TRAINED_DIR, "individualbb.pt")
    DATA_ROOT = os.path.join(BASE_DIR, "data", "digits_data")
    REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    if not os.path.exists(INDIV_MODEL_PATH):
        print(f"❌ Error: Individual model not found at {INDIV_MODEL_PATH}")
        sys.exit(1)

    print("\n--- Stage 3: Individual Bounding Box Evaluation ---")
    model = YOLO(INDIV_MODEL_PATH)
    
    all_samples = list(iter_new_samples(DATA_ROOT))
    import random
    random.seed(42)
    
    # Exclude categories without individual digit annotations
    excluded_categories = ['race_numbers', 'ocr_trains']
    filtered_samples = [s for s in all_samples if s['category'] not in excluded_categories]
    
    random.shuffle(filtered_samples)
    eval_samples = filtered_samples[:args.max_samples]
    
    results = []
    all_ious = []
    all_confs = []
    
    for s in tqdm(eval_samples, desc="Evaluating Individual BB"):
        img = cv2.imread(s['image_path'])
        if img is None: continue
        
        # Use GT Global BB to isolate Stage 3
        gt_global_boxes, digit_info, _, has_digits = get_gt_from_anno(s['anno_path'])
        if not digit_info or not has_digits: continue
        
        gx1, gy1, gx2, gy2 = map(int, gt_global_boxes[0])
        h, w = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        if crop.size == 0: continue
            
        sharp = enhance_digit(crop, upscale_factor=2.0)
        
        res = model.predict(source=sharp, imgsz=256, verbose=False)
        pred_indiv_boxes = []
        s3_ious = []
        
        if res and len(res[0].boxes) > 0:
            pred_indiv_boxes = res[0].boxes.xyxy.cpu().numpy()
            pred_confs = res[0].boxes.conf.cpu().numpy()
            
            for idx, digit in enumerate(digit_info):
                dx1, dy1, dx2, dy2 = digit['bbox']
                # Map to sharpened crop coordinate
                nx1, ny1 = (dx1 - gx1) * 2.0, (dy1 - gy1) * 2.0
                nx2, ny2 = (dx2 - gx1) * 2.0, (dy2 - gy1) * 2.0
                gt_box_sharp = (nx1, ny1, nx2, ny2)
                
                best_iou = 0
                best_conf = 0
                for p_idx, pbox in enumerate(pred_indiv_boxes):
                    iou = calculate_iou(gt_box_sharp, pbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_conf = pred_confs[p_idx]
                
                s3_ious.append(best_iou)
                all_ious.append(best_iou)
                all_confs.append(best_conf)

        results.append({
            'sample_id': s['sample_id'],
            'category': s['category'],
            'avg_iou': np.mean(s3_ious) if s3_ious else 0,
            'num_gt_digits': len(digit_info),
            'num_pred_digits': len(pred_indiv_boxes),
            'recall': len(s3_ious) / len(digit_info) if len(digit_info) > 0 else 0
        })

    df = pd.DataFrame(results)
    
    # Global Metrics
    mean_iou = np.mean(all_ious) if all_ious else 0
    mean_conf = np.mean(all_confs) if all_confs else 0
    total_recall = df['num_pred_digits'].sum() / df['num_gt_digits'].sum() if df['num_gt_digits'].sum() > 0 else 0
    
    print("\n" + "="*40)
    print("📊 STAGE 3: INDIVIDUAL BB METRICS")
    print("="*40)
    print(f"Mean IoU (all digits): {mean_iou:.4f}")
    print(f"Mean Confidence:       {mean_conf:.4f}")
    print(f"Overall Recall:        {total_recall:.2%}")
    print(f"Total GT Digits:       {df['num_gt_digits'].sum()}")
    print(f"Total Pred Digits:     {df['num_pred_digits'].sum()}")
    
    print("\n📈 PERFORMANCE BY CATEGORY:")
    cat_metrics = df.groupby('category').agg({
        'avg_iou': 'mean',
        'num_gt_digits': 'sum',
        'num_pred_digits': 'sum'
    })
    cat_metrics['Recall'] = cat_metrics['num_pred_digits'] / cat_metrics['num_gt_digits']
    print(cat_metrics[['avg_iou', 'Recall']])

    # Save report
    csv_path = os.path.join(REPORTS_DIR, "stage3_individual_bbox_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed report saved to: {csv_path}")

if __name__ == "__main__":
    main()
