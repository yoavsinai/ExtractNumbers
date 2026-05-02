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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Global Bounding Box Evaluation")
    parser.add_argument("--max-samples", type=int, default=500)
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
    import random
    random.seed(42)
    random.shuffle(all_samples)
    eval_samples = all_samples[:args.max_samples]
    
    results = []
    ious = []
    
    for s in tqdm(eval_samples, desc="Evaluating Global BB"):
        img = cv2.imread(s['image_path'])
        if img is None: continue
        
        global_boxes, _, _, _ = get_gt_from_anno(s['anno_path'])
        if not global_boxes: continue
        
        res = model.predict(source=img, imgsz=256, verbose=False)
        pred_global = None
        iou = 0.0
        conf = 0.0
        
        if res and len(res[0].boxes) > 0:
            best_idx = res[0].boxes.conf.argmax().item()
            pred_global = res[0].boxes.xyxy[best_idx].cpu().numpy()
            conf = res[0].boxes.conf[best_idx].item()
            
            if global_boxes:
                iou = calculate_iou(global_boxes[0], pred_global)
                ious.append(iou)
        
        results.append({
            'sample_id': s['sample_id'],
            'category': s['category'],
            'iou': iou,
            'confidence': conf,
            'detected': pred_global is not None,
            'hit_05': iou >= 0.5,
            'hit_075': iou >= 0.75
        })

    df = pd.DataFrame(results)
    
    # Global Metrics
    mean_iou = df['iou'].mean()
    detection_rate = df['detected'].mean()
    accuracy_05 = df['hit_05'].mean()
    accuracy_075 = df['hit_075'].mean()
    mean_conf = df[df['detected']]['confidence'].mean()
    
    print("\n" + "="*40)
    print("📊 STAGE 1: GLOBAL BB METRICS")
    print("="*40)
    print(f"Mean IoU:           {mean_iou:.4f}")
    print(f"Detection Rate:     {detection_rate:.2%}")
    print(f"mAP@0.5 (IoU >= 0.5): {accuracy_05:.2%}")
    print(f"mAP@0.75 (IoU >= 0.75): {accuracy_075:.2%}")
    print(f"Mean Confidence:    {mean_conf:.4f}")
    
    print("\n📈 PERFORMANCE BY CATEGORY:")
    cat_metrics = df.groupby('category').agg({
        'iou': 'mean',
        'detected': 'mean',
        'hit_05': 'mean',
        'confidence': 'mean'
    }).rename(columns={
        'iou': 'Mean IoU',
        'detected': 'Det Rate',
        'hit_05': 'Acc@0.5',
        'confidence': 'Avg Conf'
    })
    print(cat_metrics)

    # Save report
    csv_path = os.path.join(REPORTS_DIR, "stage1_global_bbox_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed report saved to: {csv_path}")

if __name__ == "__main__":
    main()
