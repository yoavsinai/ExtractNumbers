import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop
from image_preprocessing.digit_preprocessor import enhance_digit
from utils.data_utils import iter_new_samples, get_gt_from_anno
from utils.metrics import calculate_iou

def get_full_gt_number(anno_path):
    """Reconstruct the number sequence from annotations.json."""
    with open(anno_path, 'r') as f:
        data = json.load(f)
    digits = []
    for number in data.get('detected_numbers', []):
        for digit in number.get('digits', []):
            digits.append({'x': digit['bounding_box']['x'], 'label': str(digit['label'])})
    digits.sort(key=lambda d: d['x'])
    return "".join([d['label'] for d in digits])

def calculate_digit_accuracy(gt, pred):
    """Calculate positioning accuracy and succession rate."""
    correct = 0
    total = len(gt)
    successions = 0
    possible_successions = 0
    
    for i in range(min(len(gt), len(pred))):
        if gt[i] == pred[i]:
            correct += 1
            # Check succession: if current is correct, is the next one correct?
            if i + 1 < min(len(gt), len(pred)):
                possible_successions += 1
                if gt[i+1] == pred[i+1]:
                    successions += 1
                    
    succession_rate = successions / possible_successions if possible_successions > 0 else 1.0
    return correct, total, succession_rate

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Pipeline Benchmark & Visualization")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--save-viz", action="store_true", help="Save the evaluation dashboard image")
    parser.add_argument("--analyze-errors", action="store_true", help="Generate detailed error analysis visualization")
    args = parser.parse_args()

    # Structured Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")
    REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    GLOBAL_MODEL_PATH = os.path.join(TRAINED_DIR, "globalbb.pt")
    INDIV_MODEL_PATH = os.path.join(TRAINED_DIR, "individualbb.pt")
    CLASSIFIER_PATH = os.path.join(TRAINED_DIR, "digit_classifier.pth")
    DATA_ROOT = os.path.join(BASE_DIR, "data", "digits_data")
    
    device = get_device()
    
    # 1. Load Models
    print("\n--- Phase 1: Loading Models ---")
    if not all([os.path.exists(p) for p in [GLOBAL_MODEL_PATH, INDIV_MODEL_PATH, CLASSIFIER_PATH]]):
        print("❌ Error: Missing trained model weights. Run the main pipeline first.")
        sys.exit(1)

    global_model = YOLO(GLOBAL_MODEL_PATH)
    indiv_model = YOLO(INDIV_MODEL_PATH)
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    print("✓ All models loaded successfully.")

    # 2. Prepare Samples
    samples = iter_new_samples(DATA_ROOT)
    import random
    random.seed(42)
    random.shuffle(samples)
    eval_samples = samples[:args.max_samples]
    
    results = []
    
    # 3. Main Evaluation Loop
    print(f"\n--- Phase 2: Running Pipeline Benchmark (N={len(eval_samples)}) ---")
    for s in tqdm(eval_samples):
        img_path = s['image_path']
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Ground Truth
        gt_global_boxes, digit_info = get_gt_from_anno(s['anno_path'])
        gt_number = get_full_gt_number(s['anno_path'])
        
        # -- Step 1: GlobalBB Detection --
        res1 = global_model.predict(source=img, imgsz=256, verbose=False)
        pred_global = None
        s1_iou = 0.0
        
        if res1 and len(res1[0].boxes) > 0:
            best_idx = res1[0].boxes.conf.argmax().item()
            pred_global = res1[0].boxes.xyxy[best_idx].cpu().numpy()
            
            if gt_global_boxes:
                s1_iou = calculate_iou(gt_global_boxes[0], pred_global)
        
        if pred_global is None:
            results.append({'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 'correct': False, 'category': s['category'], 's1_iou': 0, 'digit_acc': 0})
            continue
            
        gx1, gy1, gx2, gy2 = map(int, pred_global)
        h, w = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        
        if crop.size == 0:
            results.append({'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 'correct': False, 'category': s['category'], 's1_iou': s1_iou, 'digit_acc': 0})
            continue
            
        # -- Step 2: Sharpening --
        sharp = enhance_digit(crop, upscale_factor=2.0)
        
        # -- Step 3: IndividualBB Detection --
        res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
        pred_indiv_boxes = []
        s2_ious = []
        
        if res2 and len(res2[0].boxes) > 0:
            pred_indiv_boxes = res2[0].boxes.xyxy.cpu().numpy()
            pred_indiv_boxes = sorted(pred_indiv_boxes, key=lambda b: b[0])
            
            for digit in digit_info:
                dx1, dy1, dx2, dy2 = digit['bbox']
                nx1, ny1 = (dx1 - gx1) * 2.0, (dy1 - gy1) * 2.0
                nx2, ny2 = (dx2 - gx1) * 2.0, (dy2 - gy1) * 2.0
                gt_box_sharp = (nx1, ny1, nx2, ny2)
                
                best_iou = 0
                for pbox in pred_indiv_boxes:
                    iou = calculate_iou(gt_box_sharp, pbox)
                    best_iou = max(best_iou, iou)
                s2_ious.append(best_iou)

        # -- Step 4: Classification & Assembly --
        predicted_digits = []
        pred_crops = []
        for ibox in pred_indiv_boxes:
            try:
                inputs = preprocess_crop(sharp, (ibox[0], ibox[1], ibox[2], ibox[3])).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = classifier(inputs)
                    digit = out.argmax(dim=1).item()
                    predicted_digits.append(str(digit))
                    
                    # For error analysis
                    ix1, iy1, ix2, iy2 = map(int, ibox)
                    d_crop = sharp[max(0,iy1):min(sharp.shape[0],iy2), max(0,ix1):min(sharp.shape[1],ix2)]
                    pred_crops.append(d_crop)
            except:
                continue
        
        pred_number = "".join(predicted_digits)
        correct_digits, total_gt_digits = calculate_digit_accuracy(gt_number, pred_number)
        
        results.append({
            'sample_id': s['sample_id'],
            'gt': gt_number,
            'pred': pred_number,
            'correct': pred_number == gt_number,
            'digit_acc': correct_digits / total_gt_digits if total_gt_digits > 0 else 0,
            'correct_digits': correct_digits,
            'total_digits': total_gt_digits,
            'category': s['category'],
            's1_iou': s1_iou,
            's2_iou_avg': np.mean(s2_ious) if s2_ious else 0,
            # Data for visualization
            'vis_img': img,
            'vis_crop': crop,
            'vis_sharp': sharp,
            'vis_gx': (gx1, gy1, gx2, gy2),
            'vis_iboxes': pred_indiv_boxes,
            'vis_pred_crops': pred_crops,
            'vis_preds': predicted_digits
        })

    # 4. Reporting
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 FINAL PIPELINE BENCHMARK")
    print("="*50)
    print(f"Full Sequence Accuracy:       {df['correct'].mean():.2%}")
    print(f"Mean Digit Accuracy (Pos):    {df['digit_acc'].mean():.2%}")
    print(f"Single Digit Succession Rate: {df['succession_rate'].mean():.2%}")
    print(f"Stage 1 (Global) Mean IoU:    {df['s1_iou'].mean():.4f}")
    print(f"Stage 3 (Indiv)  Mean IoU:    {df['s2_iou_avg'].mean():.4f}")
    
    print("\n📈 PERFORMANCE BY CATEGORY:")
    cat_stats = df.groupby('category').agg({
        'correct': 'mean',
        'digit_acc': 'mean',
        'succession_rate': 'mean',
        's1_iou': 'mean',
        's2_iou_avg': 'mean'
    }).rename(columns={
        'correct': 'Seq Acc',
        'digit_acc': 'Digit Acc',
        'succession_rate': 'Succ Rate',
        's1_iou': 'S1 IoU',
        's2_iou_avg': 'S2 IoU'
    })
    print(cat_stats)

    # 5. Dashboard Generation
    if args.save_viz:
        print("\n--- Generating Dashboard ---")
        viz_samples = []
        successes = [r for r in results if r['correct']]
        failures = [r for r in results if not r['correct']]
        viz_samples.extend(successes[:2])
        viz_samples.extend(failures[:2])
        
        fig, axes = plt.subplots(len(viz_samples), 4, figsize=(22, 5 * len(viz_samples)))
        if len(viz_samples) == 1: axes = axes.reshape(1, -1)
        
        for i, res in enumerate(viz_samples):
            img_rgb = cv2.cvtColor(res['vis_img'], cv2.COLOR_BGR2RGB)
            x1,y1,x2,y2 = res['vis_gx']
            cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (255, 0, 0), 4)
            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f"1. Global (IoU: {res['s1_iou']:.2f})")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(cv2.cvtColor(res['vis_crop'], cv2.COLOR_BGR2RGB))
            axes[i, 1].set_title("2. Raw Crop")
            axes[i, 1].axis('off')

            sharp_rgb = cv2.cvtColor(res['vis_sharp'], cv2.COLOR_BGR2RGB)
            for ibox in res['vis_iboxes']:
                 cv2.rectangle(sharp_rgb, (int(ibox[0]), int(ibox[1])), (int(ibox[2]), int(ibox[3])), (255, 255, 0), 2)
            axes[i, 2].imshow(sharp_rgb)
            axes[i, 2].set_title("3. Individual Detection")
            axes[i, 2].axis('off')
            
            axes[i, 3].axis('off')
            color = "green" if res['correct'] else "red"
            txt = f"GT:   {res['gt']}\nPred: {res['pred']}\n\nCategory: {res['category']}"
            axes[i, 3].text(0.1, 0.5, txt, fontsize=14, fontweight='bold', color=color, verticalalignment='center')

        plt.suptitle("FULL PIPELINE PERFORMANCE DASHBOARD", fontsize=20, fontweight='bold', y=0.98)
        plt.savefig(os.path.join(VIS_DIR, "pipeline_dashboard.png"), bbox_inches='tight', dpi=120)

    # 6. Error Analysis Visualization (Merged from visualize_error_analysis.py)
    if args.analyze_errors:
        print("\n--- Generating Detailed Error Analysis ---")
        failures = [r for r in results if not r['correct']][:4]
        if not failures: failures = results[:4] # Fallback if no failures
        
        fig = plt.figure(figsize=(24, 18))
        for i, res in enumerate(failures):
             # 1. Original + Global
             ax1 = plt.subplot(4, 5, i*5 + 1)
             img_rgb = cv2.cvtColor(res['vis_img'], cv2.COLOR_BGR2RGB)
             x1,y1,x2,y2 = res['vis_gx']
             cv2.rectangle(img_rgb, (x1,y1), (x2,y2), (255, 0, 0), 4)
             ax1.imshow(img_rgb); ax1.axis('off')
             ax1.set_title(f"Original + Global", color="red", fontweight='bold')
             
             # 2. Raw Crop
             ax2 = plt.subplot(4, 5, i*5 + 2)
             ax2.imshow(cv2.cvtColor(res['vis_crop'], cv2.COLOR_BGR2RGB)); ax2.axis('off')
             ax2.set_title("Raw Crop")
             
             # 3. Sharpened + Individual
             ax3 = plt.subplot(4, 5, i*5 + 3)
             sharp_rgb = cv2.cvtColor(res['vis_sharp'], cv2.COLOR_BGR2RGB)
             for bx in res['vis_iboxes']:
                 cv2.rectangle(sharp_rgb, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (255, 255, 0), 2)
             ax3.imshow(sharp_rgb); ax3.axis('off')
             ax3.set_title("Sharpened + Individual")
             
             # 4. Classification Strip
             ax4 = plt.subplot(4, 5, i*5 + 4)
             if res['vis_pred_crops']:
                 strip = np.hstack([cv2.resize(c, (64, 64)) for c in res['vis_pred_crops'] if c.size > 0])
                 ax4.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
                 ax4.set_title(f"Preds: {' '.join(res['vis_preds'])}")
             ax4.axis('off')
             
             # 5. Summary
             ax5 = plt.subplot(4, 5, i*5 + 5); ax5.axis('off')
             txt = f"GT:   {res['gt']}\nPred: {res['pred']}\nCat:  {res['category']}"
             ax5.text(0.1, 0.5, txt, fontsize=14, fontweight='bold', verticalalignment='center')

        plt.suptitle("DETAILED ERROR ANALYSIS", fontsize=22, fontweight='bold', y=0.98)
        plt.savefig(os.path.join(VIS_DIR, "detailed_error_analysis.png"), bbox_inches='tight', dpi=120)

    # Save CSV
    df_mini = df.drop(columns=[c for c in df.columns if c.startswith('vis_')])
    df_mini.to_csv(os.path.join(REPORTS_DIR, "pipeline_metrics.csv"), index=False)
    print(f"\n💾 Detailed results saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()
