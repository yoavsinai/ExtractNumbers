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
from sklearn.metrics import classification_report

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop
from image_preprocessing.digit_preprocessor import enhance_digit
from utils.data_utils import iter_new_samples, get_gt_from_anno
from utils.metrics import calculate_iou
from utils.bbox_utils import merge_global_boxes, nms_individual_boxes

# Redundant get_full_gt_number removed as get_gt_from_anno now returns the full label.

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
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--save-viz", action="store_true", help="Save the evaluation dashboard image")
    parser.add_argument("--analyze-errors", action="store_true", help="Generate detailed error analysis visualization")
    parser.add_argument("--data-root", type=str, default=os.path.join(BASE_DIR, "data", "digits_data"), help="Path to the dataset root")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"), help="Base directory for outputs")
    # parser.add_argument("--enhancement"... removed to test both)
    args = parser.parse_args()

    # Structured Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    VIS_DIR = os.path.join(args.output_dir, "visualizations")
    REPORTS_DIR = os.path.join(args.output_dir, "reports")
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    GLOBAL_MODEL_PATH = os.path.join(TRAINED_DIR, "globalbb.pt")
    INDIV_MODEL_PATH = os.path.join(TRAINED_DIR, "individualbb.pt")
    CLASSIFIER_PATH = os.path.join(TRAINED_DIR, "digit_recognizer.pt")
    DATA_ROOT = args.data_root
    
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

    # 2. Prepare Samples - Convert iterator to list first
    print("\nPreparing samples...")
    all_samples = list(iter_new_samples(DATA_ROOT))
    
    # Group by category for balanced sampling
    from collections import defaultdict
    samples_by_cat = defaultdict(list)
    excluded_categories = ['race_number', 'race_numbers', 'ocr_train', 'ocr_trains'] #since they are dont have ground trouth to check it
    for s in all_samples:
        if s['category'] not in excluded_categories:
            samples_by_cat[s['category']].append(s)
        
    import random
    random.seed(42)
    eval_samples = []
    
    if samples_by_cat:
        total_samples = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            samples_per_cat = int(args.max_samples * (len(samps) / total_samples))
            eval_samples.extend(samps[:samples_per_cat])
            
        # Shuffle the final evaluation list to mix datasets during processing
        random.shuffle(eval_samples)
    
    results = []
    
    # 3. Main Evaluation Loop
    print(f"\n--- Phase 2: Running Pipeline Benchmark (N={len(eval_samples)}) ---")
    for s in tqdm(eval_samples):
        img_path = s['image_path']
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Ground Truth
        gt_global_boxes, digit_info, has_digit_boxes, gt_number = get_gt_from_anno(s['anno_path'])
        has_label = bool(gt_number)
        
        # -- Step 1: GlobalBB Detection --
        res1 = global_model.predict(source=img, imgsz=256, verbose=False)
        pred_global = None
        s1_iou = 0.0
        
        if res1 and len(res1[0].boxes) > 0:
            all_gboxes = res1[0].boxes.xyxy.cpu().numpy()
            pred_global = merge_global_boxes(all_gboxes)
            
            if gt_global_boxes:
                s1_iou = calculate_iou(gt_global_boxes[0], pred_global)
        
        if pred_global is None:
            digit_pairs = [(g, 'N') for g in gt_number] if has_label else []
            res_entry = {
                'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 
                'correct': False if has_label else None, 
                'category': s['category'], 's1_iou': 0, 'digit_acc': 0 if has_digit_boxes else None, 'enhancement': enh_method,
                'has_digit_boxes': has_digit_boxes,
                'has_label': has_label,
                'digit_pairs': digit_pairs,
                'vis_img': img,
                'vis_crop': img,
                'vis_sharp': img,
                'vis_gx': (0, 0, img.shape[1], img.shape[0]),
                'vis_iboxes': [],
                'vis_pred_crops': [],
                'vis_preds': []
            }
            results.append(res_entry)
            continue
            
        gx1, gy1, gx2, gy2 = map(int, pred_global)
        h, w = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        
        if crop.size == 0:
            digit_pairs = [(g, 'N') for g in gt_number] if has_label else []
            res_entry = {
                'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 
                'correct': False if has_label else None, 
                'category': s['category'], 's1_iou': s1_iou, 'digit_acc': 0 if has_digit_boxes else None, 'enhancement': enh_method,
                'has_digit_boxes': has_digit_boxes,
                'has_label': has_label,
                'digit_pairs': digit_pairs,
                'vis_img': img,
                'vis_crop': crop,
                'vis_sharp': crop,
                'vis_gx': (gx1, gy1, gx2, gy2),
                'vis_iboxes': [],
                'vis_pred_crops': [],
                'vis_preds': []
            }
            results.append(res_entry)
            continue
            
        for enh_method in ["none", "esrgan"]:
            # -- Step 2: Sharpening --
            sharp = enhance_digit(crop, upscale_factor=2.0, method=enh_method)
            scale = 2.0 if enh_method in ["esrgan", "opencv"] else 1.0
            
            # -- Step 3: IndividualBB Detection --
            res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
            pred_indiv_boxes = []
            s2_ious = []
            
            if res2 and len(res2[0].boxes) > 0:
                iboxes = res2[0].boxes.xyxy.cpu().numpy()
                iconfs = res2[0].boxes.conf.cpu().numpy()
                pred_indiv_boxes, _ = nms_individual_boxes(iboxes, iconfs, iou_thresh=0.45)
                pred_indiv_boxes = sorted(pred_indiv_boxes, key=lambda b: b[0])
                
                for digit in digit_info:
                    dx1, dy1, dx2, dy2 = digit['bbox']
                    nx1, ny1 = (dx1 - gx1) * scale, (dy1 - gy1) * scale
                    nx2, ny2 = (dx2 - gx1) * scale, (dy2 - gy1) * scale
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
            correct_digits, total_gt_digits, succession_rate = calculate_digit_accuracy(gt_number, pred_number)
            
            digit_pairs = []
            if has_label:
                for i in range(max(len(gt_number), len(pred_number))):
                    g = gt_number[i] if i < len(gt_number) else 'N'
                    p = pred_number[i] if i < len(pred_number) else 'N'
                    digit_pairs.append((g, p))

            results.append({
                'sample_id': s['sample_id'],
                'gt': gt_number if has_label else "N/A",
                'pred': pred_number,
                'correct': (pred_number == gt_number) if has_label else None,
                'digit_acc': (correct_digits / total_gt_digits) if has_digit_boxes and total_gt_digits > 0 else None,
                'succession_rate': succession_rate if has_digit_boxes else None,
                'correct_digits': correct_digits if has_digit_boxes else None,
                'total_digits': total_gt_digits if has_digit_boxes else None,
                'category': s['category'],
                'enhancement': enh_method,
                's1_iou': s1_iou,
                's2_iou_avg': np.mean(s2_ious) if s2_ious and has_digit_boxes else None,
                'has_digit_boxes': has_digit_boxes,
                'has_label': has_label,
                'digit_pairs': digit_pairs,
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
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*50)
    for cat in df['category'].unique():
        cat_df = df[(df['category'] == cat) & (df['has_label'] == True)]
        y_true = []
        y_pred = []
        for pairs in cat_df['digit_pairs']:
            if isinstance(pairs, list):
                for g, p in pairs:
                    y_true.append(g)
                    y_pred.append(p)
        
        if y_true:
            print(f"\n{cat}")
            labels = [str(i) for i in range(10)]
            print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
            
    print("\n" + "="*50)
    print("📊 FINAL PIPELINE BENCHMARK")
    print("="*50)
    print(f"Full Sequence Accuracy:       {df[df['has_label'] == True]['correct'].mean():.2%}")
    print(f"Mean Digit Accuracy (Pos):    {df[df['has_digit_boxes'] == True]['digit_acc'].mean():.2%}")
    # print(f"Single Digit Succession Rate: {df['succession_rate'].mean():.2%}") 
    print(f"Stage 1 (Global) Mean IoU:    {df['s1_iou'].mean():.4f} (All Samples)")
    print(f"Stage 3 (Indiv)  Mean IoU:    {df[df['has_digit_boxes'] == True]['s2_iou_avg'].mean():.4f}")
    
    print("\n📈 PERFORMANCE BY CATEGORY:")
    # Create the grouped stats efficiently without triggering DataFrameGroupBy.apply warnings
    cat_stats = pd.DataFrame({
        'Seq Acc': df[df['has_label'] == True].groupby('category')['correct'].mean(),
        'Digit Acc': df[df['has_digit_boxes'] == True].groupby('category')['digit_acc'].mean(),
        'S1 IoU': df.groupby('category')['s1_iou'].mean(),
        'S2 IoU': df[df['has_digit_boxes'] == True].groupby('category')['s2_iou_avg'].mean(),
        'Count': df.groupby('category').size(),
        'Labeled': df.groupby('category')['has_digit_boxes'].sum()
    }).reindex(df['category'].unique())

    cat_stats['Count'] = cat_stats['Count'].fillna(0).astype(int)
    cat_stats['Labeled'] = cat_stats['Labeled'].fillna(0).astype(int)
    
    # Format metrics cleanly, turning NaNs into 'N/A'
    for col in ['Seq Acc', 'Digit Acc', 'S1 IoU', 'S2 IoU']:
        cat_stats[col] = cat_stats[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

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
