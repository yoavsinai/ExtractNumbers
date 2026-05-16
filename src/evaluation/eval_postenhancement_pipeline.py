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
from image_preprocessing import get_enhancer
from utils.data_utils import iter_new_samples, get_gt_from_anno
from utils.metrics import calculate_iou

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
    parser = argparse.ArgumentParser(description="Enhancement Pipeline Benchmark & Visualization")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--save-viz", action="store_true", help="Save the evaluation dashboard image")
    parser.add_argument("--analyze-errors", action="store_true", help="Generate detailed error analysis visualization")
    parser.add_argument("--data-root", type=str, default=os.path.join(BASE_DIR, "data", "digits_data"), help="Path to the dataset root")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"), help="Base directory for outputs")
    parser.add_argument("--enhancement", type=str, default="unsharp_mask", choices=["none", "unsharp_mask", "clahe", "esrgan", "edsr", "lapsrn", "realcugan", "bsrgan", "swiniR", "diffusion", "opencv"], help="Choose the sharpening/enhancement model ('opencv' is an alias for 'unsharp_mask')")
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
    
    # 1. Load Models and Enhancer
    print("\n--- Phase 1: Loading Models & Enhancer ---")
    if not all([os.path.exists(p) for p in [GLOBAL_MODEL_PATH, INDIV_MODEL_PATH, CLASSIFIER_PATH]]):
        print("❌ Error: Missing trained model weights. Run the main pipeline first.")
        sys.exit(1)

    global_model = YOLO(GLOBAL_MODEL_PATH)
    indiv_model = YOLO(INDIV_MODEL_PATH)
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    print("✓ Models loaded successfully.")

    # Initialize the selected enhancer model once
    print(f"✓ Initializing enhancement method: {args.enhancement}")
    enhancer = get_enhancer(args.enhancement, scale_factor=2.0)

    # 2. Prepare Samples
    print("\nPreparing samples...")
    all_samples = list(iter_new_samples(DATA_ROOT))
    
    # Group by category for balanced sampling
    from collections import defaultdict
    samples_by_cat = defaultdict(list)
    excluded_categories = ['race_number', 'race_numbers', 'ocr_train', 'ocr_trains']
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
            samples_per_cat = max(1, int(round(args.max_samples * (len(samps) / total_samples))))
            eval_samples.extend(samps[:samples_per_cat])
            
        # Shuffle the final evaluation list to mix datasets during processing
        random.shuffle(eval_samples)
    
    results = []
    
    # 3. Main Evaluation Loop
    print(f"\n--- Phase 2: Running Enhancement Benchmark (N={len(eval_samples)}) ---")
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
            best_idx = res1[0].boxes.conf.argmax().item()
            pred_global = res1[0].boxes.xyxy[best_idx].cpu().numpy()
            
            if gt_global_boxes:
                s1_iou = calculate_iou(gt_global_boxes[0], pred_global)
        
        if pred_global is None:
            digit_pairs = [(g, 'N') for g in gt_number] if has_label else []
            results.append({
                'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 
                'correct': False if has_label else None, 
                'category': s['category'], 's1_iou': 0, 'digit_acc': 0 if has_digit_boxes else None,
                'has_digit_boxes': has_digit_boxes, 'has_label': has_label, 'digit_pairs': digit_pairs
            })
            continue
            
        gx1, gy1, gx2, gy2 = map(int, pred_global)
        h, w = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        
        if crop.size == 0:
            digit_pairs = [(g, 'N') for g in gt_number] if has_label else []
            results.append({
                'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 
                'correct': False if has_label else None, 
                'category': s['category'], 's1_iou': s1_iou, 'digit_acc': 0 if has_digit_boxes else None,
                'has_digit_boxes': has_digit_boxes, 'has_label': has_label, 'digit_pairs': digit_pairs
            })
            continue
            
        # -- Step 2: No sharpening yet --
        sharp = crop
        scale = 1.0
        
        # -- Step 3: IndividualBB Detection --
        res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
        pred_indiv_boxes = []
        s2_ious = []
        
        if res2 and len(res2[0].boxes) > 0:
            pred_indiv_boxes = res2[0].boxes.xyxy.cpu().numpy()
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

        # -- Step 4: Classification & Assembly (Post-Enhancement) --
        predicted_digits = []
        pred_crops = []
        for ibox in pred_indiv_boxes:
            try:
                # 1. Crop the raw digit out
                ix1, iy1, ix2, iy2 = map(int, ibox)
                digit_crop = sharp[max(0,iy1):min(sharp.shape[0],iy2), max(0,ix1):min(sharp.shape[1],ix2)]
                
                # 2. Enhance just this specific digit
                if digit_crop.size > 0:
                    sharp_digit = enhancer.enhance(digit_crop)
                else:
                    sharp_digit = digit_crop
                
                # 3. Classify
                if sharp_digit.size > 0:
                    h, w = sharp_digit.shape[:2]
                    inputs = preprocess_crop(sharp_digit, (0, 0, w, h)).unsqueeze(0).to(device)
                else:
                    inputs = preprocess_crop(sharp, (ibox[0], ibox[1], ibox[2], ibox[3])).unsqueeze(0).to(device)
                    
                with torch.no_grad():
                    out = classifier(inputs)
                    digit = out.argmax(dim=1).item()
                    predicted_digits.append(str(digit))
                    pred_crops.append(sharp_digit)
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
            'sample_id': s['sample_id'], 'gt': gt_number if has_label else "N/A", 'pred': pred_number,
            'correct': (pred_number == gt_number) if has_label else None,
            'digit_acc': (correct_digits / total_gt_digits) if has_digit_boxes and total_gt_digits > 0 else None,
            'succession_rate': succession_rate if has_digit_boxes else None,
            'category': s['category'], 's1_iou': s1_iou,
            's2_iou_avg': np.mean(s2_ious) if s2_ious and has_digit_boxes else None,
            'has_digit_boxes': has_digit_boxes, 'has_label': has_label, 'digit_pairs': digit_pairs,
            'vis_img': img, 'vis_crop': crop, 'vis_sharp': sharp,
            'vis_gx': (gx1, gy1, gx2, gy2), 'vis_iboxes': pred_indiv_boxes,
            'vis_pred_crops': pred_crops, 'vis_preds': predicted_digits
        })

    # 4. Reporting
    df = pd.DataFrame(results)
    report_lines = []
    def log_print(text=""):
        print(text)
        report_lines.append(str(text))
        
    log_print("\n" + "="*50)
    log_print("📊 DETAILED CLASSIFICATION REPORT")
    log_print("="*50)
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
            log_print(f"\n{cat}")
            labels = [str(i) for i in range(10)]
            log_print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
            
    log_print("\n" + "="*50)
    log_print(f"📊 ENHANCEMENT BENCHMARK: {args.enhancement.upper()}")
    log_print("="*50)
    log_print(f"Full Sequence Accuracy:       {df[df['has_label'] == True]['correct'].mean():.2%}")
    log_print(f"Mean Digit Accuracy (Pos):    {df[df['has_digit_boxes'] == True]['digit_acc'].mean():.2%}")
    log_print(f"Stage 1 (Global) Mean IoU:    {df['s1_iou'].mean():.4f} (All Samples)")
    log_print(f"Stage 3 (Indiv)  Mean IoU:    {df[df['has_digit_boxes'] == True]['s2_iou_avg'].mean():.4f}")
    log_print(f"Succession Rate:              {df[df['has_digit_boxes'] == True]['succession_rate'].mean():.2%}")
    
    log_print("\n📈 PERFORMANCE BY CATEGORY:")
    cat_stats = pd.DataFrame({
        'Seq Acc': df[df['has_label'] == True].groupby('category')['correct'].mean(),
        'Digit Acc': df[df['has_digit_boxes'] == True].groupby('category')['digit_acc'].mean(),
        'Succ Rate': df[df['has_digit_boxes'] == True].groupby('category')['succession_rate'].mean(),
        'S1 IoU': df.groupby('category')['s1_iou'].mean(),
        'S2 IoU': df[df['has_digit_boxes'] == True].groupby('category')['s2_iou_avg'].mean(),
        'Count': df.groupby('category').size(),
        'Labeled': df.groupby('category')['has_digit_boxes'].sum()
    }).reindex(df['category'].unique())

    cat_stats['Count'] = cat_stats['Count'].fillna(0).astype(int)
    cat_stats['Labeled'] = cat_stats['Labeled'].fillna(0).astype(int)
    for col in ['Seq Acc', 'Digit Acc', 'Succ Rate', 'S1 IoU', 'S2 IoU']:
        cat_stats[col] = cat_stats[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    log_print(cat_stats.to_string())
    
    report_txt_path = os.path.join(REPORTS_DIR, f"postenhancement_summary_{args.enhancement}.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📝 Text report saved to: {report_txt_path}")

    if args.save_viz:
        print("\n--- Generating Dashboards ---")
        # Code for dashboard omitted for brevity in this specific evaluation run, 
        # but can be easily re-enabled from eval_pipeline if needed.
        pass
        
    df_mini = df.drop(columns=[c for c in df.columns if c.startswith('vis_')])
    df_mini.to_csv(os.path.join(REPORTS_DIR, f"postenhancement_metrics_{args.enhancement}.csv"), index=False)
    print(f"\n💾 Detailed results saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()