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
from utils.metrics import calculate_iou, print_metrics_report

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
    """Calculate how many digits match at the same position."""
    correct = 0
    # Match up to the shortest length
    for a, b in zip(gt, pred):
        if a == b:
            correct += 1
    return correct, len(gt)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Pipeline Benchmark & Visualization")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--save-viz", action="store_true", help="Save the evaluation dashboard image")
    args = parser.parse_args()

    # Structured Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")
    REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
    
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
    stage1_ious = []
    stage2_ious = []
    
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
            # Take the box with highest confidence
            best_idx = res1[0].boxes.conf.argmax().item()
            pred_global = res1[0].boxes.xyxy[best_idx].cpu().numpy()
            
            # IoU with first GT global box (usually only one)
            if gt_global_boxes:
                s1_iou = calculate_iou(gt_global_boxes[0], pred_global)
                stage1_ious.append(s1_iou)
        
        if pred_global is None:
            results.append({'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 'correct': False, 'category': s['category'], 's1_iou': 0})
            continue
            
        gx1, gy1, gx2, gy2 = map(int, pred_global)
        h, w = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        
        if crop.size == 0:
            results.append({'sample_id': s['sample_id'], 'gt': gt_number, 'pred': '', 'correct': False, 'category': s['category'], 's1_iou': s1_iou})
            continue
            
        # -- Step 2: Sharpening --
        sharp = enhance_digit(crop, upscale_factor=2.0)
        
        # -- Step 3: IndividualBB Detection --
        res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
        pred_indiv_boxes = []
        s2_ious = []
        
        if res2 and len(res2[0].boxes) > 0:
            pred_indiv_boxes = res2[0].boxes.xyxy.cpu().numpy()
            # Sort by x for number assembly
            pred_indiv_boxes = sorted(pred_indiv_boxes, key=lambda b: b[0])
            
            # Calculate Individual IoUs for metric reporting
            for digit in digit_info:
                dx1, dy1, dx2, dy2 = digit['bbox']
                # Map to sharpened crop coordinate
                nx1, ny1 = (dx1 - gx1) * 2.0, (dy1 - gy1) * 2.0
                nx2, ny2 = (dx2 - gx1) * 2.0, (dy2 - gy1) * 2.0
                gt_box_sharp = (nx1, ny1, nx2, ny2)
                
                best_iou = 0
                for pbox in pred_indiv_boxes:
                    iou = calculate_iou(gt_box_sharp, pbox)
                    best_iou = max(best_iou, iou)
                s2_ious.append(best_iou)
                stage2_ious.append(best_iou)

        # -- Step 4: Classification & Assembly --
        predicted_digits = []
        for ibox in pred_indiv_boxes:
            try:
                inputs = preprocess_crop(sharp, (ibox[0], ibox[1], ibox[2], ibox[3])).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = classifier(inputs)
                    digit = out.argmax(dim=1).item()
                    predicted_digits.append(str(digit))
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
            'vis_gx1': gx1, 'vis_gy1': gy1, 'vis_gx2': gx2, 'vis_gy2': gy2,
            'vis_iboxes': pred_indiv_boxes
        })

    # 4. Reporting
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("📊 FINAL PIPELINE BENCHMARK")
    print("="*50)
    print(f"Full Sequence Accuracy:     {df['correct'].mean():.2%}")
    print(f"Mean Digit Accuracy (CER-like): {df['digit_acc'].mean():.2%}")
    print(f"Stage 1 (Global) Mean IoU:      {np.mean(stage1_ious):.4f}")
    print(f"Stage 2 (Indiv)  Mean IoU:      {np.mean(stage2_ious):.4f}")
    
    print("\nSequence Accuracy by Category:")
    print(df.groupby('category')['correct'].mean())
    
    print("\nDigit Accuracy by Category:")
    print(df.groupby('category')['digit_acc'].mean())

    # 5. Dashboard Generation
    if args.save_viz:
        print("\n--- Phase 3: Generating Dashboard ---")
        viz_samples = []
        # Try to pick 2 success and 2 failures
        successes = [r for r in results if r['correct']]
        failures = [r for r in results if not r['correct']]
        
        viz_samples.extend(successes[:2])
        viz_samples.extend(failures[:2])
        
        fig, axes = plt.subplots(len(viz_samples), 4, figsize=(22, 5 * len(viz_samples)))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        for i, res in enumerate(viz_samples):
            # Panel 1: Original + Global BB
            img_rgb = cv2.cvtColor(res['vis_img'], cv2.COLOR_BGR2RGB)
            cv2.rectangle(img_rgb, (res['vis_gx1'], res['vis_gy1']), (res['vis_gx2'], res['vis_gy2']), (255, 0, 0), 4)
            axes[i, 0].imshow(img_rgb)
            axes[i, 0].set_title(f"1. Global Detection\n(IoU: {res['s1_iou']:.2f})")
            axes[i, 0].axis('off')
            
            # Panel 2: Raw Crop (Stage 2 input)
            crop_rgb = cv2.cvtColor(res['vis_crop'], cv2.COLOR_BGR2RGB)
            axes[i, 1].imshow(crop_rgb)
            axes[i, 1].set_title("2. Raw Crop\n(Input for Enhancement)")
            axes[i, 1].axis('off')

            # Panel 3: Sharp + Individual BBs
            sharp_rgb = cv2.cvtColor(res['vis_sharp'], cv2.COLOR_BGR2RGB)
            for ibox in res['vis_iboxes']:
                 cv2.rectangle(sharp_rgb, (int(ibox[0]), int(ibox[1])), (int(ibox[2]), int(ibox[3])), (255, 255, 0), 2)
            axes[i, 2].imshow(sharp_rgb)
            axes[i, 2].set_title("3. Individual Detection\n(Post-Sharpening)")
            axes[i, 2].axis('off')
            
            # Panel 4: Results
            axes[i, 3].axis('off')
            status = "SUCCESS" if res['correct'] else "FAILURE"
            color = "green" if res['correct'] else "red"
            txt = f"[{status}]\n\nGT:   {res['gt']}\nPred: {res['pred']}\n\nDigit Acc: {res['correct_digits']}/{res['total_digits']}\nCategory: {res['category']}"
            axes[i, 3].text(0.1, 0.5, txt, fontsize=14, fontweight='bold', color=color, verticalalignment='center')

        plt.suptitle("FULL PIPELINE PERFORMANCE DASHBOARD", fontsize=20, fontweight='bold', y=0.98)
        viz_path = os.path.join(VIS_DIR, "evaluation_dashboard.png")
        plt.savefig(viz_path, bbox_inches='tight', dpi=120)
        print(f"✨ Dashboard saved to: {viz_path}")

    # Save CSV
    csv_path = os.path.join(REPORTS_DIR, "pipeline_metrics.csv")
    df_mini = df.drop(columns=[c for c in df.columns if c.startswith('vis_')])
    df_mini.to_csv(csv_path, index=False)
    print(f"💾 Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    main()
