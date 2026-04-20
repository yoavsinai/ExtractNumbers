import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src"))

from digit_recognizer.digit_recognizer import build_digit_model, get_device
from image_preprocessing.digit_preprocessor import enhance_digit
from bounding_box.globalbb_detector import _read_mask_grayscale
from utils.metrics import calculate_iou, print_metrics_report
from utils.data_utils import iter_new_samples, get_gt_from_anno as get_gt_from_anno_raw

def get_gt_from_anno(anno_path):
    """Wrapper for unified GT extraction."""
    global_boxes, digit_info = get_gt_from_anno_raw(anno_path)
    individual_boxes = [d['bbox'] for d in digit_info]
    labels = [d['label'] for d in digit_info]
    return global_boxes, individual_boxes, labels

def evaluate_pipeline(model_path, global_weights, individual_weights, data_root, max_samples=50):
    print("🚀 EVALUATING FULL PIPELINE PERFORMANCE (NEW DATA STRUCTURE)")
    print("=" * 70)

    device = get_device()
    
    # Load Models
    from ultralytics import YOLO
    print("Loading Stage 1: GlobalBB...")
    global_model = YOLO(global_weights)
    
    print("Loading Stage 3: IndividualBB...")
    individual_model = YOLO(individual_weights)
    
    print("Loading Stage 5: Digit Classifier...")
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()

    # Load Samples using new structure
    all_samples = iter_new_samples(data_root)
    
    # Shuffle and limit
    import random
    random.seed(42)
    random.shuffle(all_samples)
    samples = all_samples[:max_samples]

    if not samples:
        print(f"❌ No samples found in {data_root}. Please check the path and data structure.")
        return []

    results = []
    
    # Metric accumulators
    stage1_ious = []
    stage2_ious = []
    y_true_digits = []
    y_pred_digits = []

    for s in tqdm(samples, desc="Processing Samples"):
        img_path = s["image_path"]
        anno_path = s["anno_path"]
        cat = s["category"]
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        # GT Data from JSON
        gt_global_boxes, gt_individual_boxes, gt_labels = get_gt_from_anno(anno_path)
        
        # Stage 1: Global Detection
        res1 = global_model.predict(source=img, imgsz=256, verbose=False)
        pred_global_boxes = []
        if res1 and len(res1[0].boxes) > 0:
            pred_global_boxes = res1[0].boxes.xyxy.detach().cpu().numpy()
            
        # Calc Stage 1 IoU
        sample_s1_ious = []
        for gt_box in (gt_global_boxes or []):
            best_iou = 0
            for p_box in pred_global_boxes:
                iou = calculate_iou(gt_box, p_box)
                best_iou = max(best_iou, iou)
            sample_s1_ious.append(best_iou)
            stage1_ious.append(best_iou)
        
        # Process detections
        sharp_crops = []
        if len(pred_global_boxes) > 0:
            # Use the most confident one
            best_idx = res1[0].boxes.conf.argmax().item()
            gx1, gy1, gx2, gy2 = map(int, pred_global_boxes[best_idx])
            h, w = img.shape[:2]
            gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
            crop = img[gy1:gy2, gx1:gx2]
            
            if crop.size > 0:
                # Stage 2: Enhancement
                sharp = enhance_digit(crop, upscale_factor=2.0)
                
                # Stage 3: Individual Detection
                res3 = individual_model.predict(source=sharp, imgsz=256, verbose=False)
                pred_indiv_boxes = []
                if res3 and len(res3[0].boxes) > 0:
                    pred_indiv_boxes = res3[0].boxes.xyxy.detach().cpu().numpy()
                
                # Calc Stage 2 IoU
                sample_s2_ious = []
                for (dx1, dy1, dx2, dy2) in gt_individual_boxes:
                    nx1 = (dx1 - gx1) * 2.0
                    ny1 = (dy1 - gy1) * 2.0
                    nx2 = (dx2 - gx1) * 2.0
                    ny2 = (dy2 - gy1) * 2.0
                    
                    gt_box_sharp = (nx1, ny1, nx2, ny2)
                    best_iou = 0
                    for p_box in pred_indiv_boxes:
                        iou = calculate_iou(gt_box_sharp, p_box)
                        best_iou = max(best_iou, iou)
                    sample_s2_ious.append(best_iou)
                    stage2_ious.append(best_iou)

                # Stage 4: Classification
                sorted_indices = np.argsort(pred_indiv_boxes[:, 0]) if len(pred_indiv_boxes) > 0 else []
                sample_preds = []
                
                from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage
                transform = Compose([
                    Resize((64, 64)),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                for idx in sorted_indices:
                    ix1, iy1, ix2, iy2 = map(int, pred_indiv_boxes[idx])
                    sh, sw = sharp.shape[:2]
                    ix1, iy1, ix2, iy2 = max(0, ix1), max(0, iy1), min(sw, ix2), min(sh, iy2)
                    digit_crop = sharp[iy1:iy2, ix1:ix2]
                    
                    if digit_crop.size > 0:
                        digit_crop_rgb = cv2.cvtColor(digit_crop, cv2.COLOR_BGR2RGB)
                        tensor = transform(ToPILImage()(digit_crop_rgb)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = classifier(tensor)
                            pred_digit = out.argmax(dim=-1).item()
                            sample_preds.append(pred_digit)
                
                # Classification tracking
                correct_count = 0
                if len(sample_preds) == len(gt_labels):
                    for p, g in zip(sample_preds, gt_labels):
                        if p == g: correct_count += 1
                        y_true_digits.append(g)
                        y_pred_digits.append(p)
                
                # Check for perfect match
                is_perfect = (len(sample_preds) == len(gt_labels) and correct_count == len(gt_labels) and (np.mean(sample_s1_ious) if sample_s1_ious else 0) > 0.8)
                
                results.append({
                    'sample': s,
                    'is_perfect': is_perfect,
                    'correct_digits': correct_count,
                    'total_digits': len(gt_labels),
                    'pred_digits': sample_preds,
                    'gt_labels': gt_labels,
                    's1_iou': np.mean(sample_s1_ious) if sample_s1_ious else 0,
                    's2_iou': np.mean(sample_s2_ious) if sample_s2_ious else 0,
                    'img': img,
                    'sharp': sharp,
                    'pred_global': (gx1, gy1, gx2, gy2),
                    'pred_indiv': pred_indiv_boxes[sorted_indices] if len(pred_indiv_boxes) > 0 else []
                })

    # Summary Metrics
    print("\n" + "="*50)
    print("📊 AGGREGATE PIPELINE METRICS")
    print("="*50)
    print(f"Stage 1 (Global BB) Mean IoU:     {np.mean(stage1_ious):.4f}")
    print(f"Stage 2 (Individual BB) Mean IoU: {np.mean(stage2_ious):.4f}")
    
    # Classification Report
    metrics, report = print_metrics_report(y_true_digits, y_pred_digits, title="Per-Digit Performance")
    
    # Presentation
    perfect_samples = [r for r in results if r['is_perfect']][:2]
    failed_samples = [r for r in results if not r['is_perfect']][:2]
    
    # If not enough perfect, take best ones
    if len(perfect_samples) < 2:
        sorted_results = sorted(results, key=lambda x: (x['correct_digits']/max(1,x['total_digits']), x['s1_iou']), reverse=True)
        perfect_samples = sorted_results[:2]
        
    # If not enough failed, take worst ones
    if len(failed_samples) < 2:
        sorted_results = sorted(results, key=lambda x: (x['correct_digits']/max(1,x['total_digits']), x['s1_iou']))
        failed_samples = sorted_results[:2]

    display_samples = perfect_samples + failed_samples
    labels = ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE"]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    plt.subplots_adjust(hspace=0.4)
    
    for i, (res, label) in enumerate(zip(display_samples, labels)):
        # Original + Global Detection
        img_orig = cv2.cvtColor(res['img'], cv2.COLOR_BGR2RGB)
        gx1, gy1, gx2, gy2 = res['pred_global']
        cv2.rectangle(img_orig, (gx1, gy1), (gx2, gy2), (255, 0, 0), 3)
        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"{label} - Original & Global BB\nIoU: {res['s1_iou']:.2f}", fontweight='bold', color='green' if 'SUCCESS' in label else 'red')
        axes[i, 0].axis('off')
        
        # Sharpened + Individual Detection
        img_sharp = cv2.cvtColor(res['sharp'], cv2.COLOR_BGR2RGB)
        for ix1, iy1, ix2, iy2 in res['pred_indiv']:
            cv2.rectangle(img_sharp, (int(ix1), int(iy1)), (int(ix2), int(iy2)), (255, 255, 0), 2)
        axes[i, 1].imshow(img_sharp)
        axes[i, 1].set_title(f"Sharpened & Individual BB\noU: {res['s2_iou']:.2f}", fontsize=10)
        axes[i, 1].axis('off')
        
        # Results Text
        axes[i, 2].axis('off')
        res_text = f"Category: {res['sample']['category']}\n"
        res_text += f"Ground Truth: {res['gt_labels']}\n"
        res_text += f"Predictions:  {res['pred_digits']}\n"
        res_text += f"Correct: {res['correct_digits']}/{res['total_digits']}"
        axes[i, 2].text(0.1, 0.5, res_text, fontsize=12, fontweight='bold', verticalalignment='center')

    plt.suptitle("FULL PIPELINE EVALUATION DASHBOARD", fontsize=20, fontweight='bold', y=0.95)
    
    output_path = os.path.join(BASE_DIR, "outputs", "pipeline_evaluation_vibrant.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"\n✨ Dashboard saved to: {output_path}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=os.path.join(BASE_DIR, "outputs", "bbox_comparison", "digit_classifier.pth"))
    parser.add_argument("--global-weights", type=str, default=os.path.join(BASE_DIR, "outputs", "bbox_comparison", "globalbb_run", "weights", "best.pt"))
    parser.add_argument("--individual-weights", type=str, default=os.path.join(BASE_DIR, "outputs", "bbox_comparison", "individualbb_run", "weights", "best.pt"))
    parser.add_argument("--data-root", type=str, default=os.path.join(BASE_DIR, "data", "digits_data"))
    parser.add_argument("--max-samples", type=int, default=30)
    
    args = parser.parse_args()
    
    # Check paths
    missing = []
    for p in [args.model_path, args.global_weights, args.individual_weights, args.data_root]:
        if not os.path.exists(p): missing.append(p)
    
    if missing:
        print("❌ Missing files/directories:")
        for m in missing: print(f"  - {m}")
        sys.exit(1)
        
    evaluate_pipeline(args.model_path, args.global_weights, args.individual_weights, args.data_root, args.max_samples)
