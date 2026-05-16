import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop, DigitDataset
from image_preprocessing.digit_preprocessor import (
    enhance_digit, enhance_without_sharpening, enhance_with_traditional_methods
)
from utils.data_utils import iter_new_samples, get_gt_from_anno

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 4: Digit Recognition Evaluation")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--compare-enhancements", action="store_true", help="Compare impact of different sharpening methods")
    args = parser.parse_args()

    # Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    CLASSIFIER_PATH = os.path.join(TRAINED_DIR, "digit_recognizer.pt")
    DATA_ROOT = os.path.join(BASE_DIR, "data", "digits_data")
    REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    device = get_device()
    
    if not os.path.exists(CLASSIFIER_PATH):
        print(f"❌ Error: Classifier model not found at {CLASSIFIER_PATH}")
        sys.exit(1)

    print("\n" + "="*40)
    print("📊 STAGE 4: DIGIT RECOGNITION METRICS")
    print("="*40)
    
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    
    all_samples = list(iter_new_samples(DATA_ROOT))
    import random
    random.seed(42)
    
    # Exclude categories without individual digit annotations
    excluded_categories = ['race_numbers', 'ocr_trains']
    from collections import defaultdict
    samples_by_cat = defaultdict(list)
    for s in all_samples:
        if s['category'] not in excluded_categories:
            samples_by_cat[s['category']].append(s)
            
    eval_samples = []
    if samples_by_cat:
        total_samples = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            num_samples = int(args.max_samples * (len(samps) / total_samples))
            eval_samples.extend(samps[:num_samples])
        random.shuffle(eval_samples)
    
    results = []
    y_true = []
    y_pred = []
    
    for s in tqdm(eval_samples, desc="Evaluating Recognition"):
        img = cv2.imread(s['image_path'])
        if img is None: continue
        
        # Use GT boxes to isolate classification performance
        gt_global_boxes, digit_info, has_digit_boxes, _ = get_gt_from_anno(s['anno_path'])
        if not gt_global_boxes or not has_digit_boxes: continue
        
        gx1, gy1, gx2, gy2 = map(int, gt_global_boxes[0])
        h, w = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        if crop.size == 0: continue
            
        for enh_method in ["none", "esrgan"]:
            sharp = enhance_digit(crop, upscale_factor=2.0, method=enh_method)
            scale = 2.0 if enh_method in ["esrgan", "opencv"] else 1.0
            
            for digit in digit_info:
                dx1, dy1, dx2, dy2 = digit['bbox']
                # Map to sharpened crop coordinate
                nx1, ny1 = (dx1 - gx1) * scale, (dy1 - gy1) * scale
                nx2, ny2 = (dx2 - gx1) * scale, (dy2 - gy1) * scale
                ibox = (nx1, ny1, nx2, ny2)
                
                try:
                    inputs = preprocess_crop(sharp, ibox).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = classifier(inputs)
                        pred_digit = out.argmax(dim=1).item()
                        gt_digit = int(digit['label'])
                        
                        y_true.append(gt_digit)
                        y_pred.append(pred_digit)
                        
                        results.append({
                            'sample_id': s['sample_id'],
                            'category': s['category'],
                            'enhancement': enh_method,
                            'gt': gt_digit,
                            'pred': pred_digit,
                            'correct': pred_digit == gt_digit
                        })
                except:
                    continue

    df = pd.DataFrame(results)
    
    # Global Metrics
    total_acc = df['correct'].mean()
    print(f"Overall Accuracy: {total_acc:.2%}")
    
    print("\nDetailed Classification Report:")
    unique_labels = sorted(set(y_true))
    target_names = [str(i) for i in unique_labels]
    report = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names)
    print(report)
    
    print("\n📈 ACCURACY BY CATEGORY:")
    cat_acc = df.groupby('category')['correct'].mean()
    print(cat_acc)

    # Save reports
    csv_path = os.path.join(REPORTS_DIR, "stage4_digit_recog_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    with open(os.path.join(REPORTS_DIR, "stage4_classification_report.txt"), "w") as f:
        f.write(report)
    
    if args.compare_enhancements:
        print("\n--- Enhancement Method Comparison ---")
        methods = {
            'Real-ESRGAN': lambda c: enhance_digit(c, upscale_factor=2.0),
            'Traditional': lambda c: enhance_with_traditional_methods(c, target_size=64),
            'No-Sharpen': lambda c: enhance_without_sharpening(c, target_size=64)
        }
        
        comparison_results = []
        for name, func in methods.items():
            correct = 0
            total = 0
            for s in eval_samples[:50]: # Test on smaller subset for speed
                 img = cv2.imread(s['image_path'])
                 if img is None: continue
                 gt_global_boxes, digit_info, has_digits, _ = get_gt_from_anno(s['anno_path'])
                 if not gt_global_boxes or not has_digits: continue
                 gx1, gy1, gx2, gy2 = map(int, gt_global_boxes[0])
                 crop = img[gy1:gy2, gx1:gx2]
                 if crop.size == 0: continue
                 
                 enhanced = func(crop)
                 for digit in digit_info:
                      # Simplistic mapping (might need adjustment for traditional/no-sharpen which resize differently)
                      # For comparison, we use the standard preprocess_crop which handles resize
                      # We just provide the enhanced crop
                      try:
                           # Traditional/No-Sharpen functions return pre-processed small crops
                           # but here we want to test the *method* on the whole sequence crop
                           if name == 'Real-ESRGAN':
                                scale = 2.0
                           else:
                                scale = 1.0 # Traditional/No-Sharpen usually don't upscale the sequence crop like ESRGAN
                                
                           # Re-calc ibox for the enhanced crop
                           nx1, ny1 = (digit['bbox'][0] - gx1) * scale, (digit['bbox'][1] - gy1) * scale
                           nx2, ny2 = (digit['bbox'][2] - gx1) * scale, (digit['bbox'][3] - gy1) * scale
                           
                           inputs = preprocess_crop(enhanced, (nx1, ny1, nx2, ny2)).unsqueeze(0).to(device)
                           with torch.no_grad():
                                if classifier(inputs).argmax(dim=1).item() == int(digit['label']):
                                     correct += 1
                                total += 1
                      except: continue
            
            acc = correct / total if total > 0 else 0
            print(f"{name:12}: {acc:.2%}")
            comparison_results.append({'method': name, 'accuracy': acc})
        
        pd.DataFrame(comparison_results).to_csv(os.path.join(REPORTS_DIR, "enhancement_comparison.csv"), index=False)

    print(f"\n💾 Results saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()
