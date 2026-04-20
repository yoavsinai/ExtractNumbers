import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop
from image_preprocessing.digit_preprocessor import enhance_digit
from utils.data_utils import iter_new_samples, get_gt_from_anno

def get_full_gt_number(anno_path):
    import json
    with open(anno_path, 'r') as f:
        data = json.load(f)
    digits = []
    for number in data.get('detected_numbers', []):
        for digit in number.get('digits', []):
            digits.append({'x': digit['bounding_box']['x'], 'label': str(digit['label'])})
    digits.sort(key=lambda d: d['x'])
    return "".join([d['label'] for d in digits])

def main():
    # Structured Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")
    
    GLOBAL_MODEL_PATH = os.path.join(TRAINED_DIR, "globalbb.pt")
    INDIV_MODEL_PATH = os.path.join(TRAINED_DIR, "individualbb.pt")
    CLASSIFIER_PATH = os.path.join(TRAINED_DIR, "digit_classifier.pth")
    DATA_ROOT = os.path.join(BASE_DIR, "data", "digits_data")
    
    device = get_device()
    
    # Load Models
    print("Loading models...")
    global_model = YOLO(GLOBAL_MODEL_PATH)
    indiv_model = YOLO(INDIV_MODEL_PATH)
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    
    samples = iter_new_samples(DATA_ROOT)
    import random
    random.seed(42)
    random.shuffle(samples)
    
    successes = []
    failures = []
    
    print("Finding 2 successes and 2 failures...")
    for s in tqdm(samples):
        img = cv2.imread(s['image_path'])
        if img is None: continue
        
        # 1. Global Detection
        res1 = global_model.predict(source=img, imgsz=256, verbose=False)
        if not res1 or len(res1[0].boxes) == 0:
            if len(failures) < 2: failures.append({'s': s, 'error': 'GlobalBB missed'})
            continue
            
        gbox = res1[0].boxes[0]
        gx1, gy1, gx2, gy2 = map(int, gbox.xyxy[0].cpu().numpy())
        h, w = img.shape[:2]
        gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        
        # 2. Sharpening
        sharp = enhance_digit(crop, upscale_factor=2.0)
        
        # 3. Individual Detection
        res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
        if not res2 or len(res2[0].boxes) == 0:
            if len(failures) < 2: failures.append({'s': s, 'error': 'IndivBB missed', 'img': img, 'crop': crop, 'sharp': sharp, 'gx': (gx1,gy1,gx2,gy2)})
            continue
            
        iboxes = res2[0].boxes.xyxy.cpu().numpy()
        iboxes = sorted(iboxes, key=lambda b: b[0])
        
        # 4. Classification
        preds = []
        pred_crops = []
        for ibox in iboxes:
            ix1, iy1, ix2, iy2 = map(int, ibox)
            d_crop = sharp[max(0,iy1):min(sharp.shape[0],iy2), max(0,ix1):min(sharp.shape[1],ix2)]
            pred_crops.append(d_crop)
            
            inputs = preprocess_crop(sharp, (ibox[0], ibox[1], ibox[2], ibox[3])).unsqueeze(0).to(device)
            with torch.no_grad():
                preds.append(str(classifier(inputs).argmax(dim=1).item()))
        
        pred_num = "".join(preds)
        gt_num = get_full_gt_number(s['anno_path'])
        
        entry = {
            's': s, 'gt': gt_num, 'pred': pred_num, 'img': img, 'crop': crop, 'sharp': sharp, 
            'gx': (gx1,gy1,gx2,gy2), 'iboxes': iboxes, 'pred_crops': pred_crops, 'preds': preds
        }
        
        if pred_num == gt_num:
            if len(successes) < 2: successes.append(entry)
        else:
            if len(failures) < 2: failures.append(entry)
            
        if len(successes) >= 2 and len(failures) >= 2:
            break

    # Visualization
    all_viz = successes + failures
    fig = plt.figure(figsize=(24, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)
    
    for i, res in enumerate(all_viz):
        status = "SUCCESS" if i < 2 else "FAILURE"
        color = "green" if i < 2 else "red"
        
        # 1. Original + Global
        ax1 = plt.subplot(4, 5, i*5 + 1)
        img_rgb = cv2.cvtColor(res['img'], cv2.COLOR_BGR2RGB)
        if 'gx' in res:
            x1, y1, x2, y2 = res['gx']
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)
        ax1.imshow(img_rgb)
        ax1.set_title(f"[{status}] Original + Global", color=color, fontweight='bold')
        ax1.axis('off')
        
        # 2. Raw Crop
        ax2 = plt.subplot(4, 5, i*5 + 2)
        if 'crop' in res:
            ax2.imshow(cv2.cvtColor(res['crop'], cv2.COLOR_BGR2RGB))
            ax2.set_title("Raw Crop (Unsharp)")
        ax2.axis('off')
        
        # 3. Sharpened + Individual
        ax3 = plt.subplot(4, 5, i*5 + 3)
        if 'sharp' in res:
            sharp_rgb = cv2.cvtColor(res['sharp'], cv2.COLOR_BGR2RGB)
            if 'iboxes' in res:
                for bx in res['iboxes']:
                    cv2.rectangle(sharp_rgb, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (255, 255, 0), 2)
            ax3.imshow(sharp_rgb)
            ax3.set_title("Sharpened + Individual")
        ax3.axis('off')
        
        # 4. Classification Breakdown
        ax4 = plt.subplot(4, 5, i*5 + 4)
        ax4.axis('off')
        if 'pred_crops' in res:
            # Combine individual digit crops into one strip for visualization
            crops = [cv2.resize(c, (64, 64)) for c in res['pred_crops'] if c.size > 0]
            if crops:
                strip = np.hstack(crops)
                ax4.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
                ax4.set_title(f"Classified Digits: {' '.join(res['preds'])}")
        
        # 5. Summary Info
        ax5 = plt.subplot(4, 5, i*5 + 5)
        ax5.axis('off')
        gt = res.get('gt', 'N/A')
        pred = res.get('pred', 'N/A')
        txt = f"GT:   {gt}\nPred: {pred}\nSource: {res['s']['category']}"
        ax5.text(0.1, 0.5, txt, fontsize=14, fontweight='bold', verticalalignment='center')

    plt.suptitle("FULL PIPELINE ERROR ANALYSIS: Every Stage Breakdown", fontsize=22, fontweight='bold', y=0.98)
    out_path = os.path.join(VIS_DIR, "error_analysis.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=120)
    print(f"\n✨ Detailed analysis saved to: {out_path}")

if __name__ == "__main__":
    main()
