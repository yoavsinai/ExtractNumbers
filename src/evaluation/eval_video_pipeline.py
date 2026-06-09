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

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop
from src.image_preprocessing.digit_preprocessor import enhance_digit
from src.utils.data_utils import iter_video_samples, get_video_gt_from_anno, create_mock_video_dataset
from src.utils.metrics import calculate_iou
from src.utils.bbox_utils import merge_global_boxes, nms_individual_boxes
from src.inference.frame_selector import FrameSelector

def calculate_digit_accuracy(gt, pred):
    """Calculate positioning accuracy and succession rate."""
    correct = 0
    total = len(gt)
    successions = 0
    possible_successions = 0
    
    for i in range(min(len(gt), len(pred))):
        if gt[i] == pred[i]:
            correct += 1
            if i + 1 < min(len(gt), len(pred)):
                possible_successions += 1
                if gt[i+1] == pred[i+1]:
                    successions += 1
                    
    succession_rate = successions / possible_successions if possible_successions > 0 else 1.0
    return correct, total, succession_rate

def load_video_frames(video_path: str) -> list:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def main():
    import argparse
    import random
    from collections import defaultdict
    
    # Avoid thread contention overhead on high-core CPU machines
    torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser(description="Full Staged Video Pipeline Benchmark")
    parser.add_argument("--max-samples", type=int, default=10, help="Max video samples to evaluate")
    parser.add_argument("--save-viz", action="store_true", help="Save the evaluation dashboard image")
    parser.add_argument("--analyze-errors", action="store_true", help="Generate detailed error analysis visualization")
    parser.add_argument("--data-root", type=str, default=os.path.join(BASE_DIR, "data", "video_data"), help="Path to video dataset root")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"), help="Base directory for outputs")
    parser.add_argument("--balanced", action="store_true", help="Use equal/balanced split for video categories")
    parser.add_argument("--strategy", type=str, default="annotated", choices=["annotated", "uniform", "motion_and_blur", "random_1_in_10"],
                        help="Frame selection strategy. 'annotated' evaluates exactly the labeled frames.")
    parser.add_argument("--k", type=int, default=5, help="Number of frames to select per video (ignored if strategy is 'annotated')")
    args = parser.parse_args()

    # Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    VIS_DIR = os.path.join(args.output_dir, "visualizations")
    REPORTS_DIR = os.path.join(args.output_dir, "reports")
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    GLOBAL_MODEL_PATH = os.path.join(TRAINED_DIR, "globalbb.pt")
    INDIV_MODEL_PATH = os.path.join(TRAINED_DIR, "individualbb.pt")
    CLASSIFIER_PATH = os.path.join(TRAINED_DIR, "digit_recognizer.pt")
    
    device = get_device()
    
    # Load Models
    print("\n--- Video Pipeline Evaluation: Loading Models ---")
    if not all([os.path.exists(p) for p in [GLOBAL_MODEL_PATH, INDIV_MODEL_PATH, CLASSIFIER_PATH]]):
        print("❌ Error: Missing trained model weights. Run the main training pipeline first.")
        sys.exit(1)

    global_model = YOLO(GLOBAL_MODEL_PATH)
    indiv_model = YOLO(INDIV_MODEL_PATH)
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    print("✓ All models loaded successfully.")

    # Prepare Video Samples
    print(f"\nScanning for video samples under: {args.data_root}")
    video_samples = list(iter_video_samples(args.data_root))
    
    if not video_samples:
        print("⚠️ No video samples found. Generating a mock annotated video dataset...")
        create_mock_video_dataset(args.data_root)
        video_samples = list(iter_video_samples(args.data_root))
        
    if not video_samples:
        print("❌ Error: Could not locate or generate video samples.")
        sys.exit(1)

    # Group by category for sampling
    samples_by_cat = defaultdict(list)
    for s in video_samples:
        samples_by_cat[s['category']].append(s)
        
    random.seed(42)
    eval_videos = []
    
    if args.max_samples <= 0 or args.max_samples is None:
        eval_videos = video_samples
    elif args.balanced:
        target_per_cat = args.max_samples // len(samples_by_cat)
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            eval_videos.extend(samps[:min(target_per_cat, len(samps))])
    else:
        total_vids = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            vids_per_cat = max(1, int(round(args.max_samples * (len(samps) / total_vids))))
            eval_videos.extend(samps[:vids_per_cat])
            
    random.shuffle(eval_videos)
    print(f"Loaded {len(eval_videos)} video samples for evaluation.")

    results = []
    
    # Evaluate frame-by-frame
    for vid_sample in tqdm(eval_videos, desc="Evaluating Videos"):
        vid_path = vid_sample['video_path']
        anno_path = vid_sample['anno_path']
        category = vid_sample['category']
        sample_id = vid_sample['sample_id']
        
        try:
            frames = load_video_frames(vid_path)
        except Exception as e:
            print(f"❌ Failed to load video {vid_path}: {e}")
            continue
            
        # Extract annotated frame indices
        try:
            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            annotated_frame_keys = list(anno_json.get("frames", {}).keys())
            annotated_frame_indices = sorted([int(k) for k in annotated_frame_keys])
        except Exception as e:
            print(f"❌ Failed to read annotations {anno_path}: {e}")
            continue
            
        if not annotated_frame_indices:
            print(f"⚠️ No annotated frames found for video {sample_id}")
            continue
            
        # Determine which frames to process
        if args.strategy == "annotated":
            eval_frame_indices = [idx for idx in annotated_frame_indices if idx < len(frames)]
        else:
            selector = FrameSelector(strategy=args.strategy, top_k=args.k)
            selected_indices = selector.select_indices(frames)
            # Evaluate only the selected frames that have annotations
            eval_frame_indices = [idx for idx in selected_indices if idx in annotated_frame_indices and idx < len(frames)]
            
        if not eval_frame_indices:
            continue
            
        # Process selected frames
        for frame_idx in eval_frame_indices:
            img = frames[frame_idx]
            
            # Ground Truth annotations for this frame
            gt_global_boxes, digit_info, has_digit_boxes, gt_number = get_video_gt_from_anno(anno_path, frame_idx)
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
                    'video_id': sample_id,
                    'frame_idx': frame_idx,
                    'gt': gt_number if has_label else "N/A",
                    'pred': '',
                    'correct': False if has_label else None,
                    'digit_acc': 0.0 if has_digit_boxes else None,
                    'succession_rate': None,
                    'correct_digits': 0 if has_digit_boxes else None,
                    'total_digits': len(digit_info) if has_digit_boxes else None,
                    'category': category,
                    's1_iou': 0.0,
                    's2_iou_avg': 0.0 if has_digit_boxes else None,
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
                    'video_id': sample_id,
                    'frame_idx': frame_idx,
                    'gt': gt_number if has_label else "N/A",
                    'pred': '',
                    'correct': False if has_label else None,
                    'digit_acc': 0.0 if has_digit_boxes else None,
                    'succession_rate': None,
                    'correct_digits': 0 if has_digit_boxes else None,
                    'total_digits': len(digit_info) if has_digit_boxes else None,
                    'category': category,
                    's1_iou': s1_iou,
                    's2_iou_avg': 0.0 if has_digit_boxes else None,
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
                
            # -- Step 2: Pass-through (No enhancement) --
            sharp = crop
            scale = 1.0
            
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
                        
                        ix1, iy1, ix2, iy2 = map(int, ibox)
                        d_crop = sharp[max(0,iy1):min(sharp.shape[0],iy2), max(0,ix1):min(sharp.shape[1],ix2)]
                        pred_crops.append(d_crop)
                except Exception:
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
                'video_id': sample_id,
                'frame_idx': frame_idx,
                'gt': gt_number if has_label else "N/A",
                'pred': pred_number,
                'correct': (pred_number == gt_number) if has_label else None,
                'digit_acc': (correct_digits / total_gt_digits) if has_digit_boxes and total_gt_digits > 0 else None,
                'succession_rate': succession_rate if has_digit_boxes else None,
                'correct_digits': correct_digits if has_digit_boxes else None,
                'total_digits': total_gt_digits if has_digit_boxes else None,
                'category': category,
                's1_iou': s1_iou,
                's2_iou_avg': np.mean(s2_ious) if s2_ious and has_digit_boxes else None,
                'has_digit_boxes': has_digit_boxes,
                'has_label': has_label,
                'digit_pairs': digit_pairs,
                # Visualization data
                'vis_img': img,
                'vis_crop': crop,
                'vis_sharp': sharp,
                'vis_gx': (gx1, gy1, gx2, gy2),
                'vis_iboxes': pred_indiv_boxes,
                'vis_pred_crops': pred_crops,
                'vis_preds': predicted_digits
            })

    if not results:
        print("⚠️ No evaluation results generated.")
        sys.exit(0)

    df = pd.DataFrame(results)
    
    report_lines = []
    def log_print(text=""):
        print(text)
        report_lines.append(str(text))

    log_print("\n" + "="*60)
    log_print("📊 FINAL VIDEO PIPELINE BENCHMARK")
    log_print("="*60)
    
    labeled_df = df[df['has_label'] == True]
    digit_boxed_df = df[df['has_digit_boxes'] == True]
    
    full_seq_acc = labeled_df['correct'].mean() if len(labeled_df) > 0 else 0.0
    mean_digit_acc = digit_boxed_df['digit_acc'].mean() if len(digit_boxed_df) > 0 else 0.0
    s1_iou = df['s1_iou'].mean()
    s3_iou = digit_boxed_df['s2_iou_avg'].mean() if len(digit_boxed_df) > 0 else 0.0
    
    log_print(f"Full Sequence Accuracy:       {full_seq_acc:.2%}")
    log_print(f"Mean Digit Accuracy (Pos):    {mean_digit_acc:.2%}")
    log_print(f"Stage 1 (Global) Mean IoU:    {s1_iou:.4f} (All Frames)")
    log_print(f"Stage 3 (Indiv)  Mean IoU:    {s3_iou:.4f}")
    
    log_print("\n📈 PERFORMANCE BY VIDEO CATEGORY:")
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
    
    for col in ['Seq Acc', 'Digit Acc', 'S1 IoU', 'S2 IoU']:
        cat_stats[col] = cat_stats[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    log_print(cat_stats.to_string())

    # Dashboard Generation
    if args.save_viz:
        print("\n--- Generating Video Pipeline Dashboard ---")
        viz_samples = []
        successes = [r for r in results if r['correct']]
        failures = [r for r in results if not r['correct']]
        viz_samples.extend(successes[:2])
        viz_samples.extend(failures[:2])
        
        fig, axes = plt.subplots(len(viz_samples), 4, figsize=(22, 5 * len(viz_samples)))
        if len(viz_samples) == 1: axes = axes.reshape(1, -1)
        
        for i, res in enumerate(viz_samples):
            img_rgb = cv2.cvtColor(res['vis_img'], cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = res['vis_gx']
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)
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
            txt = f"Video: {res['video_id']}\nFrame: {res['frame_idx']}\nGT:   {res['gt']}\nPred: {res['pred']}\n\nCategory: {res['category']}"
            axes[i, 3].text(0.1, 0.5, txt, fontsize=14, fontweight='bold', color=color, verticalalignment='center')

        plt.suptitle("VIDEO PIPELINE PERFORMANCE DASHBOARD", fontsize=20, fontweight='bold', y=0.98)
        viz_path = os.path.join(VIS_DIR, "video_pipeline_dashboard.png")
        plt.savefig(viz_path, bbox_inches='tight', dpi=120)
        print(f"✓ Dashboard saved to {viz_path}")

    # Detailed Error Analysis
    if args.analyze_errors:
        print("\n--- Generating Video Detailed Error Analysis ---")
        failures = [r for r in results if not r['correct']][:4]
        if not failures: failures = results[:4]
        
        fig = plt.figure(figsize=(24, 18))
        for i, res in enumerate(failures):
             ax1 = plt.subplot(4, 5, i*5 + 1)
             img_rgb = cv2.cvtColor(res['vis_img'], cv2.COLOR_BGR2RGB)
             x1, y1, x2, y2 = res['vis_gx']
             cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)
             ax1.imshow(img_rgb)
             ax1.axis('off')
             ax1.set_title("Original + Global", color="red", fontweight='bold')
             
             ax2 = plt.subplot(4, 5, i*5 + 2)
             ax2.imshow(cv2.cvtColor(res['vis_crop'], cv2.COLOR_BGR2RGB))
             ax2.axis('off')
             ax2.set_title("Raw Crop")
             
             ax3 = plt.subplot(4, 5, i*5 + 3)
             sharp_rgb = cv2.cvtColor(res['vis_sharp'], cv2.COLOR_BGR2RGB)
             for bx in res['vis_iboxes']:
                 cv2.rectangle(sharp_rgb, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), (255, 255, 0), 2)
             ax3.imshow(sharp_rgb)
             ax3.axis('off')
             ax3.set_title("Sharpened + Individual")
             
             ax4 = plt.subplot(4, 5, i*5 + 4)
             if res['vis_pred_crops']:
                 strip = np.hstack([cv2.resize(c, (64, 64)) for c in res['vis_pred_crops'] if c.size > 0])
                 ax4.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))
                 ax4.set_title(f"Preds: {' '.join(res['vis_preds'])}")
             ax4.axis('off')
             
             ax5 = plt.subplot(4, 5, i*5 + 5)
             ax5.axis('off')
             txt = f"Video: {res['video_id']}\nFrame: {res['frame_idx']}\nGT:   {res['gt']}\nPred: {res['pred']}\nCat:  {res['category']}"
             ax5.text(0.1, 0.5, txt, fontsize=14, fontweight='bold', verticalalignment='center')

        plt.suptitle("VIDEO DETAILED ERROR ANALYSIS", fontsize=22, fontweight='bold', y=0.98)
        err_path = os.path.join(VIS_DIR, "video_detailed_error_analysis.png")
        plt.savefig(err_path, bbox_inches='tight', dpi=120)
        print(f"✓ Error analysis saved to {err_path}")

    # Save CSV and Text Reports
    df_mini = df.drop(columns=[c for c in df.columns if c.startswith('vis_')])
    df_mini.to_csv(os.path.join(REPORTS_DIR, "video_pipeline_metrics.csv"), index=False)
    
    with open(os.path.join(REPORTS_DIR, "video_pipeline_summary.txt"), "w") as f:
        f.write("\n".join(report_lines))
        
    print(f"\n💾 Video pipeline results saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()
