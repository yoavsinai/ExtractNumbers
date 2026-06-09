import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.utils.data_utils import iter_video_samples, get_video_gt_from_anno, create_mock_video_dataset
from src.digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop
from src.inference.frame_selector import FrameSelector

def load_video_frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
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
    
    parser = argparse.ArgumentParser(description="Stage 4: Digit Recognition Video Evaluation")
    parser.add_argument("--max-samples", type=int, default=10, help="Max video samples to evaluate")
    parser.add_argument("--balanced", action="store_true", help="Use equal/balanced split for video categories")
    parser.add_argument("--data-root", type=str, default=os.path.join(BASE_DIR, "data", "video_data"), help="Path to video dataset root")
    parser.add_argument("--output-dir", type=str, default=os.path.join(BASE_DIR, "outputs"), help="Base directory for outputs")
    parser.add_argument("--strategy", type=str, default="annotated", choices=["annotated", "uniform", "motion_and_blur", "random_1_in_10"],
                        help="Frame selection strategy")
    parser.add_argument("--k", type=int, default=5, help="Number of frames to select per video (ignored if strategy is 'annotated')")
    args = parser.parse_args()

    # Paths
    TRAINED_DIR = os.path.join(BASE_DIR, "outputs", "trained_models")
    CLASSIFIER_PATH = os.path.join(TRAINED_DIR, "digit_recognizer.pt")
    REPORTS_DIR = os.path.join(args.output_dir, "reports")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    device = get_device()

    if not os.path.exists(CLASSIFIER_PATH):
        print(f"❌ Error: Classifier model not found at {CLASSIFIER_PATH}")
        sys.exit(1)

    print("\n--- Stage 4: Digit Recognition Video Evaluation ---")
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    print("✓ Classifier loaded successfully.")

    video_samples = list(iter_video_samples(args.data_root))
    if not video_samples:
        print("⚠️ No video samples found. Generating mock video dataset...")
        create_mock_video_dataset(args.data_root)
        video_samples = list(iter_video_samples(args.data_root))

    if not video_samples:
        print("❌ Error: No video samples available.")
        sys.exit(1)

    # Group by category for sampling
    samples_by_cat = defaultdict(list)
    for s in video_samples:
        samples_by_cat[s['category']].append(s)

    random.seed(42)
    eval_videos = []
    if args.balanced:
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

    y_true = []
    y_pred = []
    results = []

    for vid_sample in tqdm(eval_videos, desc="Evaluating Digit Recognition in Videos"):
        vid_path = vid_sample['video_path']
        anno_path = vid_sample['anno_path']
        category = vid_sample['category']
        sample_id = vid_sample['sample_id']
        
        try:
            frames = load_video_frames(vid_path)
        except Exception:
            continue
            
        try:
            with open(anno_path, 'r') as f:
                anno_json = json.load(f)
            annotated_frame_keys = list(anno_json.get("frames", {}).keys())
            annotated_frame_indices = sorted([int(k) for k in annotated_frame_keys])
        except Exception:
            continue
            
        # Determine frame indices to evaluate
        if args.strategy == "annotated":
            eval_frame_indices = [idx for idx in annotated_frame_indices if idx < len(frames)]
        else:
            selector = FrameSelector(strategy=args.strategy, top_k=args.k)
            selected_indices = selector.select_indices(frames)
            eval_frame_indices = [idx for idx in selected_indices if idx in annotated_frame_indices and idx < len(frames)]

        for frame_idx in eval_frame_indices:
            img = frames[frame_idx]
            _, digit_info, has_digit_boxes, _ = get_video_gt_from_anno(anno_path, frame_idx)
            if not has_digit_boxes:
                continue

            for digit in digit_info:
                bbox = digit['bbox']
                label = digit['label']
                if label == -1 or label == "digit" or label is None:
                    continue  # skip MNIST digits — no ground truth label
                label = str(label)
                
                try:
                    # Crop using GT boxes
                    t_crop = preprocess_crop(img, bbox)
                    with torch.no_grad():
                        out = classifier(t_crop.unsqueeze(0).to(device))
                        pred_digit = str(out.argmax(dim=1).item())
                        
                    y_true.append(label)
                    y_pred.append(pred_digit)
                    
                    results.append({
                        'video_id': sample_id,
                        'frame_idx': frame_idx,
                        'category': category,
                        'gt': label,
                        'pred': pred_digit,
                        'correct': label == pred_digit
                    })
                except Exception:
                    continue

    if not y_true:
        print("⚠️ No evaluation results generated.")
        sys.exit(0)

    report_lines = []
    def log_print(text=""):
        print(text)
        report_lines.append(str(text))

    log_print("\n" + "="*50)
    log_print("📊 VIDEO STAGE 4: DIGIT CLASSIFICATION METRICS")
    log_print("="*50)

    overall_acc = accuracy_score(y_true, y_pred)
    log_print(f"Overall Classification Accuracy: {overall_acc:.2%}")
    log_print(f"Total Evaluated Digits:          {len(y_true)}")

    log_print("\n📋 CLASSIFICATION REPORT:")
    labels = [str(i) for i in range(10)]
    log_print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    # Print accuracy per category
    df = pd.DataFrame(results)
    log_print("📈 ACCURACY BY CATEGORY:")
    for cat in sorted(df['category'].unique()):
        cat_df = df[df['category'] == cat]
        cat_acc = cat_df['correct'].mean()
        log_print(f"  {cat:<15}: {cat_acc:.2%} (Count: {len(cat_df)})")

    # Save reports
    report_txt_path = os.path.join(REPORTS_DIR, "video_stage4_digit_recog_summary.txt")
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📝 Text report saved to: {report_txt_path}")

    csv_path = os.path.join(REPORTS_DIR, "video_stage4_digit_recog_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Detailed CSV saved to: {csv_path}")

if __name__ == "__main__":
    main()
