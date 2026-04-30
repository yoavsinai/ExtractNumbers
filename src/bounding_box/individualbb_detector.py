import argparse
import os
import random
import shutil
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, set_logging
except ImportError:
    pass

import sys
import warnings
# Use default logging to allow progress bars
import logging
import warnings
warnings.filterwarnings('ignore')
# Ensure we can import from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import enhance_digit
from bounding_box.globalbb_detector import ensure_dir, stratified_split
from utils.data_utils import iter_new_samples, get_gt_from_anno

def make_individualbb_dataset(
    samples: List[Dict[str, str]],
    categories: List[str],
    individualbb_out_root: str,
    split_ratio: float,
    seed: int,
    min_area: int = 10,
) -> str:
    """
    Creates an IndividualBB dataset:
    1. Crops each image to the union of all digits (GlobalBB ground truth).
    2. Applies sharpening (Upscale + Bilateral + Unsharp Mask) to the crop.
    3. Maps internal digit boxes to the new crop coordinate system.
    4. Saves images and labels in YOLO format.
    """
    import yaml

    train_samples, val_samples = stratified_split(
        samples, split_ratio=split_ratio, seed=seed, categories=categories
    )

    images_dir = os.path.join(individualbb_out_root, "images")
    labels_dir = os.path.join(individualbb_out_root, "labels")
    
    if os.path.exists(individualbb_out_root):
        shutil.rmtree(individualbb_out_root)
        
    for split in ("train", "val"):
        ensure_dir(os.path.join(images_dir, split))
        ensure_dir(os.path.join(labels_dir, split))

    def process_one(split: str, sample: Dict[str, str]) -> int:
        image_path = sample["image_path"]
        anno_path = sample["anno_path"]
        category = sample["category"]
        idx = sample["sample_id"].split("/", 1)[1]
        stem = f"{category}_{idx}"
        
        dst_img_path = os.path.join(images_dir, split, f"{stem}.jpg")
        dst_lbl_path = os.path.join(labels_dir, split, f"{stem}.txt")

        img = cv2.imread(image_path)
        if img is None: return 0
        
        # 1. Get the global bounding boxes and digit boxes
        global_boxes, digit_info, has_digit_boxes, _ = get_gt_from_anno(anno_path)
        
        if not global_boxes or not has_digit_boxes:
            return 0
        
        # Use the first global box found (usually only one for number sequence)
        gx1, gy1, gx2, gy2 = global_boxes[0]
        
        # Add a small margin (5%)
        gw, gh = gx2 - gx1, gy2 - gy1
        margin_x = int(gw * 0.05)
        margin_y = int(gh * 0.05)
        
        H, W = img.shape[:2]
        gx1 = max(0, gx1 - margin_x)
        gy1 = max(0, gy1 - margin_y)
        gx2 = min(W, gx2 + margin_x)
        gy2 = min(H, gy2 + margin_y)
        
        crop = img[int(gy1):int(gy2), int(gx1):int(gx2)]
        if crop.size == 0:
            return 0
            
        # 2. Sharpen the crop
        sharpened = enhance_digit(crop, upscale_factor=2.0)
        
        # 3. Translate digit boxes
        cw = gx2 - gx1
        ch = gy2 - gy1
        
        valid_digits = []
        for digit in digit_info:
            dx1, dy1, dx2, dy2 = digit['bbox']
            # Map to crop coordinate
            nx1 = max(0, dx1 - gx1)
            ny1 = max(0, dy1 - gy1)
            nx2 = min(cw, dx2 - gx1)
            ny2 = min(ch, dy2 - gy1)
            
            # Check if digit is actually inside the crop
            if nx2 > nx1 and ny2 > ny1:
                # YOLO format: cls, x_center, y_center, width, height
                xc = (nx1 + nx2) / 2 / cw
                yc = (ny1 + ny2) / 2 / ch
                bw = (nx2 - nx1) / cw
                bh = (ny2 - ny1) / ch
                valid_digits.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        
        if not valid_digits:
            return 0
            
        # 4. Save
        success = cv2.imwrite(dst_img_path, sharpened)
        if not success:
            return 0
            
        with open(dst_lbl_path, "w") as f:
            f.write("\n".join(valid_digits))
            
        return len(valid_digits)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        count = 0
        for s in tqdm(split_samples, desc=f"Generating Sharpened IndividualBB {split_name}"):
            process_one(split_name, s)
            
    data_yaml_path = os.path.join(individualbb_out_root, "data.yaml")
    # Use absolute paths for training to avoid OneDrive relative path issues
    abs_root = os.path.abspath(individualbb_out_root)
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump({
            "path": abs_root,
            "train": "images/train",
            "val": "images/val",
            "names": ["digit"]
        }, f)
        
    return data_yaml_path

def train_individualbb(data_yaml, output_dir, epochs=20, batch=16, img_size=256, device=""):
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        epochs = trainer.epochs
        metrics = getattr(trainer, 'metrics', {})
        # Extract losses: trainer.loss_items is available in fit_epoch_end
        items = trainer.loss_items
        loss = (sum(items) / len(items)) if len(items) > 0 else 0
        if hasattr(loss, 'item'):
            loss = loss.item()
        map50 = metrics.get('metrics/mAP50(B)', 0)
        print(f"Epoch {epoch}/{epochs}: loss={loss:.4f}, mAP50={map50:.4f}")

    model = YOLO("yolov8n.pt")
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        device=device,
        project=os.path.join(output_dir, "individualbb_runs"),
        name="run1",
        exist_ok=True,
        verbose=True,
        plots=False,
        save=True
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Individual Digit Detection Trainer (Stage 4)")
    parser.add_argument("--dataset-root", default=os.path.join(BASE_DIR, "data", "digits_data"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "outputs", "bbox_comparison"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true", help="Skip training if weights folder exists.")
    parser.add_argument("--force-train", action="store_true", help="Force training even if weights exist.")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate sharpened dataset, no training.")
    parser.add_argument("--train-only", action="store_true", help="Only run training, skip dataset generation.")
    args = parser.parse_args()
    
    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    
    categories = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    print(f"Found datasets for IndividualBB: {categories}")

    samples = iter_new_samples(dataset_root)
    
    individual_out_root = os.path.join(args.output_dir, "individualbb_dataset")
    weights_path = os.path.join(args.output_dir, "individualbb_run", "weights", "best.pt")

    if args.prepare_only:
        make_individualbb_dataset(samples, categories, individual_out_root, 0.8, 42)
        return

    if args.train_only:
        if args.skip_train and os.path.exists(weights_path):
            return
        
        data_yaml = os.path.join(individual_out_root, "data.yaml")
        if os.path.exists(data_yaml):
            train_individualbb(data_yaml, args.output_dir, epochs=args.epochs)
            
            ensure_dir(os.path.join(args.output_dir, "individualbb_run", "weights"))
            best_pt = os.path.join(args.output_dir, "individualbb_runs", "run1", "weights", "best.pt")
            if os.path.exists(best_pt):
                shutil.copy2(best_pt, weights_path)
        return

    # Logic: Only train if weights are missing OR --force-train is specified
    if (not os.path.exists(weights_path)) or args.force_train:
        make_individualbb_dataset(samples, categories, individual_out_root, 0.8, 42)
        data_yaml = os.path.join(individual_out_root, "data.yaml")
        train_individualbb(data_yaml, args.output_dir, epochs=args.epochs)
        
        ensure_dir(os.path.join(args.output_dir, "individualbb_run", "weights"))
        best_pt = os.path.join(args.output_dir, "individualbb_runs", "run1", "weights", "best.pt")
        if os.path.exists(best_pt):
            shutil.copy2(best_pt, weights_path)
    # No extra prints

if __name__ == "__main__":
    main()
