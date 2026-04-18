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
# Silence logging and warnings
os.environ['TQDM_DISABLE'] = '1'
os.environ['ULTRALYTICS_VERBOSE'] = 'False'
warnings.filterwarnings('ignore')
try:
    set_logging('ultralytics', verbose=False)
    LOGGER.setLevel(logging.ERROR)
except NameError:
    pass
# Ensure we can import from src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import apply_unsharp_mask, upscale_image, apply_bilateral_filter
from bounding_box.globalbb_detector import _read_mask_grayscale, bbox_from_mask, extract_digit_bboxes, ensure_dir, stratified_split

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
        mask_path = sample["mask_path"]
        category = sample["category"]
        idx = sample["sample_id"].split("/", 1)[1]
        stem = f"{category}_{idx}"
        
        dst_img_path = os.path.join(images_dir, split, f"{stem}.jpg")
        dst_lbl_path = os.path.join(labels_dir, split, f"{stem}.txt")

        img = cv2.imread(image_path)
        mask = _read_mask_grayscale(mask_path)
        
        # 1. Get the global bounding box (union of all digits)
        # We use the morphological dilation logic to match Stage 1 ground truth
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        g_bbox = bbox_from_mask(dilated_mask)
        
        if not g_bbox:
            return 0
        
        gx1, gy1, gx2, gy2 = g_bbox
        # Add a small margin (5%)
        gw, gh = gx2 - gx1, gy2 - gy1
        margin_x = int(gw * 0.05)
        margin_y = int(gh * 0.05)
        
        H, W = img.shape[:2]
        gx1 = max(0, gx1 - margin_x)
        gy1 = max(0, gy1 - margin_y)
        gx2 = min(W, gx2 + margin_x)
        gy2 = min(H, gy2 + margin_y)
        
        crop = img[gy1:gy2, gx1:gx2]
        if crop.size == 0:
            return 0
            
        # 2. Sharpen the crop
        # We use color sharpening for YOLO feature extraction
        sharpened = upscale_image(crop, scale_factor=2.0)
        sharpened = apply_bilateral_filter(sharpened)
        sharpened = apply_unsharp_mask(sharpened, strength=2.0)
        
        # 3. Get individual digit boxes and translate
        digit_boxes = extract_digit_bboxes(mask, min_area=min_area)
        if not digit_boxes:
            return 0
            
        cw = gx2 - gx1
        ch = gy2 - gy1
        
        valid_digits = []
        for (dx1, dy1, dx2, dy2) in digit_boxes:
            # Map to crop coordinate
            nx1 = max(0, dx1 - gx1)
            ny1 = max(0, dy1 - gy1)
            nx2 = min(cw, dx2 - gx1)
            ny2 = min(ch, dy2 - gy1)
            
            # Check if digit is actually inside the crop
            if nx2 > nx1 and ny2 > ny1:
                # YOLO format: cls, x_center, y_center, width, height (normalized to crop size)
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
        for s in split_samples:
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
        verbose=False,
        plots=False,
        save=True
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Individual Digit Detection Trainer (Stage 4)")
    parser.add_argument("--dataset-root", default=os.path.join(BASE_DIR, "data", "segmentation"))
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "outputs", "bbox_comparison"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate sharpened dataset, no training.")
    parser.add_argument("--train-only", action="store_true", help="Only run training, skip dataset generation.")
    args = parser.parse_args()

    categories = ["natural", "synthetic", "handwritten"]
    from bounding_box.globalbb_detector import iter_samples
    samples = iter_samples(args.dataset_root, categories)
    
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

    if not args.skip_train or not os.path.exists(weights_path):
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
