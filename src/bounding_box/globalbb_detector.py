import argparse
import os
import random
import shutil
import time
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Use default logging to allow progress bars
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src"))

from utils.data_utils import iter_new_samples, get_gt_from_anno

def xyxy_to_globalbb_bbox(
    xyxy: Tuple[float, float, float, float], w: int, h: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x_center = (x1 + x2) / 2.0 / w
    y_center = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return float(x_center), float(y_center), float(bw), float(bh)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stratified_split(
    samples: List[Dict[str, str]], split_ratio: float, seed: int, categories: List[str]
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rng = random.Random(seed)
    train: List[Dict[str, str]] = []
    val: List[Dict[str, str]] = []
    for cat in categories:
        cat_samples = [s for s in samples if s["category"] == cat]
        rng.shuffle(cat_samples)
        n_train = int(len(cat_samples) * split_ratio)
        train.extend(cat_samples[:n_train])
        val.extend(cat_samples[n_train:])
    return train, val


def make_globalbb_dataset(
    samples: List[Dict[str, str]],
    categories: List[str],
    dataset_root: str,
    globalbb_out_root: str,
    split_ratio: float,
    seed: int,
) -> str:
    """
    Creates a GlobalBB dataset folder with:
      images/train, images/val, labels/train, labels/val
    Returns the path to the generated data.yaml.
    """
    import yaml

    train_samples, val_samples = stratified_split(
        samples, split_ratio=split_ratio, seed=seed, categories=categories
    )

    images_train_dir = os.path.join(globalbb_out_root, "images", "train")
    images_val_dir = os.path.join(globalbb_out_root, "images", "val")
    labels_train_dir = os.path.join(globalbb_out_root, "labels", "train")
    labels_val_dir = os.path.join(globalbb_out_root, "labels", "val")
    ensure_dir(images_train_dir)
    ensure_dir(images_val_dir)
    ensure_dir(labels_train_dir)
    ensure_dir(labels_val_dir)

    def process_one(split: str, sample: Dict[str, str]) -> bool:
        image_path = sample["image_path"]
        anno_path = sample["anno_path"]
        category = sample["category"]
        idx = sample["sample_id"].split("/", 1)[1]
        stem = f"{category}_{idx}"

        dst_img_dir = images_train_dir if split == "train" else images_val_dir
        dst_lbl_dir = labels_train_dir if split == "train" else labels_val_dir
        dst_img_path = os.path.join(dst_img_dir, f"{stem}.jpg")
        dst_lbl_path = os.path.join(dst_lbl_dir, f"{stem}.txt")

        # Read image to get dimensions.
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]

        global_boxes, _ = get_gt_from_anno(anno_path)

        if not global_boxes:
            return False

        shutil.copy2(image_path, dst_img_path)
        with open(dst_lbl_path, "w", encoding="utf-8", newline="") as f:
            for x1, y1, x2, y2 in global_boxes:
                # Coordinate mapping is already in (x1, y1, x2, y2)
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                box_w = (x2 - x1) / w
                box_h = (y2 - y1) / h
                f.write(f"0 {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}\n")
        return True

    # Clear previous conversions (only inside output folder).
    if os.path.exists(globalbb_out_root):
        shutil.rmtree(globalbb_out_root)
    ensure_dir(images_train_dir)
    ensure_dir(images_val_dir)
    ensure_dir(labels_train_dir)
    ensure_dir(labels_val_dir)
    
    # List to store preview images
    preview_grid = []
    category_counts = {cat: 0 for cat in categories}

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        kept = 0
        for s in tqdm(split_samples, desc=f"Converting {split_name} to GlobalBB", ncols=90):
            ok = process_one(split_name, s)
            if ok:
                kept += 1
                # Capture for preview (up to 3 per category)
                cat = s["category"]
                if category_counts[cat] < 3:
                    img = cv2.imread(s["image_path"])
                    global_boxes, _ = get_gt_from_anno(s["anno_path"])
                    
                    # Draw for preview
                    preview_img = img.copy()
                    for x1, y1, x2, y2 in global_boxes:
                        cv2.rectangle(preview_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Add category label
                    cv2.putText(preview_img, cat, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    preview_grid.append(preview_img)
                    category_counts[cat] += 1
            
        print(f"GlobalBB conversion: kept {kept}/{len(split_samples)} samples in {split_name}")

    if preview_grid:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < len(preview_grid):
                ax.imshow(cv2.cvtColor(preview_grid[i], cv2.COLOR_BGR2RGB))
            ax.axis('off')
        preview_path = os.path.join(os.path.dirname(globalbb_out_root), "preview_labels_before_training.png")
        plt.tight_layout()
        plt.savefig(preview_path)
        plt.close()
        print(f"=> Saved label preview to: {preview_path}")

    data_yaml_path = os.path.join(globalbb_out_root, "data.yaml")
    # Use absolute paths to avoid OneDrive/stale reference issues
    abs_root = os.path.abspath(globalbb_out_root)
    yaml_obj = {
        "path": abs_root,
        "train": "images/train",
        "val": "images/val",
        "names": ["number_sequence"],
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_obj, f)

    return data_yaml_path


def make_digit_globalbb_dataset(
    samples: List[Dict[str, str]],
    categories: List[str],
    globalbb_out_root: str,
    split_ratio: float,
    seed: int,
    min_area: int = 10,
) -> str:
    """Create a GlobalBB dataset where every digit gets its own label box.

    Each mask.png is scanned contour-by-contour. Images with exactly one digit
    get one label line; images with N digits get N label lines. Images whose mask
    yields zero valid contours are skipped entirely.

    This dataset is used for the second-stage digit detector (applied to the
    sharpened crops produced by the first GlobalBB).
    """
    import yaml

    train_samples, val_samples = stratified_split(
        samples, split_ratio=split_ratio, seed=seed, categories=categories
    )

    if os.path.exists(globalbb_out_root):
        shutil.rmtree(globalbb_out_root)
    for split in ("train", "val"):
        ensure_dir(os.path.join(globalbb_out_root, "images", split))
        ensure_dir(os.path.join(globalbb_out_root, "labels", split))

    def process_one(split: str, sample: Dict[str, str]) -> int:
        image_path = sample["image_path"]
        anno_path = sample["anno_path"]
        category = sample["category"]
        idx = sample["sample_id"].split("/", 1)[1]
        stem = f"{category}_{idx}"
        dst_img = os.path.join(globalbb_out_root, "images", split, f"{stem}.jpg")
        dst_lbl = os.path.join(globalbb_out_root, "labels", split, f"{stem}.txt")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return 0
        h, w = img.shape[:2]

        _, digit_info = get_gt_from_anno(anno_path)

        # Skip images where the mask contains no valid digit regions.
        if not digit_info:
            return 0

        shutil.copy2(image_path, dst_img)
        with open(dst_lbl, "w", encoding="utf-8", newline="") as f:
            for digit in digit_info:
                x1, y1, x2, y2 = digit['bbox']
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw_n = (x2 - x1) / w
                bh_n = (y2 - y1) / h
                f.write(f"0 {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")
        return len(digit_info)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        kept = 0
        total_digits = 0
        for s in tqdm(split_samples, desc=f"Converting {split_name} (per-digit)", ncols=90):
            n = process_one(split_name, s)
            if n > 0:
                kept += 1
                total_digits += n
        print(
            f"Digit dataset [{split_name}]: {kept}/{len(split_samples)} images, "
            f"{total_digits} digit labels "
            f"({'avg ' + f'{total_digits/kept:.1f}' if kept else 'n/a'} digits/image)"
        )

    data_yaml_path = os.path.join(globalbb_out_root, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"path": globalbb_out_root, "train": "images/train", "val": "images/val", "names": ["digit"]},
            f,
        )
    return data_yaml_path


def globalbb_predict_all(
    model,
    samples: List[Dict[str, str]],
    categories: List[str],
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    output_csv: str,
) -> None:
    """Run Batch Inference in small chunks to maximize GPU utility while preventing OOM errors."""
    rows: List[Dict[str, object]] = []
    chunk_size = 100 # Reduced to 100 for safer memory management
    
    print(f"Starting GPU-accelerated chunked inference (Total: {len(samples)} images, Chunk Size: {chunk_size})...")
    
    # We use a master tqdm bar to track overall progress across all chunks
    pbar = tqdm(total=len(samples), desc="GlobalBB Batch Inference", ncols=90)
    
    try:
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i : i + chunk_size]
            chunk_paths = [s["image_path"] for s in chunk]
            
            # Use 'stream=True' to keep memory usage minimal
            results_gen = model.predict(
                source=chunk_paths,
                imgsz=img_size,
                conf=conf_thres,
                iou=iou_thres,
                stream=True,
                verbose=False
            )

            # Iterating through the generator processes one chunk
            for results, sample in zip(results_gen, chunk):
                if results.boxes and len(results.boxes) > 0:
                    boxes = results.boxes
                    confs = boxes.conf.detach().cpu().numpy()
                    coords = boxes.xyxy.detach().cpu().numpy()

                    for k in range(len(confs)):
                        rows.append({
                            "sample_id": sample["sample_id"],
                            "category": sample["category"],
                            "image_path": sample["image_path"],
                            "pred_x1": float(coords[k][0]),
                            "pred_y1": float(coords[k][1]),
                            "pred_x2": float(coords[k][2]),
                            "pred_y2": float(coords[k][3]),
                            "pred_conf": float(confs[k]),
                        })
                else:
                    rows.append({
                        "sample_id": sample["sample_id"], 
                        "category": sample["category"], 
                        "image_path": sample["image_path"],
                        "pred_x1": None, "pred_y1": None, "pred_x2": None, "pred_y2": None,
                        "pred_conf": None
                    })
                
                pbar.update(1)
                
    except Exception as e:
        print(f"\nCRITICAL ERROR during batch inference: {e}")
        pbar.close()
        sys.exit(1)

    pbar.close()
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Batch inference complete. Results saved to {output_csv}")


def save_digit_gt_boxes(samples: List[Dict[str, str]], output_csv: str, min_area: int = 10) -> None:
    """Save per-digit ground-truth boxes extracted from annotations.json."""
    rows: List[Dict[str, object]] = []
    for s in tqdm(samples, desc="Extracting digit GT boxes from annotations", ncols=90):
        _, digit_info = get_gt_from_anno(s["anno_path"])
        for idx, digit in enumerate(digit_info):
            x1, y1, x2, y2 = digit['bbox']
            rows.append({
                "sample_id": s["sample_id"],
                "category": s["category"],
                "image_path": s["image_path"],
                "digit_idx": idx,
                "gt_x1": x1, "gt_y1": y1, "gt_x2": x2, "gt_y2": y2,
                "label": digit['label']
            })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="GlobalBB bounding-box detector from masks")
    parser.add_argument(
        "--dataset-root",
        default=os.path.join("data", "digits_data"),
        help="Path to data/digits_data root.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "bbox_comparison"),
        help="Where to store GlobalBB dataset conversion, weights, and predictions.",
    )
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--globalbb-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="",
                        help="Examples: 'cpu', '0', '0,1'. Empty auto-detect.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if weights folder exists.")
    parser.add_argument("--force-train", action="store_true", help="Force training even if weights exist.")
    parser.add_argument(
        "--overwrite-conversion",
        action="store_true",
        help="Force regeneration of GlobalBB dataset conversion.",
    )
    args = parser.parse_args()

    # Discover categories dynamically from folders in dataset_root
    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    
    categories = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    print(f"Found datasets: {categories}")

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    samples = iter_new_samples(dataset_root)

    # Convert masks -> GlobalBB labels and create a GlobalBB dataset for training.
    globalbb_out_root = os.path.join(output_dir, "globalbb_dataset")
    conversion_yaml_path = os.path.join(globalbb_out_root, "data.yaml")
    if args.overwrite_conversion or not os.path.exists(conversion_yaml_path):
        print("Creating GlobalBB dataset conversion...")
        data_yaml_path = make_globalbb_dataset(
            samples=samples,
            categories=categories,
            dataset_root=dataset_root,
            globalbb_out_root=globalbb_out_root,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
    else:
        data_yaml_path = conversion_yaml_path

    # Build per-digit GlobalBB dataset (one label line per digit) for the second-stage detector.
    digit_globalbb_out_root = os.path.join(output_dir, "globalbb_digit_dataset")
    digit_conversion_yaml = os.path.join(digit_globalbb_out_root, "data.yaml")
    if args.overwrite_conversion or not os.path.exists(digit_conversion_yaml):
        print("Creating per-digit GlobalBB dataset (one box per digit)...")
        make_digit_globalbb_dataset(
            samples=samples,
            categories=categories,
            globalbb_out_root=digit_globalbb_out_root,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
    else:
        print(f"Per-digit GlobalBB dataset already exists: {digit_globalbb_out_root}")

    # Save per-digit GT boxes as a CSV (used for final evaluation after step 4,
    # NOT as GlobalBB training labels — those use a single union box per image).
    digit_gt_csv = os.path.join(output_dir, "digit_gt_boxes.csv")
    if args.overwrite_conversion or not os.path.exists(digit_gt_csv):
        print("Extracting per-digit ground-truth boxes from masks...")
        save_digit_gt_boxes(samples, digit_gt_csv)
        print(f"Digit GT boxes saved to: {digit_gt_csv}")

    # Train GlobalBB.
    globalbb_weights_dir = os.path.join(output_dir, "globalbb_run")
    best_pt_path = os.path.join(globalbb_weights_dir, "weights", "best.pt")

    if (not os.path.exists(best_pt_path)) or args.force_train:
        if os.path.exists(best_pt_path):
            print("=> Existing weights found, but --force-train is enabled. Retraining...")
        else:
            print("=> No existing weights found. Starting training...")
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing GlobalBB dependency. Please run `pip install ultralytics`."
            ) from e

        if args.device:
            device = args.device
        else:
            try:
                import torch

                device = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        print("Training GlobalBB detector...")
        model = YOLO(args.globalbb_weights)
        # ultralytics creates a run directory; we direct its base save_dir for stability.
        results = model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            device=device,
            project=os.path.join(output_dir, "globalbb_runs"),
            name="run1",
            exist_ok=True,
            verbose=True,
        )

        # best weights path is typically: runs/detect/train*/weights/best.pt
        # but we keep it resilient by checking known locations.
        inferred_best = None
        try:
            save_dir = results.save_dir  # type: ignore[attr-defined]
            candidate = os.path.join(save_dir, "weights", "best.pt")
            if os.path.exists(candidate):
                inferred_best = candidate
        except Exception:
            inferred_best = None

        if inferred_best is None or not os.path.exists(inferred_best):
            # Fallback to ultralytics default layout within globalbb_out_root.
            # If this fails, user will see missing file error below.
            inferred_best = best_pt_path

        ensure_dir(globalbb_weights_dir)
        # Copy best weights into our stable path.
        if inferred_best != best_pt_path and os.path.exists(inferred_best):
            ensure_dir(os.path.dirname(best_pt_path))
            shutil.copy2(inferred_best, best_pt_path)
    else:
        print(f"Skipping training; using existing weights: {best_pt_path}")

    if not os.path.exists(best_pt_path):
        raise FileNotFoundError(
            f"Could not locate GlobalBB best weights at: {best_pt_path}. "
            f"Run without `--skip-train` to train."
        )

    # Inference on the full dataset.
    from ultralytics import YOLO  # type: ignore

    print("Running GlobalBB inference on full dataset...")
    model = YOLO(best_pt_path)

    output_csv = os.path.join(output_dir, "globalbb_predictions.csv")
    if os.path.exists(output_csv) and not (args.force_train or (not args.skip_train and 'best_pt_path' in locals() and os.path.exists(best_pt_path))):
         # If we just trained, we should probably run inference anyway to be safe, 
         # but if we skipped training and csv exists, we skip inference.
         if args.skip_train:
             print(f"=> Found existing GlobalBB predictions at {output_csv}. Skipping inference.")
             return

    globalbb_predict_all(
        model=model,
        samples=samples,
        categories=categories,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.nms_iou,
        output_csv=output_csv,
    )
    print(f"GlobalBB predictions saved to: {output_csv}")


if __name__ == "__main__":
    main()

