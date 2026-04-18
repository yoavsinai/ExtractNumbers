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


def _read_mask_grayscale(mask_path: str) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        # Convert color masks to grayscale.
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Returns a single union bbox (x1,y1,x2,y2) over all non-zero pixels."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1


def extract_digit_bboxes(mask: np.ndarray, min_area: int = 10) -> List[Tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) for each individual digit contour, sorted left-to-right.

    Used only for final evaluation ground truth — NOT for YOLO training labels.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw * bh >= min_area:
            boxes.append((bx, by, bx + bw, by + bh))
    return sorted(boxes, key=lambda b: b[0])


def xyxy_to_yolo_bbox(
    xyxy: Tuple[int, int, int, int], w: int, h: int
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    x_center = (x1 + x2) / 2.0 / w
    y_center = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return float(x_center), float(y_center), float(bw), float(bh)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iter_samples(dataset_root: str, categories: List[str]) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    for cat in categories:
        cat_dir = os.path.join(dataset_root, cat)
        if not os.path.isdir(cat_dir):
            raise FileNotFoundError(f"Missing category directory: {cat_dir}")
        for folder in sorted(os.listdir(cat_dir)):
            sample_dir = os.path.join(cat_dir, folder)
            if not os.path.isdir(sample_dir):
                continue
            image_path = os.path.join(sample_dir, "image.jpg")
            mask_path = os.path.join(sample_dir, "mask.png")
            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                continue
            sample_id = f"{cat}/{folder}"
            samples.append(
                {
                    "category": cat,
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "mask_path": mask_path,
                }
            )
    if not samples:
        raise RuntimeError(f"No samples found under: {dataset_root}")
    return samples


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


def make_yolo_dataset(
    samples: List[Dict[str, str]],
    categories: List[str],
    dataset_root: str,
    yolo_out_root: str,
    split_ratio: float,
    seed: int,
) -> str:
    """
    Creates a YOLO dataset folder with:
      images/train, images/val, labels/train, labels/val
    Returns the path to the generated data.yaml.
    """
    import yaml

    train_samples, val_samples = stratified_split(
        samples, split_ratio=split_ratio, seed=seed, categories=categories
    )

    images_train_dir = os.path.join(yolo_out_root, "images", "train")
    images_val_dir = os.path.join(yolo_out_root, "images", "val")
    labels_train_dir = os.path.join(yolo_out_root, "labels", "train")
    labels_val_dir = os.path.join(yolo_out_root, "labels", "val")
    ensure_dir(images_train_dir)
    ensure_dir(images_val_dir)
    ensure_dir(labels_train_dir)
    ensure_dir(labels_val_dir)

    def process_one(split: str, sample: Dict[str, str]) -> bool:
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
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

        mask = _read_mask_grayscale(mask_path)

        # Apply dilation to merge nearby disjoint characters into a single global bounding box
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return False

        valid_boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw * bh >= 10:  # Filter out tiny noisy boxes.
                valid_boxes.append((bx, by, bw, bh))

        # Since multiple digits now merge into a single bounding box sequence...
        if len(valid_boxes) == 0:
            return False

        shutil.copy2(image_path, dst_img_path)
        with open(dst_lbl_path, "w", encoding="utf-8", newline="") as f:
            for bx, by, bw, bh in valid_boxes:
                xc = (bx + bw / 2) / w
                yc = (by + bh / 2) / h
                box_w = bw / w
                box_h = bh / h
                f.write(f"0 {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}\n")
        return True

    # Clear previous conversions (only inside output folder).
    if os.path.exists(yolo_out_root):
        shutil.rmtree(yolo_out_root)
    ensure_dir(images_train_dir)
    ensure_dir(images_val_dir)
    ensure_dir(labels_train_dir)
    ensure_dir(labels_val_dir)

    for split_name, split_samples in [("train", train_samples), ("val", val_samples)]:
        kept = 0
        for s in tqdm(split_samples, desc=f"Converting {split_name} to YOLO", ncols=90):
            ok = process_one(split_name, s)
            kept += int(ok)
        print(f"YOLO conversion: kept {kept}/{len(split_samples)} samples in {split_name}")

    data_yaml_path = os.path.join(yolo_out_root, "data.yaml")
    yaml_obj = {
        "path": yolo_out_root,
        "train": "images/train",
        "val": "images/val",
        "names": ["number_sequence"],
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_obj, f)

    return data_yaml_path


def make_digit_yolo_dataset(
    samples: List[Dict[str, str]],
    categories: List[str],
    yolo_out_root: str,
    split_ratio: float,
    seed: int,
    min_area: int = 10,
) -> str:
    """Create a YOLO dataset where every digit gets its own label box.

    Each mask.png is scanned contour-by-contour. Images with exactly one digit
    get one label line; images with N digits get N label lines. Images whose mask
    yields zero valid contours are skipped entirely.

    This dataset is used for the second-stage digit detector (applied to the
    sharpened crops produced by the first YOLO).
    """
    import yaml

    train_samples, val_samples = stratified_split(
        samples, split_ratio=split_ratio, seed=seed, categories=categories
    )

    if os.path.exists(yolo_out_root):
        shutil.rmtree(yolo_out_root)
    for split in ("train", "val"):
        ensure_dir(os.path.join(yolo_out_root, "images", split))
        ensure_dir(os.path.join(yolo_out_root, "labels", split))

    def process_one(split: str, sample: Dict[str, str]) -> int:
        image_path = sample["image_path"]
        mask_path = sample["mask_path"]
        category = sample["category"]
        idx = sample["sample_id"].split("/", 1)[1]
        stem = f"{category}_{idx}"
        dst_img = os.path.join(yolo_out_root, "images", split, f"{stem}.jpg")
        dst_lbl = os.path.join(yolo_out_root, "labels", split, f"{stem}.txt")

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            return 0
        h, w = img.shape[:2]

        mask = _read_mask_grayscale(mask_path)
        digit_boxes = extract_digit_bboxes(mask, min_area=min_area)

        # Skip images where the mask contains no valid digit regions.
        if not digit_boxes:
            return 0

        shutil.copy2(image_path, dst_img)
        with open(dst_lbl, "w", encoding="utf-8", newline="") as f:
            for x1, y1, x2, y2 in digit_boxes:
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw_n = (x2 - x1) / w
                bh_n = (y2 - y1) / h
                f.write(f"0 {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")
        return len(digit_boxes)

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

    data_yaml_path = os.path.join(yolo_out_root, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"path": yolo_out_root, "train": "images/train", "val": "images/val", "names": ["digit"]},
            f,
        )
    return data_yaml_path


def yolo_predict_all(
    model,
    samples: List[Dict[str, str]],
    categories: List[str],
    img_size: int,
    conf_thres: float,
    iou_thres: float,
    output_csv: str,
) -> None:
    rows: List[Dict[str, object]] = []
    
    for s in tqdm(samples, desc="YOLO inference (Multi-box)", ncols=90):
        image_path = s["image_path"]
        category = s["category"]
        sample_id = s["sample_id"]

        t0 = time.perf_counter()
        results = model.predict(
            source=image_path,
            imgsz=img_size,
            conf=conf_thres,
            iou=iou_thres,
            verbose=False,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.detach().cpu().numpy()
            coords = boxes.xyxy.detach().cpu().numpy() # [x1, y1, x2, y2]

            for i in range(len(confs)):
                rows.append({
                    "sample_id": sample_id,
                    "category": category,
                    "image_path": image_path,
                    "pred_x1": float(coords[i][0]),
                    "pred_y1": float(coords[i][1]),
                    "pred_x2": float(coords[i][2]),
                    "pred_y2": float(coords[i][3]),
                    "pred_conf": float(confs[i]),
                    "inference_time_ms": dt_ms
                })
        else:
            rows.append({
                "sample_id": sample_id, "category": category, "image_path": image_path,
                "pred_x1": None, "pred_y1": None, "pred_x2": None, "pred_y2": None,
                "pred_conf": None, "inference_time_ms": dt_ms
            })

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def save_digit_gt_boxes(samples: List[Dict[str, str]], output_csv: str, min_area: int = 10) -> None:
    """Save per-digit ground-truth boxes extracted from masks.

    Each row is one digit bbox. Used for final evaluation after step 4 (post-sharpening
    digit detection), NOT for YOLO training.
    """
    rows: List[Dict[str, object]] = []
    for s in tqdm(samples, desc="Extracting digit GT boxes from masks", ncols=90):
        mask = _read_mask_grayscale(s["mask_path"])
        for idx, (x1, y1, x2, y2) in enumerate(extract_digit_bboxes(mask, min_area=min_area)):
            rows.append({
                "sample_id": s["sample_id"],
                "category": s["category"],
                "image_path": s["image_path"],
                "digit_idx": idx,
                "gt_x1": x1, "gt_y1": y1, "gt_x2": x2, "gt_y2": y2,
            })
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO bounding-box detector from masks")
    parser.add_argument(
        "--dataset-root",
        default=os.path.join("data", "segmentation"),
        help="Path to data/segmentation root.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("outputs", "bbox_comparison"),
        help="Where to store YOLO dataset conversion, weights, and predictions.",
    )
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--yolo-weights", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="",
                        help="Examples: 'cpu', '0', '0,1'. Empty auto-detect.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run inference using saved weights.",
    )
    parser.add_argument(
        "--overwrite-conversion",
        action="store_true",
        help="Force regeneration of YOLO dataset conversion.",
    )
    args = parser.parse_args()

    categories = ["natural", "synthetic", "handwritten"]
    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    samples = iter_samples(dataset_root, categories=categories)

    # Convert masks -> YOLO labels and create a YOLO dataset for training.
    yolo_out_root = os.path.join(output_dir, "yolo_dataset")
    conversion_yaml_path = os.path.join(yolo_out_root, "data.yaml")
    if args.overwrite_conversion or not os.path.exists(conversion_yaml_path):
        print("Creating YOLO dataset conversion...")
        data_yaml_path = make_yolo_dataset(
            samples=samples,
            categories=categories,
            dataset_root=dataset_root,
            yolo_out_root=yolo_out_root,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
    else:
        data_yaml_path = conversion_yaml_path

    # Build per-digit YOLO dataset (one label line per digit) for the second-stage detector.
    digit_yolo_out_root = os.path.join(output_dir, "yolo_digit_dataset")
    digit_conversion_yaml = os.path.join(digit_yolo_out_root, "data.yaml")
    if args.overwrite_conversion or not os.path.exists(digit_conversion_yaml):
        print("Creating per-digit YOLO dataset (one box per digit)...")
        make_digit_yolo_dataset(
            samples=samples,
            categories=categories,
            yolo_out_root=digit_yolo_out_root,
            split_ratio=args.split_ratio,
            seed=args.seed,
        )
    else:
        print(f"Per-digit YOLO dataset already exists: {digit_yolo_out_root}")

    # Save per-digit GT boxes as a CSV (used for final evaluation after step 4,
    # NOT as YOLO training labels — those use a single union box per image).
    digit_gt_csv = os.path.join(output_dir, "digit_gt_boxes.csv")
    if args.overwrite_conversion or not os.path.exists(digit_gt_csv):
        print("Extracting per-digit ground-truth boxes from masks...")
        save_digit_gt_boxes(samples, digit_gt_csv)
        print(f"Digit GT boxes saved to: {digit_gt_csv}")

    # Train YOLO.
    yolo_weights_dir = os.path.join(output_dir, "yolo_run")
    best_pt_path = os.path.join(yolo_weights_dir, "weights", "best.pt")

    if not args.skip_train or not os.path.exists(best_pt_path):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing YOLO dependency. Please run `pip install ultralytics`."
            ) from e

        if args.device:
            device = args.device
        else:
            try:
                import torch

                device = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        print("Training YOLO detector...")
        model = YOLO(args.yolo_weights)
        # ultralytics creates a run directory; we direct its base save_dir for stability.
        results = model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            device=device,
            project=os.path.join(output_dir, "yolo_runs"),
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
            # Fallback to ultralytics default layout within yolo_out_root.
            # If this fails, user will see missing file error below.
            inferred_best = best_pt_path

        ensure_dir(yolo_weights_dir)
        # Copy best weights into our stable path.
        if inferred_best != best_pt_path and os.path.exists(inferred_best):
            ensure_dir(os.path.dirname(best_pt_path))
            shutil.copy2(inferred_best, best_pt_path)
    else:
        print(f"Skipping training; using existing weights: {best_pt_path}")

    if not os.path.exists(best_pt_path):
        raise FileNotFoundError(
            f"Could not locate YOLO best weights at: {best_pt_path}. "
            f"Run without `--skip-train` to train."
        )

    # Inference on the full dataset.
    from ultralytics import YOLO  # type: ignore

    print("Running YOLO inference on full dataset...")
    model = YOLO(best_pt_path)

    output_csv = os.path.join(output_dir, "yolo_predictions.csv")
    yolo_predict_all(
        model=model,
        samples=samples,
        categories=categories,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.nms_iou,
        output_csv=output_csv,
    )
    print(f"YOLO predictions saved to: {output_csv}")


if __name__ == "__main__":
    main()

