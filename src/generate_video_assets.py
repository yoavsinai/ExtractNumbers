"""
generate_video_assets.py
========================
Picks 3 sample images from each data type (SVHN, Race Numbers, Handwritten)
— 9 images total — and runs each through the full 4-stage pipeline, saving:

  video_assets/
    01_samples/          – the 9 raw input images
    02_global_bb/        – raw image with the GlobalBB rectangle drawn
    03_sharpened/        – enhanced / sharpened crop
    04_individual_bb/    – sharpened crop with per-digit boxes drawn
    05_classification/   – sharpened crop annotated with digit labels

Usage
-----
  python src/generate_video_assets.py [--model-dir outputs/trained_models] [--data-root data/digits_data] [--out-dir video_assets]
"""

import argparse
import os
import sys
import random

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from utils.data_utils import iter_new_samples
from image_preprocessing.digit_preprocessor import enhance_digit
from digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop

# ---------------------------------------------------------------------------
# Data type definitions
# ---------------------------------------------------------------------------

DATA_TYPES = ["SVHN", "Race Numbers", "Handwritten"]
SAMPLES_PER_TYPE = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def draw_boxes(img_bgr: np.ndarray, boxes, color=(0, 255, 0), thickness=3, labels=None) -> np.ndarray:
    out = img_bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels and i < len(labels):
            cv2.putText(out, str(labels[i]), (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return out


def pick_samples(data_root: str, seed: int = 42) -> list:
    """Return exactly SAMPLES_PER_TYPE images for each data type in DATA_TYPES.

    Samples are chosen randomly (controlled by *seed*) from the pool of
    available images whose ``category`` field matches each data type name
    (case-insensitive substring match is used as a fallback so minor naming
    differences do not break the selection).
    """
    all_samples = iter_new_samples(data_root)
    if not all_samples:
        raise FileNotFoundError(f"No samples found under: {data_root}")

    rng = random.Random(seed)

    # Group by category
    by_cat: dict = {}
    for s in all_samples:
        by_cat.setdefault(s["category"], []).append(s)

    chosen = []
    for dtype in DATA_TYPES:
        # Try exact match first, then case-insensitive substring match
        pool = by_cat.get(dtype)
        if pool is None:
            for cat, samples in by_cat.items():
                if dtype.lower() in cat.lower() or cat.lower() in dtype.lower():
                    pool = samples
                    break

        if not pool:
            print(f"  WARN: no samples found for data type '{dtype}' — skipping.")
            continue

        rng.shuffle(pool)
        selected = pool[:SAMPLES_PER_TYPE]
        if len(selected) < SAMPLES_PER_TYPE:
            print(
                f"  WARN: only {len(selected)} sample(s) available for '{dtype}' "
                f"(expected {SAMPLES_PER_TYPE})."
            )
        chosen.extend(selected)

    return chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate video assets for the pipeline demo.")
    parser.add_argument("--model-dir",  default=os.path.join(BASE_DIR, "outputs", "trained_models"))
    parser.add_argument("--data-root",  default=os.path.join(BASE_DIR, "data", "digits_data"))
    parser.add_argument("--out-dir",    default=os.path.join(BASE_DIR, "video_assets"))
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Verify models exist
    # -----------------------------------------------------------------------
    GLOBAL_PT    = os.path.join(args.model_dir, "globalbb.pt")
    INDIV_PT     = os.path.join(args.model_dir, "individualbb.pt")
    CLASSIF_PT   = os.path.join(args.model_dir, "digit_recognizer.pt")

    missing = [p for p in [GLOBAL_PT, INDIV_PT, CLASSIF_PT] if not os.path.exists(p)]
    if missing:
        print("ERROR – trained model files not found:")
        for m in missing:
            print(f"  {m}")
        print("Run the training pipeline first (python src/training/train_pipeline.py).")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Load models
    # -----------------------------------------------------------------------
    print("Loading models …")
    from ultralytics import YOLO
    global_model = YOLO(GLOBAL_PT)
    indiv_model  = YOLO(INDIV_PT)

    device = get_device()
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIF_PT, map_location=device))
    classifier.to(device).eval()
    print("  ✓ All models loaded.")

    # -----------------------------------------------------------------------
    # 3. Pick samples (3 per data type, 9 total)
    # -----------------------------------------------------------------------
    samples = pick_samples(args.data_root, seed=args.seed)
    print(f"\nSelected {len(samples)} samples ({SAMPLES_PER_TYPE} per data type):")
    for s in samples:
        print(f"  [{s['category']}]  {s['image_path']}")

    # -----------------------------------------------------------------------
    # 4. Output directories
    # -----------------------------------------------------------------------
    dirs = {
        "raw":      ensure(os.path.join(args.out_dir, "01_samples")),
        "global":   ensure(os.path.join(args.out_dir, "02_global_bb")),
        "sharp":    ensure(os.path.join(args.out_dir, "03_sharpened")),
        "indiv":    ensure(os.path.join(args.out_dir, "04_individual_bb")),
        "classif":  ensure(os.path.join(args.out_dir, "05_classification")),
    }

    # -----------------------------------------------------------------------
    # 5. Per-sample pipeline
    # -----------------------------------------------------------------------
    for idx, s in enumerate(samples, start=1):
        tag  = f"sample_{idx:02d}_{s['category']}"
        img  = cv2.imread(s["image_path"])
        if img is None:
            print(f"  WARN: cannot read {s['image_path']} – skipping.")
            continue

        # ── Save raw image ──────────────────────────────────────────────────
        raw_path = os.path.join(dirs["raw"], f"{tag}.png")
        cv2.imwrite(raw_path, img)

        # ── Stage 1 : GlobalBB ──────────────────────────────────────────────
        res1 = global_model.predict(source=img, imgsz=256, verbose=False)
        pred_global = None
        global_conf = 0.0
        if res1 and len(res1[0].boxes) > 0:
            best_idx    = res1[0].boxes.conf.argmax().item()
            pred_global = res1[0].boxes.xyxy[best_idx].cpu().numpy()
            global_conf = float(res1[0].boxes.conf[best_idx].item())

        if pred_global is None:
            print(f"  [{tag}] GlobalBB: no detection – skipping remaining stages.")
            continue

        gx1, gy1, gx2, gy2 = map(int, pred_global)
        H, W = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(W, gx2), min(H, gy2)

        # Annotate with thick green rectangle + confidence
        global_vis = draw_boxes(img, [[gx1, gy1, gx2, gy2]],
                                 color=(0, 200, 0), thickness=4)
        cv2.putText(global_vis, f"GlobalBB  conf={global_conf:.2f}",
                    (gx1, max(gy1 - 10, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imwrite(os.path.join(dirs["global"], f"{tag}.png"), global_vis)

        # ── Stage 2 : Sharpening ────────────────────────────────────────────
        crop   = img[gy1:gy2, gx1:gx2]
        sharp  = enhance_digit(crop, upscale_factor=2.0)
        cv2.imwrite(os.path.join(dirs["sharp"], f"{tag}.png"), sharp)

        # ── Stage 3 : IndividualBB ──────────────────────────────────────────
        res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
        iboxes = []
        if res2 and len(res2[0].boxes) > 0:
            iboxes = res2[0].boxes.xyxy.cpu().numpy().tolist()
            iboxes = sorted(iboxes, key=lambda b: b[0])

        indiv_vis = draw_boxes(sharp, iboxes, color=(255, 200, 0), thickness=2)
        cv2.imwrite(os.path.join(dirs["indiv"], f"{tag}.png"), indiv_vis)

        # ── Stage 4 : Classification ────────────────────────────────────────
        digits = []
        classif_vis = sharp.copy()
        for ibox in iboxes:
            try:
                t = preprocess_crop(sharp, (ibox[0], ibox[1], ibox[2], ibox[3]))
                with torch.no_grad():
                    out   = classifier(t.unsqueeze(0).to(device))
                    probs = torch.softmax(out, dim=1)
                    digit = int(probs.argmax().item())
                    conf  = float(probs.max().item())
                digits.append({"digit": digit, "conf": conf})
                x1, y1 = int(ibox[0]), int(ibox[1])
                x2, y2 = int(ibox[2]), int(ibox[3])
                cv2.rectangle(classif_vis, (x1, y1), (x2, y2), (0, 100, 255), 2)
                cv2.putText(classif_vis, str(digit),
                            (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)
            except Exception as e:
                print(f"    WARN classification: {e}")

        predicted_number = "".join(str(d["digit"]) for d in digits)

        # Add final prediction banner
        banner_h = 40
        banner   = np.zeros((banner_h, classif_vis.shape[1], 3), dtype=np.uint8)
        cv2.putText(banner, f"Predicted: {predicted_number}",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 120), 2)
        classif_vis = np.vstack([classif_vis, banner])
        cv2.imwrite(os.path.join(dirs["classif"], f"{tag}.png"), classif_vis)

        print(f"  [{tag}]  predicted: {predicted_number}")

    print("\nDone! All video assets are in:", args.out_dir)


if __name__ == "__main__":
    main()
