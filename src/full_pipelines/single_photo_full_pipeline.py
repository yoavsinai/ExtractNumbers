import sys
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.models as models

# Import from existing modules (assuming they are in the same project)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BoundingBox'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DigitRecognizer'))

from yolo_detector import xyxy_to_yolo_bbox  # if needed, but we'll use ultralytics directly
from digit_recognizer import build_digit_model, preprocess_crop, get_device

def load_yolo_model(weights_path):
    try:
        from ultralytics import YOLO
        return YOLO(weights_path)
    except ImportError:
        raise RuntimeError("YOLO not available. Install ultralytics.")

def load_digit_model(model_path):
    model = build_digit_model()
    device = get_device()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def run_yolo_on_image(yolo_model, image_path, conf_thres=0.25, iou_thres=0.7):
    results = yolo_model.predict(
        source=image_path,
        imgsz=256,
        conf=conf_thres,
        iou=iou_thres,
        verbose=False,
    )
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        coords = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        return coords, confs
    return [], []

def recognize_digits(digit_model, image_path, bboxes):
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    all_digits = []
    device = get_device()
    
    # Sort yolo boxes left to right
    bboxes = sorted(bboxes, key=lambda b: b[0])

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        
        try:
            # Preprocess and recognize the digit
            inputs = preprocess_crop(img, (x1, y1, x2, y2)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = digit_model(inputs)
                probs = torch.softmax(out, dim=-1)
                pred = int(probs.argmax(dim=-1).item())
                conf = float(probs.max().item())
            all_digits.append((pred, conf))
        except Exception as e:
            print(f"Skipping digit in bbox {bbox}: {e}")

    return all_digits

def main():
    if len(sys.argv) < 3:
        print("Usage: python singlePhotoPipeline.py <image_path> <output_mode>")
        print("output_mode: 'print' to print results, 'silent' for no output")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_mode = sys.argv[2]
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    # Paths (adjust as needed)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yolo_weights = os.path.join(base_dir, "outputs", "bbox_comparison", "yolo_run", "weights", "best.pt")
    digit_weights = os.path.join(base_dir, "outputs", "bbox_comparison", "digit_classifier.pth")
    
    if not os.path.exists(yolo_weights):
        print(f"YOLO weights not found: {yolo_weights}")
        sys.exit(1)
    
    if not os.path.exists(digit_weights):
        print(f"Digit classifier weights not found: {digit_weights}")
        sys.exit(1)
    
    # Load models
    yolo_model = load_yolo_model(yolo_weights)
    digit_model = load_digit_model(digit_weights)
    
    # Run detection
    bboxes, confs = run_yolo_on_image(yolo_model, image_path)
    
    if len(bboxes) == 0:
        result = "No digits detected"
    else:
        # Sort bboxes by x1
        bboxes = sorted(bboxes, key=lambda b: b[0])
        digits = recognize_digits(digit_model, image_path, bboxes)
        number_str = "".join(str(d[0]) for d in digits)
        result = f"Recognized number: {number_str}"
    
    if output_mode == "print":
        print(result)
    # For 'silent', do nothing

if __name__ == "__main__":
    main()