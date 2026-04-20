import os
import sys
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import enhance_digit
from digit_recognizer.digit_recognizer import build_digit_model, get_device, preprocess_crop

def predict_image(image_path, model_dir):
    device = get_device()
    
    # Individual model components
    GLOBAL_MODEL_PATH = os.path.join(model_dir, "globalbb.pt")
    INDIV_MODEL_PATH = os.path.join(model_dir, "individualbb.pt")
    CLASSIFIER_PATH = os.path.join(model_dir, "digit_classifier.pth")
    
    # Load Models
    global_model = YOLO(GLOBAL_MODEL_PATH)
    indiv_model = YOLO(INDIV_MODEL_PATH)
    classifier = build_digit_model()
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
    classifier.to(device).eval()
    
    # Read Image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image."
    
    # 1. Global Detection
    res1 = global_model.predict(source=img, imgsz=256, verbose=False)
    if not res1 or len(res1[0].boxes) == 0:
        return "No numbers detected."
    
    # Use best global box
    gbox = res1[0].boxes[0]
    gx1, gy1, gx2, gy2 = map(int, gbox.xyxy[0].cpu().numpy())
    h, w = img.shape[:2]
    gx1, gy1, gx2, gy2 = max(0, gx1), max(0, gy1), min(w, gx2), min(h, gy2)
    crop = img[gy1:gy2, gx1:gx2]
    
    # 2. Enhancement
    sharp = enhance_digit(crop, upscale_factor=2.0)
    
    # 3. Individual Detection
    res2 = indiv_model.predict(source=sharp, imgsz=256, verbose=False)
    if not res2 or len(res2[0].boxes) == 0:
        return "Detected container but no individual digits found."
    
    # Sort boxes left to right
    iboxes = res2[0].boxes.xyxy.cpu().numpy()
    iboxes = sorted(iboxes, key=lambda b: b[0])
    
    # 4. Classification
    digits = []
    for ibox in iboxes:
        inputs = preprocess_crop(sharp, (ibox[0], ibox[1], ibox[2], ibox[3])).unsqueeze(0).to(device)
        with torch.no_grad():
            digit = classifier(inputs).argmax(dim=1).item()
            digits.append(str(digit))
            
    return "".join(digits)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict number in a single image.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--model-dir", default=os.path.join(BASE_DIR, "outputs", "trained_models"), help="Directory containing models")
    args = parser.parse_args()
    
    result = predict_image(args.image_path, args.model_dir)
    print(f"\nFinal Predicted Number: {result}")

if __name__ == "__main__":
    main()
