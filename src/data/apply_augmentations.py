import os
import json
import random
from tqdm import tqdm
from PIL import Image
try:
    from data import augmentations
except ImportError:
    import augmentations

def process_all(base_dir, prob=0.3):
    data_dir = os.path.join(base_dir, "data", "digits_data")
    if not os.path.exists(data_dir):
        print("Data directory not found. Skipping augmentations.")
        return

    # Find all sample folders
    all_samples = []
    for root, dirs, files in os.walk(data_dir):
        if "annotations.json" in files and "original.png" in files:
            all_samples.append(root)

    for sample_path in tqdm(all_samples, desc="Augmenting"):
        json_path = os.path.join(sample_path, "annotations.json")
        img_path = os.path.join(sample_path, "original.png")

        with open(json_path, 'r') as f:
            metadata = json.load(f)

        img = Image.open(img_path)
        
        # 1. Noise
        img = augmentations.apply_noise(img, prob=prob)
        # 2. Blur
        img = augmentations.apply_blur(img, prob=prob)
        
        # 3. Stretch (affects coordinates safely across all numbers/digits)
        all_boxes = []
        for num in metadata["detected_numbers"]:
            all_boxes.append(num["full_bounding_box"])
            for dig in num["digits"]:
                all_boxes.append(dig["bounding_box"])
        
        img, new_w, new_h = augmentations.apply_stretch(img, all_boxes, prob=prob)
        
        # Update image metadata
        metadata["image_metadata"]["width"] = new_w
        metadata["image_metadata"]["height"] = new_h
        
        # Save back
        img.save(img_path)
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", type=float, default=0.3)
    args = parser.parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    process_all(base_dir, prob=args.prob)
