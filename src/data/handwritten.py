import os
import json
import random
import numpy as np
import kagglehub
from torchvision import datasets
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageOps
import glob

def prepare(output_base_dir, limit=None):
    print("--- Preparing Synthetic Handwritten Dataset (Realistic) ---")
    
    # 1. Download MNIST
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    mnist_train = datasets.MNIST(root=raw_dir, train=True, download=True)
    
    digit_pool = {i: [] for i in range(10)}
    for img, label in mnist_train:
        digit_pool[label].append(img)
    
    bg_path = kagglehub.dataset_download("prasunroy/natural-images")
    bg_files = glob.glob(os.path.join(bg_path, "**", "*.jpg"), recursive=True)
    
    dataset_dir = os.path.join(output_base_dir, "handwritten")
    os.makedirs(dataset_dir, exist_ok=True)
    num_samples = limit if limit else 1000
    
    for i in tqdm(range(num_samples), desc="Handwritten"):
        n_digits = random.randint(1, 6)
        scale = random.uniform(2.0, 4.0) # Scale up the 28x28 digits
        
        selected_digits = []
        full_value = ""
        for _ in range(n_digits):
            val = random.randint(0, 9)
            img = random.choice(digit_pool[val])
            # Resize
            new_size = (int(img.width * scale), int(img.height * scale))
            img_resized = img.resize(new_size, Image.Resampling.BILINEAR)
            selected_digits.append((val, img_resized))
            full_value += str(val)
            
        spacing = random.randint(int(-4 * scale), int(6 * scale))
        total_number_width = sum(d[1].width for d in selected_digits) + (n_digits - 1) * spacing
        max_digit_height = max(d[1].height for d in selected_digits)

        # 3. Create/Select background
        bg_w, bg_h = random.randint(600, 1000), random.randint(600, 1000)
        if bg_files:
            bg_img_path = random.choice(bg_files)
            try:
                bg_full = Image.open(bg_img_path).convert("RGB")
                w_orig, h_orig = bg_full.size
                if w_orig > bg_w and h_orig > bg_h:
                    x_crop = random.randint(0, w_orig - bg_w)
                    y_crop = random.randint(0, h_orig - bg_h)
                    canvas = bg_full.crop((x_crop, y_crop, x_crop + bg_w, y_crop + bg_h))
                else:
                    canvas = bg_full.resize((bg_w, bg_h))
            except:
                canvas = Image.new('RGB', (bg_w, bg_h), color=(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))
        else:
            canvas = Image.new('RGB', (bg_w, bg_h), color=(random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)))

        # 4. Plant the number
        start_x = random.randint(20, max(21, bg_w - total_number_width - 20))
        start_y = random.randint(20, max(21, bg_h - max_digit_height - 20))
        
        num_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        curr_x = start_x
        digits_metadata = []
        x_coords, y_coords, max_x, max_y = [], [], [], []
        
        for val, img_res in selected_digits:
            v_jitter = random.randint(int(-5 * scale), int(5 * scale))
            y_pos = start_y + v_jitter
            
            digit_colorized = ImageOps.colorize(img_res.convert("L"), black="black", white=num_color)
            canvas.paste(digit_colorized, (curr_x, y_pos), mask=img_res.convert("L"))
            
            wd, ht = img_res.width, img_res.height
            digits_metadata.append({
                "label": val,
                "bounding_box": {
                    "x": float(curr_x),
                    "y": float(y_pos),
                    "width": float(wd),
                    "height": float(ht)
                }
            })
            
            x_coords.append(float(curr_x))
            y_coords.append(float(y_pos))
            max_x.append(float(curr_x + wd))
            max_y.append(float(y_pos + ht))
            
            curr_x += wd + spacing
            
        full_box = {
            "x": min(x_coords),
            "y": min(y_coords),
            "width": max(max_x) - min(x_coords),
            "height": max(max_y) - min(y_coords)
        }
        
        detected_numbers = [
                {
                    "full_value": full_value,
                    "full_bounding_box": full_box,
                    "digits": digits_metadata
                }
            ]

        # 5. Add distractors
        draw = ImageDraw.Draw(canvas)
        for _ in range(random.randint(3, 8)):
            x1, y1 = random.randint(0, bg_w), random.randint(0, bg_h)
            x2, y2 = random.randint(0, bg_w), random.randint(0, bg_h)
            draw.line((x1, y1, x2, y2), fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), width=random.randint(1, 2))

        # 6. Save
        sample_folder = os.path.join(dataset_dir, f"sample_{i}")
        os.makedirs(sample_folder, exist_ok=True)
        canvas.save(os.path.join(sample_folder, "original.png"))
        
        metadata = {
            "image_metadata": {
                "sample_index": i,
                "filename": "original.png",
                "width": bg_w,
                "height": bg_h
            },
            "detected_numbers": [
                {
                    "full_value": full_value,
                    "full_bounding_box": full_box,
                    "digits": digits_metadata
                }
            ]
        }
        with open(os.path.join(sample_folder, "annotations.json"), "w") as f:
            json.dump(metadata, f, indent=4)
