import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

DATA_PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/segmentation/synthetic'))

TARGET_SIZE = (256, 256)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUMBERS = "0123456789"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_synthetic_seg_images(num_samples=500):
    ensure_dir(DATA_PROCESSED)
    print(f"Generating {num_samples} synthetic segmentation images...")
    
    for i in range(num_samples):
        # 1. Create a random background color
        bg_color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        img = Image.new('RGB', TARGET_SIZE, color=bg_color)
        
        # We also create a pitch black mask image
        mask = Image.new('L', TARGET_SIZE, color=0)
        
        d_img = ImageDraw.Draw(img)
        d_mask = ImageDraw.Draw(mask)
        
        # 2. Add structural noise (lines, ellipses)
        for _ in range(random.randint(5, 15)):
            x1, y1 = random.randint(0, TARGET_SIZE[0]), random.randint(0, TARGET_SIZE[1])
            x2, y2 = random.randint(0, TARGET_SIZE[0]), random.randint(0, TARGET_SIZE[1])
            d_img.line([(x1, y1), (x2, y2)], fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), width=random.randint(1, 4))
            
        # 3. Add English letters (Noise)
        num_letters = random.randint(10, 30)
        for _ in range(num_letters):
            char = random.choice(LETTERS)
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            x, y = random.randint(0, TARGET_SIZE[0]-20), random.randint(0, TARGET_SIZE[1]-20)
            d_img.text((x, y), char, fill=color)
            
        # 4. Add target Numbers and draw them on BOTH the image and the mask
        num_target_strings = random.randint(2, 5)
        for _ in range(num_target_strings):
            number_length = random.randint(1, 4)
            number_str = "".join([random.choice(NUMBERS) for _ in range(number_length)])
            
            # Make sure numbers have high contrast or random color
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            x, y = random.randint(20, TARGET_SIZE[0]-60), random.randint(20, TARGET_SIZE[1]-40)
            
            # Draw on main image
            d_img.text((x, y), number_str, fill=color)
            
            # Find the bounding box of the drawn text
            bbox = d_img.textbbox((x, y), number_str)
            
            # Draw EXACTLY on mask in pure white (255) as a solid block/rectangle to match SVHN
            d_mask.rectangle(bbox, fill=255)
            
        # 5. Save the generated pair into a dedicated subfolder
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask_np = np.array(mask)
        # Ensure it's strictly 1-bit style (0 or 255)
        mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)
        
        sample_dir = os.path.join(DATA_PROCESSED, str(i))
        ensure_dir(sample_dir)
        
        cv2.imwrite(os.path.join(sample_dir, "image.jpg"), img_bgr)
        cv2.imwrite(os.path.join(sample_dir, "mask.png"), mask_np)

if __name__ == "__main__":
    create_synthetic_seg_images(500)
    print("Synthetic segmentation dataset generation complete. Files in `data/segmentation/synthetic`.")
