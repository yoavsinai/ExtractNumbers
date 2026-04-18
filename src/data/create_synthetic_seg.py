import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

DATA_PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/segmentation/synthetic'))

TARGET_SIZE = (256, 256)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
NUMBERS = "0123456789"
LUMINANCE_THRESHOLD = 128  # Backgrounds above this are considered light
DARK_COLOR_MAX = 80        # Upper bound for dark foreground channel values
LIGHT_COLOR_MIN = 175      # Lower bound for light foreground channel values

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def apply_augmentations(img_bgr, mask_np):
    """
    Apply stretch sequentially to both image and mask.
    Then apply blur, pixelation, and noise ONLY to the image.
    """
    h, w = img_bgr.shape[:2]

    # Stretching
    if random.random() < 0.6:
        stretch_w = random.uniform(0.6, 1.4)
        stretch_h = random.uniform(0.6, 1.4)
        new_w, new_h = max(1, int(w * stretch_w)), max(1, int(h * stretch_h))
        
        img_bgr = cv2.resize(img_bgr, (new_w, new_h))
        img_bgr = cv2.resize(img_bgr, (w, h))
        
        # Apply EXACT same transformations to mask using nearest neighbor to avoid blurring the mask
        mask_np = cv2.resize(mask_np, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        
    # Lower resolution (Pixelation) (Image only)
    if random.random() < 0.6:
        scale_factor = random.uniform(0.2, 0.3) # Increased min scale factor from 0.1
        small_w, small_h = max(1, int(w * scale_factor)), max(1, int(h * scale_factor))
        small = cv2.resize(img_bgr, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        img_bgr = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Blur (Image only)
    if random.random() < 0.4: # Reduced from 0.6
        kernel_size = random.choice([3, 5]) # Removed 7
        img_bgr = cv2.GaussianBlur(img_bgr, (kernel_size, kernel_size), 0)

    # White noise (Image only)
    if random.random() < 0.6: # Reduced from 0.8
        noise = np.random.normal(0, 25, img_bgr.shape).astype(np.float32) # Reduced noise level from 35
        img_bgr = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img_bgr, mask_np

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
        for _ in range(random.randint(15, 40)):
            x1, y1 = random.randint(0, TARGET_SIZE[0]), random.randint(0, TARGET_SIZE[1])
            x2, y2 = random.randint(0, TARGET_SIZE[0]), random.randint(0, TARGET_SIZE[1])
            d_img.line([(x1, y1), (x2, y2)], fill=(random.randint(0,255), random.randint(0,255), random.randint(0,255)), width=random.randint(1, 4))
            
        # 3. Add English letters (Noise)
        num_letters = random.randint(10, 30)
        # Potential font paths on Mac
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Black.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
            "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
            "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf"
        ]
        # Filter existing fonts
        available_fonts = [f for f in font_paths if os.path.exists(f)]
        
        def get_random_font(size):
            if available_fonts:
                return ImageFont.truetype(random.choice(available_fonts), size)
            return ImageFont.load_default(size=size)

        for _ in range(num_letters):
            char = random.choice(LETTERS)
            font_size = random.randint(25, 50)
            font = get_random_font(font_size)
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            x, y = random.randint(0, TARGET_SIZE[0]-font_size), random.randint(0, TARGET_SIZE[1]-font_size)
            d_img.text((x, y), char, fill=color, font=font)
            
        # 4. Add target Numbers and draw them on BOTH the image and the mask
        num_target_strings = random.randint(2, 5)
        placed_bboxes = []
        
        for _ in range(num_target_strings):
            number_length = random.randint(1, 4)
            number_str = "".join([random.choice(NUMBERS) for _ in range(number_length)])
            
            # Make sure numbers have high contrast against the background.
            # Use relative luminance to decide whether to use a dark or light foreground.
            bg_luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
            if bg_luminance > LUMINANCE_THRESHOLD:
                # Light background: pick a dark foreground color
                color = (random.randint(0, DARK_COLOR_MAX), random.randint(0, DARK_COLOR_MAX), random.randint(0, DARK_COLOR_MAX))
            else:
                # Dark background: pick a light foreground color
                color = (random.randint(LIGHT_COLOR_MIN, 255), random.randint(LIGHT_COLOR_MIN, 255), random.randint(LIGHT_COLOR_MIN, 255))
            
            font_size = random.randint(35, 65)
            font = get_random_font(font_size)
            
            # Try to place without overlap
            success = False
            for _attempt in range(10): # Try 10 times to find a non-overlapping spot
                max_x = max(20, TARGET_SIZE[0] - int(font_size * number_length * 0.7) - 20)
                max_y = max(20, TARGET_SIZE[1] - font_size - 10)
                x, y = random.randint(20, max_x), random.randint(20, max_y)
                
                bbox = d_img.textbbox((x, y), number_str, font=font)
                
                # Check for overlap
                overlap = False
                for existing_bbox in placed_bboxes:
                    # Check if boxes intersect
                    if not (bbox[2] < existing_bbox[0] or bbox[0] > existing_bbox[2] or 
                            bbox[3] < existing_bbox[1] or bbox[1] > existing_bbox[3]):
                        overlap = True
                        break
                
                if not overlap:
                    # Draw on main image
                    d_img.text((x, y), number_str, fill=color, font=font)
                    
                    # Draw EXACTLY on mask but separate each digit physically
                    current_x = x
                    for char in number_str:
                        try:
                            char_bbox = d_img.textbbox((current_x, y), char, font=font)
                        except AttributeError:
                            w_c, h_c = font.getsize(char)
                            char_bbox = [current_x, y, current_x + w_c, y + h_c]
                            
                        # Shrink by 2 pixels on left/right edges to guarantee they don't merge electrically
                        cx1 = min(char_bbox[2]-1, char_bbox[0] + 1)
                        cx2 = max(char_bbox[0]+1, char_bbox[2] - 1)
                        d_mask.rectangle([cx1, char_bbox[1], cx2, char_bbox[3]], fill=255)
                        
                        try:
                            current_x += int(font.getlength(char))
                        except AttributeError:
                            current_x += font.getsize(char)[0]
                            
                    placed_bboxes.append(bbox)
                    success = True
                    break
            
            if not success:
                # If we couldn't find a spot, just skip this string to keep data clean
                continue
            
        # 5. Save the generated pair into a dedicated subfolder
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        mask_np = np.array(mask)
        # Ensure it's strictly 1-bit style (0 or 255)
        mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)
        
        # Apply augmentations (Stretch both, noise/blur/pixelate image only)
        img_bgr, mask_np = apply_augmentations(img_bgr, mask_np)
        
        sample_dir = os.path.join(DATA_PROCESSED, str(i))
        ensure_dir(sample_dir)
        
        cv2.imwrite(os.path.join(sample_dir, "image.jpg"), img_bgr)
        cv2.imwrite(os.path.join(sample_dir, "mask.png"), mask_np)

if __name__ == "__main__":
    create_synthetic_seg_images(1000)
    print("Synthetic segmentation dataset generation complete. Files in `data/segmentation/synthetic`.")
