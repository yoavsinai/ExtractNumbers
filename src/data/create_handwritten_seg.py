import os
import random
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, ImageFont

# --- Path Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_CLASSIFICATION = os.path.join(BASE_DIR, 'data/classification/single_digits')
DATA_PROCESSED = os.path.join(BASE_DIR, 'data/segmentation/handwritten')

# --- Constants & Settings ---
TARGET_SIZE = (256, 256)
DIGIT_SPACING_MARGIN = 20  # Minimum space between digits/letters
MIN_VISIBLE_RATIO = 0.015  # Minimum ink coverage
MAX_VISIBLE_RATIO = 0.40   # Maximum ink coverage to reject "blobs"
LETTERS_COUNT_RANGE = (1, 2) # Randomly place 1 to 2 large letters
DIGITS_COUNT_RANGE = (2, 4)  # Randomly place 2 to 4 digits

# --- Color Palette (R, G, B) ---
BASIC_COLORS = {
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'red': (255, 0, 0),
    'green': (0, 180, 0), # Slightly darker for contrast
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255)
}

# --- Font Loading with Windows System Support ---
def get_large_font(size):
    """
    Attempts to load a standard system font at the desired size.
    Adjusts paths for robustness on Windows/Linux environments.
    """
    possible_paths = [
        "arial.ttf", # Often works by default name
        "c:/windows/fonts/arial.ttf", 
        "c:/windows/fonts/tahoma.ttf", 
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" # Common Linux fallback
    ]
    
    for path in possible_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
            
    # If all fail, fallback to default tiny font (WARNING: size control won't work)
    print("WARNING: Could not load dynamic system font. Falling back to default (very small).")
    return ImageFont.load_default()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def preprocess_digit_clean(img):
    """
    Cleans the source digit, ensures it's white on black, and checks quality.
    """
    # 1. Standardize to Grayscale and Contrast
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    
    np_gray = np.array(gray)
    edge_avg = (np.mean(np_gray[0, :]) + np.mean(np_gray[-1, :]) + 
                np.mean(np_gray[:, 0]) + np.mean(np_gray[:, -1])) / 4
    
    # Invert if it's black-on-white source (e.g., Kaggle)
    if edge_avg > 120:
        gray = ImageOps.invert(gray)
        
    # 2. Clean Background
    # Keep some soft edges for realism
    clean_mask = gray.point(lambda x: 0 if x < 120 else x)
    
    # 3. Quality Control: Reject Blobs and Empty images
    np_mask = np.array(clean_mask)
    ink_pixels = np.sum(np_mask > 0)
    ratio = ink_pixels / np_mask.size
    
    if ratio < MIN_VISIBLE_RATIO or ratio > MAX_VISIBLE_RATIO:
        return None # Too thin or too much like a blob
        
    return clean_mask

def load_filtered_digit_files():
    digit_files = {i: [] for i in range(10)}
    print("Loading and filtering clean handwriting images...")
    for label in range(10):
        cls_dir = os.path.join(DATA_CLASSIFICATION, str(label))
        all_files = glob.glob(os.path.join(cls_dir, "*.png")) + glob.glob(os.path.join(cls_dir, "*.jpg"))
        for f in all_files:
            # Quick check for black background
            temp_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if temp_img is not None and np.mean([temp_img[0,0], temp_img[-1,-1]]) < 45:
                # Add MNIST/Handwritten files, excluding house_ SVHN
                digit_files[label].append(f)
    return digit_files

def overlaps(new_box, existing_boxes, margin):
    nx1, ny1, nx2, ny2 = new_box
    for (ex1, ey1, ex2, ey2) in existing_boxes:
        if not (nx2 + margin < ex1 or nx1 - margin > ex2 or 
                ny2 + margin < ey1 or ny1 - margin > ey2):
            return True
    return False

def get_contrasting_colors():
    """Selects random, contrasting background and foreground colors from palette."""
    keys = list(BASIC_COLORS.keys())
    
    # Quick simple contrast: avoid black on white or yellow on green.
    # We use basic luminance check to enforce contrast.
    while True:
        bg_name = random.choice(keys)
        fg_name = random.choice(keys)
        
        if bg_name == fg_name: continue
        
        bg_rgb = BASIC_COLORS[bg_name]
        fg_rgb = BASIC_COLORS[fg_name]
        
        # Avoid similar brightness (e.g., black on blue)
        # Luminance calculation (0.299R + 0.587G + 0.114B)
        l_bg = (bg_rgb[0] * 0.299 + bg_rgb[1] * 0.587 + bg_rgb[2] * 0.114) / 255
        l_fg = (fg_rgb[0] * 0.299 + fg_rgb[1] * 0.587 + fg_rgb[2] * 0.114) / 255
        
        if abs(l_bg - l_fg) > 0.5: # Sufficient brightness difference
            return bg_rgb, fg_rgb

def create_handwritten_seg(num_samples=500):
    ensure_dir(DATA_PROCESSED)
    digit_files_dict = load_filtered_digit_files()

    print(f"Generating {num_samples} production-ready samples...")
    
    # Load one scalable dynamic font once for all samples to match digit size
    large_font = get_large_font(random.randint(65, 80)) # Size roughly 65-80

    for i in range(num_samples):
        # 1. Get Contrasting Background & Foreground Colors
        bg_rgb, fg_rgb = get_contrasting_colors()
        
        img = Image.new('RGB', TARGET_SIZE, color=bg_rgb)
        mask = Image.new('L', TARGET_SIZE, color=0)
        
        d_img = ImageDraw.Draw(img)
        d_mask = ImageDraw.Draw(mask)

        # 2. Trace Subtle Noise Lines (Drawn BEFORE targets)
        for _ in range(8):
            coords = [(random.randint(0, 256), random.randint(0, 256)) for _ in range(2)]
            # Draw noise line using slightly different color from fg
            noise_color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
            d_img.line(coords, fill=noise_color, width=1)

        # 3. Add large random letters (Distractors)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        
        for _ in range(random.randint(*LETTERS_COUNT_RANGE)):
            char = random.choice(alphabet)
            
            # Determine text bounding box to place correctly
            try:
                # PIL version supporting textbbox
                bbox = d_img.textbbox((0, 0), char, font=large_font)
                w_let, h_let = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                # Fallback for old PIL versions
                w_let, h_let = large_font.getsize(char)
                
            found_pos = False
            for attempt in range(40): # Attempts to find non-overlap spot
                x = random.randint(20, TARGET_SIZE[0] - w_let - 20)
                y = random.randint(20, TARGET_SIZE[1] - h_let - 20)
                
                # Check overlaps against line noise (too detailed) or other target boxes (not existing yet)
                # The overlap check logic usually works on *placed* objects, like digits.
                # Letters are structural noise, so we just want them not to cover other targets or letters.
                new_box = (x, y, x + w_let, y + h_let)
                
                # Letters don't targets, so they don't add bounding boxes to mask
                # But they should not overlap *digits* we place later. 
                # Better approach: place letters after digits and check against them.
                # BUT request asked for letters "like the numbers". I will place them with overlap prevention.
                
                # ... overlap check conceptual change ... 
                # (Conceptual change needed here, usually letters are distractors). 
                # I will treat letters as structural noise, drawn after digits. Or drawn before with overlap check.
                # Since request wants them *like* numbers, I will place them alongside digits in the overlap check.

                # Let's simplify: letters are drawn directly on image, no mask. I will draw them without overlap check.
                # If overlap check is needed, I'd move this loop after digit placement. 
                # Let's do that for quality. Move Letter placement AFTER digits.
                pass

        # 4. Paste Handwritten Digits & Update Masks
        placed_boxes = []
        num_to_place = random.randint(*DIGITS_COUNT_RANGE)
        
        for _ in range(num_to_place):
            label = random.randint(0, 9)
            if not digit_files_dict[label]: continue
            
            digit_path = random.choice(digit_files_dict[label])
            raw_digit = Image.open(digit_path).convert('RGB')
            
            clean_stencil = preprocess_digit_clean(raw_digit)
            if clean_stencil is None: continue
            
            # Random scaling
            size = random.randint(55, 80)
            clean_stencil = clean_stencil.resize((size, size), Image.Resampling.LANCZOS)
            
            # Non-overlap logic
            found = False
            for _ in range(50):
                x, y = random.randint(15, TARGET_SIZE[0]-size-15), random.randint(15, TARGET_SIZE[1]-size-15)
                new_box = (x, y, x + size, y + size)
                
                if not overlaps(new_box, placed_boxes, DIGIT_SPACING_MARGIN):
                    placed_boxes.append(new_box)
                    # Paste digit: Use the stencil itself for natural transparency and the target FG color
                    digit_color_layer = Image.new('RGB', (size, size), color=fg_rgb)
                    img.paste(digit_color_layer, (x, y), clean_stencil)
                    
                    # Target Mask (rectangle bounding box fill white)
                    d_mask.rectangle([x, y, x + size, y + size], fill=255)
                    found = True
                    break

        # 5. NOW add large random letters (Drawn AFTER digits) - Use FG color
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        
        for _ in range(random.randint(*LETTERS_COUNT_RANGE)):
            char = random.choice(alphabet)
            
            # Determine text bounding box
            try:
                # PIL version supporting textbbox
                bbox = d_img.textbbox((0, 0), char, font=large_font)
                w_let, h_let = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                # Fallback
                w_let, h_let = large_font.getsize(char)
                
            for attempt in range(40): # Attempts to find non-overlap spot against digits
                x = random.randint(20, TARGET_SIZE[0] - w_let - 20)
                y = random.randint(20, TARGET_SIZE[1] - h_let - 20)
                new_box = (x, y, x + w_let, y + h_let)
                
                # Check against digits (placed_boxes contains digits)
                if not overlaps(new_box, placed_boxes, DIGIT_SPACING_MARGIN):
                    placed_boxes.append(new_box) # Treat letter as placed too
                    # Draw letter using the chosen FG color
                    d_img.text((x, y), char, font=large_font, fill=fg_rgb)
                    # Letters don't add bounding boxes to the Target mask (mask only shows digits).
                    break
        
        # Save results
        sample_dir = os.path.join(DATA_PROCESSED, str(i))
        ensure_dir(sample_dir)
        img.save(os.path.join(sample_dir, "image.jpg"))
        mask.save(os.path.join(sample_dir, "mask.png"))
        
        if (i+1) % 100 == 0: print(f"Progress: {i+1}/500 samples generated.")

if __name__ == "__main__":
    create_handwritten_seg(500)
    print("\nSUCCESS: Production-ready dataset is ready in data/segmentation/handwritten")