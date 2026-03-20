import os
import glob
import cv2
import random
import numpy as np
import torchvision
from PIL import Image, ImageDraw

DATA_RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
DATA_PROCESSED = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/classification'))
KAGGLE_CACHE = os.path.expanduser('~/.cache/kagglehub/datasets')

TARGET_SIZE = (64, 64)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def add_letter_noise(pil_img):
    if random.random() > 0.5:
        return pil_img
        
    d = ImageDraw.Draw(pil_img)
    num_letters = random.randint(1, 4)
    w, h = pil_img.size
    for _ in range(num_letters):
        char = random.choice(LETTERS)
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        x = random.choice([random.randint(0, 15), random.randint(w-20, w-5)])
        y = random.choice([random.randint(0, 15), random.randint(h-20, h-5)])
        d.text((x, y), char, fill=color)
    return pil_img

def process_mnist():
    print("Processing MNIST...")
    trainset = torchvision.datasets.MNIST(root=DATA_RAW, train=True, download=False)
    for i, (img, label) in enumerate(trainset):
        if i >= 5000: break
        out_dir = os.path.join(DATA_PROCESSED, "single_digits", str(label))
        ensure_dir(out_dir)
        img_rgb = img.convert("RGB").resize(TARGET_SIZE)
        img_rgb = add_letter_noise(img_rgb)
        img_rgb.save(os.path.join(out_dir, f"mnist_{i}_{label}.png"))

def process_svhn():
    print("Processing SVHN...")
    trainset = torchvision.datasets.SVHN(root=DATA_RAW, split='train', download=False)
    for i, (img, label) in enumerate(trainset):
         if i >= 5000: break
         l = 0 if label == 10 else label
         out_dir = os.path.join(DATA_PROCESSED, "single_digits", str(l))
         ensure_dir(out_dir)
         img_rgb = img.convert("RGB").resize(TARGET_SIZE)
         img_rgb = add_letter_noise(img_rgb)
         img_rgb.save(os.path.join(out_dir, f"house_{i}_{l}.png"))

def process_handwritten_kaggle():
    print("Processing Kaggle Handwritten Digits...")
    versions_root = os.path.join(KAGGLE_CACHE, 'olafkrastovski', 'handwritten-digits-0-9', 'versions')
    if not os.path.isdir(versions_root):
        return
    # Select the highest numeric version directory available
    version_dirs = [
        d for d in os.listdir(versions_root)
        if os.path.isdir(os.path.join(versions_root, d))
    ]
    numeric_versions = []
    for d in version_dirs:
        try:
            numeric_versions.append((int(d), d))
        except ValueError:
            continue
    if not numeric_versions:
        return
    latest_version_dir = os.path.join(versions_root, max(numeric_versions)[1])
    base_dir = latest_version_dir
    for label in range(10):
        cls_dir = os.path.join(base_dir, str(label))
        out_dir = os.path.join(DATA_PROCESSED, "single_digits", str(label))
        ensure_dir(out_dir)
        if not os.path.exists(cls_dir): continue
        files = glob.glob(os.path.join(cls_dir, "*.png")) + glob.glob(os.path.join(cls_dir, "*.jpg"))
        for i, f in enumerate(files[:500]):
            img = cv2.imread(f)
            if img is not None:
                img = cv2.resize(img, TARGET_SIZE)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                pil_img = add_letter_noise(pil_img)
                cv2.imwrite(os.path.join(out_dir, f"handwritten_{i}_{label}.png"), cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

def generate_multi_digits():
    print("Generating Multi-digit Numbers...")
    out_dir = os.path.join(DATA_PROCESSED, "multi_digits")
    ensure_dir(out_dir)
    
    digit_files = [] 
    for d in range(10):
        d_dir = os.path.join(DATA_PROCESSED, "single_digits", str(d))
        files = glob.glob(os.path.join(d_dir, "*.png"))
        for f in files:
            digit_files.append((f, str(d)))
            
    if len(digit_files) < 10: return
        
    for i in range(2000):
        num_digits = random.randint(2, 4)
        chosen = random.sample(digit_files, num_digits)
        imgs = [cv2.imread(f[0]) for f in chosen if cv2.imread(f[0]) is not None]
        if len(imgs) != num_digits: continue
        
        actual_number_str = "".join([f[1] for f in chosen])
        concat = np.hstack(imgs)
        
        pil_img = Image.fromarray(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB))
        pil_img = add_letter_noise(pil_img)
        cv2.imwrite(os.path.join(out_dir, f"multidigit_{i}_{actual_number_str}.png"), cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    ensure_dir(DATA_PROCESSED)
    process_mnist()
    process_svhn()
    process_handwritten_kaggle()
    generate_multi_digits()
    print("Classification dataset generation complete. Files are in `data/classification`.")
