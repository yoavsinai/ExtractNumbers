import os
import random
import shutil
import cv2
import numpy as np

# =====================================================
# Uniform number generator (0 → 10^10 - 1)
# =====================================================
def generate_random_number():
    return str(random.randint(0, 10**10 - 1))

# =====================================================
# Background with noise
# =====================================================
def create_background(height, width):
    # Standard deviation returned to 7 as requested
    base_intensity = random.randint(180, 255)
    img = np.ones((height, width, 3), dtype=np.uint8) * base_intensity

    # Gaussian noise
    noise = np.random.normal(0, 7, img.shape).astype(np.float32)
    img = img.astype(np.float32) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

# =====================================================
# Draw number with FIXED spacing and RANDOM color
# =====================================================
def draw_number_with_fixed_spacing(
    img,
    number,
    font,
    font_scale,
    thickness,
    spacing,
    color,
    start_x,
    start_y
):
    x = start_x
    y = start_y

    for digit in number:
        cv2.putText(
            img,
            digit,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        (w, h), _ = cv2.getTextSize(
            digit, font, font_scale, thickness
        )
        x += w + spacing

    return img

# =====================================================
# Create base image with LARGE random padding
# =====================================================
def create_base_image(number):
    font = random.choice([
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ])

    font_scale = random.uniform(1.8, 3.0)
    thickness = random.randint(2, 4)
    
    spacing = random.randint(5, 30)
    
    # Random dark color for the text
    color = (
        random.randint(0, 130),
        random.randint(0, 130),
        random.randint(0, 130)
    )

    # Calculate actual text dimensions
    total_text_width = 0
    max_digit_height = 0
    for digit in number:
        (w, h), baseline = cv2.getTextSize(digit, font, font_scale, thickness)
        total_text_width += w
        max_digit_height = max(max_digit_height, h)
    
    total_text_width += spacing * (len(number) - 1)

    # CREATE "WASTE" SPACE:
    # We create a canvas much larger than the text
    width = total_text_width + random.randint(150, 400)
    height = max_digit_height + random.randint(150, 300)

    img = create_background(height, width)

    # RANDOM POSITIONING:
    # Calculate available slack and pick a random starting point
    slack_x = width - total_text_width
    slack_y = height - max_digit_height

    start_x = random.randint(20, slack_x - 20)
    start_y = random.randint(max_digit_height + 20, height - 20)

    img = draw_number_with_fixed_spacing(
        img,
        number,
        font,
        font_scale,
        thickness,
        spacing,
        color,
        start_x,
        start_y
    )

    return img

# =====================================================
# Augmentations (Blur, Rotation, Perspective)
# =====================================================
def apply_motion_blur(img):
    if random.random() < 0.7:
        size = random.randint(7, 17)
        angle = random.uniform(0, 180)
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = np.ones(size)
        kernel /= size
        center = (size / 2 - 0.5, size / 2 - 0.5)
        rot = cv2.getRotationMatrix2D(center, angle, 1)
        kernel = cv2.warpAffine(kernel, rot, (size, size))
        img = cv2.filter2D(img, -1, kernel)
    return img

def apply_rotation(img):
    if random.random() < 0.8:
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]
        img = cv2.warpAffine(img, M, (new_w, new_h), borderValue=(255, 255, 255))
    return img

def apply_perspective(img):
    if random.random() < 0.6:
        h, w = img.shape[:2]
        margin = 40
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([
            [random.randint(0, margin), random.randint(0, margin)],
            [w - 1 - random.randint(0, margin), random.randint(0, margin)],
            [w - 1 - random.randint(0, margin), h - 1 - random.randint(0, margin)],
            [random.randint(0, margin), h - 1 - random.randint(0, margin)]
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))
    return img

# =====================================================
# Pipeline & Dataset Generator
# =====================================================
def create_image_for_number(number):
    img = create_base_image(number)
    img = apply_motion_blur(img)
    img = apply_rotation(img)
    img = apply_perspective(img)
    return img

def generate_dataset(output_dir="data/synthetic_v4", num_images=200):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        number = generate_random_number()
        img = create_image_for_number(number)
        filename = f"{number}_{i}.png"
        cv2.imwrite(os.path.join(output_dir, filename), img)
    return output_dir

if __name__ == "__main__":
    path = generate_dataset()
    print(f"Dataset generated at: {path}")