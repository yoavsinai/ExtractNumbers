import random
import numpy as np
from PIL import Image, ImageFilter

def apply_blur(image, prob=0.3):
    if random.random() < prob:
        radius = random.uniform(0.5, 2.0)
        return image.filter(ImageFilter.GaussianBlur(radius))
    return image

def apply_noise(image, prob=0.3):
    if random.random() < prob:
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, random.uniform(5, 20), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    return image

def apply_stretch(image, bboxes, prob=0.3):
    """
    Applies a random stretch (resize) to the image and updates bboxes.
    bboxes should be a list of dicts with 'x', 'y', 'width', 'height'.
    """
    if random.random() < prob:
        w, h = image.size
        # Random scale factor for each dimension
        scale_x = random.uniform(0.8, 1.2)
        scale_y = random.uniform(0.8, 1.2)
        
        new_w, new_h = int(w * scale_x), int(h * scale_y)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Update bboxes
        for box in bboxes:
            box['x'] *= scale_x
            box['y'] *= scale_y
            box['width'] *= scale_x
            box['height'] *= scale_y
            
        return image, new_w, new_h
    return image, image.width, image.height
