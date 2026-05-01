import cv2
import numpy as np
from .base_enhancer import BaseEnhancer

class CLAHEEnhancer(BaseEnhancer):
    """
    Traditional CV Enhancer using Bicubic Resize and CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    def __init__(self, scale_factor=2.0, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.scale_factor = scale_factor
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        # 1. Upscale using Bicubic Resize
        h, w = image.shape[:2]
        new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 2. Apply CLAHE (Handling both Grayscale and RGB images)
        if len(resized.shape) == 2:
            enhanced = self.clahe.apply(resized)
        else:
            lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        return enhanced