import cv2
import numpy as np
from .base_enhancer import BaseEnhancer

class UnsharpMaskEnhancer(BaseEnhancer):
    """
    Traditional CV Enhancer using Bicubic Resize and Unsharp Masking.
    """
    def __init__(self, scale_factor=2.0, blur_ksize=(5, 5), weight=1.5):
        self.scale_factor = scale_factor
        self.blur_ksize = blur_ksize
        self.weight = weight

    def enhance(self, image: np.ndarray) -> np.ndarray:
        # 1. AI Upscaling alternative: Bicubic Resize
        h, w = image.shape[:2]
        new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 2. Unsharp Masking for detail recovery and edge sharpening
        blurred = cv2.GaussianBlur(resized, self.blur_ksize, 0)
        sharpened = cv2.addWeighted(resized, self.weight, blurred, -(self.weight - 1), 0)
        
        return sharpened