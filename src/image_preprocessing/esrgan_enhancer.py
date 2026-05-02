import cv2
import numpy as np
import torch
from .base_enhancer import BaseEnhancer

try:
    from realesrgan import RealESRGAN
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

class ESRGANEnhancer(BaseEnhancer):
    """
    Enhancer using the Real-ESRGAN model for super-resolution,
    followed by a bilateral filter for denoising, mimicking the original pipeline.
    """
    def __init__(self, scale_factor=2, bilateral_diameter=9, sigma_color=75.0, sigma_space=75.0):
        if not REALESRGAN_AVAILABLE:
            raise ImportError("realesrgan is not installed. Please install it with 'pip install realesrgan'")
        
        self.scale = int(scale_factor)
        if self.scale not in [2, 4]:
            raise ValueError("ESRGAN Enhancer only supports scale factors of 2 or 4.")
            
        self.bilateral_diameter = bilateral_diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RealESRGAN(device, scale=self.scale)
        self.model.load_weights(f'weights/RealESRGAN_x{self.scale}plus.pth', download=True)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        # RealESRGAN expects a BGR image.
        img_bgr = image if image.ndim == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        upscaled = self.model.predict(img_bgr)
        denoised = cv2.bilateralFilter(upscaled, d=self.bilateral_diameter, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        return denoised