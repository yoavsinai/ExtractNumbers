import cv2
import numpy as np
import torch
import os
import urllib.request
from .base_enhancer import BaseEnhancer

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

class ESRGANEnhancer(BaseEnhancer):
    """
    Enhancer using the official Xintao Real-ESRGAN model for super-resolution,
    followed by a bilateral filter for denoising, mimicking the original pipeline.
    """
    def __init__(self, scale_factor=2, bilateral_diameter=9, sigma_color=75.0, sigma_space=75.0):
        if not REALESRGAN_AVAILABLE:
            raise ImportError("realesrgan or basicsr is not installed. Please check your requirements.txt")
        
        self.scale = int(scale_factor)
        if self.scale not in [2, 4]:
            raise ValueError("ESRGAN Enhancer only supports scale factors of 2 or 4.")
            
        self.bilateral_diameter = bilateral_diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine weight path and download if missing
        weight_filename = f'RealESRGAN_x{self.scale}plus.pth'
        weight_path = os.path.join('weights', weight_filename)
        
        if not os.path.exists(weight_path):
            os.makedirs('weights', exist_ok=True)
            # x2plus is in v0.2.1, x4plus is in v0.1.0
            version = "v0.2.1" if self.scale == 2 else "v0.1.0"
            url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/{version}/{weight_filename}"
            print(f"Downloading RealESRGAN weights to {weight_path}...")
            urllib.request.urlretrieve(url, weight_path)
            
        # Initialize the model structure
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale)
        
        # Initialize the official RealESRGAN upsampler
        self.upsampler = RealESRGANer(
            scale=self.scale,
            model_path=weight_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
            device=device
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        # RealESRGANer expects a BGR image.
        img_bgr = image if image.ndim == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Enhance using the official API
        upscaled, _ = self.upsampler.enhance(img_bgr, outscale=self.scale)
        
        # Apply bilateral filter as in the original pipeline
        denoised = cv2.bilateralFilter(upscaled, d=self.bilateral_diameter, sigmaColor=self.sigma_color, sigmaSpace=self.sigma_space)
        return denoised