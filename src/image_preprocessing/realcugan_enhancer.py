import cv2
import numpy as np
import torch
from .base_enhancer import BaseEnhancer

try:
    from realesrgan import RealESRGAN
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False

class RealCUGANEnhancer(BaseEnhancer):
    """
    Real-CUGAN (Real Cascade U-Net GAN) - Lightweight GAN-based super-resolution.
    Designed for anime and general image super-resolution with balance between quality and speed.
    """
    
    def __init__(self, scale_factor=2.0, model_name="no-denoise", tile_size=480):
        """
        Initialize Real-CUGAN Enhancer.
        
        Args:
            scale_factor (float): Upscaling factor. Default is 2.0.
            model_name (str): Model variant - 'no-denoise', 'conservative', 'denoise'. Default is 'no-denoise'.
            tile_size (int): Tile size for processing to manage memory. Default is 480.
        """
        self.scale_factor = int(scale_factor)
        self.model_name = model_name
        self.tile_size = tile_size
        self.model = None
        
        if self.scale_factor not in [2, 3, 4]:
            raise ValueError(f"Real-CUGAN only supports scale factors of 2, 3, or 4. Got {scale_factor}")
        
        if not REALESRGAN_AVAILABLE:
            print("⚠ Warning: realesrgan is not installed.")
            print("  Will use bicubic resize as fallback")
            print("  To use Real-CUGAN: pip install realesrgan")
            self.model = None
            return
        
        # Try to load Real-CUGAN model
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = RealESRGAN(device, scale=self.scale_factor)
            
            # Try to load Real-CUGAN weights; fallback to RealESRGAN if not available
            try:
                self.model.load_weights(f'weights/RealCUGAN_x{self.scale_factor}.pth', download=True)
                print(f"✓ Real-CUGAN model loaded")
            except Exception:
                print(f"⚠ Real-CUGAN weights not available, using RealESRGAN as fallback")
                self.model.load_weights(f'weights/RealESRGAN_x{self.scale_factor}plus.pth', download=True)
        except Exception as e:
            print(f"⚠ Warning: Could not load Real-CUGAN model: {e}")
            print(f"  Will use bicubic resize as fallback")
            self.model = None
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the input image using Real-CUGAN.
        
        Args:
            image (np.ndarray): Input image (BGR or grayscale).
        
        Returns:
            np.ndarray: Enhanced image.
        """
        if not REALESRGAN_AVAILABLE or self.model is None:
            # Fallback to bicubic resize with light sharpening
            h, w = image.shape[:2]
            new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
            result = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            return result
        
        # Convert to BGR if grayscale
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            img_bgr = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = image
        
        # Real-CUGAN expects 8-bit BGR
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
        
        # Perform upscaling with tiling for large images
        h, w = img_bgr.shape[:2]
        if h > self.tile_size or w > self.tile_size:
            result = self._enhance_with_tiling(img_bgr)
        else:
            result = self.model.predict(img_bgr)
        
        # Convert back to original format if input was grayscale
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        return result
    
    def _enhance_with_tiling(self, image: np.ndarray, overlap=16) -> np.ndarray:
        """
        Process large images by tiling to manage memory.
        
        Args:
            image (np.ndarray): Input image.
            overlap (int): Overlap between tiles for seamless blending.
        
        Returns:
            np.ndarray: Enhanced image.
        """
        h, w = image.shape[:2]
        scaled_h = h * self.scale_factor
        scaled_w = w * self.scale_factor
        
        result = np.zeros((scaled_h, scaled_w, image.shape[2]), dtype=image.dtype)
        
        for y in range(0, h, self.tile_size - overlap):
            for x in range(0, w, self.tile_size - overlap):
                x_end = min(x + self.tile_size, w)
                y_end = min(y + self.tile_size, h)
                
                tile = image[y:y_end, x:x_end]
                enhanced_tile = self.model.predict(tile)
                
                # Calculate scaled positions
                scaled_y = y * self.scale_factor
                scaled_x = x * self.scale_factor
                scaled_y_end = y_end * self.scale_factor
                scaled_x_end = x_end * self.scale_factor
                
                result[scaled_y:scaled_y_end, scaled_x:scaled_x_end] = enhanced_tile
        
        return result
