import cv2
import numpy as np
import os
import torch
from .base_enhancer import BaseEnhancer

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.upsampler import RealESRGANer
    BASICSR_AVAILABLE = True
except ImportError:
    BASICSR_AVAILABLE = False

class BSRGANEnhancer(BaseEnhancer):
    """
    BSRGAN (Blind Super-Resolution GAN) - Blind SR model that handles unknown degradations.
    Robust to various types of image degradation without explicit degradation parameters.
    """
    
    def __init__(self, scale_factor=2.0, tile_size=400):
        """
        Initialize BSRGAN Enhancer.
        
        Args:
            scale_factor (float): Upscaling factor. Default is 2.0.
            tile_size (int): Tile size for processing to manage memory. Default is 400.
        """
        self.scale_factor = int(scale_factor)
        self.tile_size = tile_size
        self.upsampler = None
        
        if self.scale_factor not in [2, 3, 4]:
            raise ValueError(f"BSRGAN only supports scale factors of 2, 3, or 4. Got {scale_factor}")
        
        if not BASICSR_AVAILABLE:
            print("⚠ Warning: basicsr is not installed.")
            print("  Will use bicubic resize as fallback")
            print("  To use BSRGAN: pip install basicsr")
            self.upsampler = None
            return
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load BSRGAN model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=self.scale_factor)
            
            # Download and load weights
            cache_dir = os.path.expanduser("~/.cache/model_weights")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, f"BSRGAN_x{self.scale_factor}.pth")
            
            if not os.path.exists(model_path):
                print(f"Downloading BSRGAN model to {model_path}...")
                import urllib.request
                url = f"https://github.com/chaofengc/BSRGAN/releases/download/v1.0.0/BSRGAN_x{self.scale_factor}.pth"
                try:
                    urllib.request.urlretrieve(url, model_path)
                    print(f"✓ Model downloaded successfully!")
                except Exception as e:
                    print(f"⚠ Could not download BSRGAN weights: {e}")
                    print(f"  Will use bicubic resize as fallback")
                    self.upsampler = None
                    return
            
            self.upsampler = RealESRGANer(
                scale=self.scale_factor,
                model_path=model_path,
                upscale_model=model,
                tile=tile_size,
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),  # Use half precision on GPU if available
            )
            print(f"✓ BSRGAN model loaded")
        except Exception as e:
            print(f"⚠ Warning: Could not load BSRGAN: {e}")
            print(f"  Will use bicubic resize as fallback")
            self.upsampler = None
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the input image using BSRGAN.
        
        Args:
            image (np.ndarray): Input image (BGR or grayscale).
        
        Returns:
            np.ndarray: Enhanced image.
        """
        if self.upsampler is None:
            # Fallback: use simple bicubic resize
            h, w = image.shape[:2]
            new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
            result = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
            return result
        
        # Convert to BGR if grayscale
        if image.ndim == 2:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            is_grayscale = True
        elif image.shape[2] == 1:
            img_bgr = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
            is_grayscale = True
        else:
            img_bgr = image
            is_grayscale = False
        
        # BSRGAN expects uint8 BGR
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
        
        # Perform upscaling
        result, _ = self.upsampler.enhance(img_bgr, outscale=self.scale_factor)
        
        # Convert back to original format if input was grayscale
        if is_grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        return result
