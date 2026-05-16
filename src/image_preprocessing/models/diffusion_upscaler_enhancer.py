import cv2
import numpy as np
import torch
from .base_enhancer import BaseEnhancer

try:
    from diffusers import StableDiffusionUpscalePipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

class DiffusionUpscalerEnhancer(BaseEnhancer):
    """
    Diffusion-based Image Upscaler using Stable Diffusion.
    Cutting-edge generative model for image upscaling with high-quality detail reconstruction.
    Note: Slower than other methods but may produce excellent quality for challenging images.
    """
    
    def __init__(self, scale_factor=4.0, num_inference_steps=20, guidance_scale=7.5):
        """
        Initialize Diffusion Upscaler Enhancer.
        
        Args:
            scale_factor (float): Upscaling factor. Default is 4.0 (diffusers pipeline fixed at 4x).
            num_inference_steps (int): Number of denoising steps. Lower = faster, higher = better quality. Default is 20.
            guidance_scale (float): Guidance scale for classifier-free guidance. Higher = more faithful. Default is 7.5.
        """
        # Note: Stable Diffusion Upscale pipeline is fixed at 4x
        self.scale_factor = 4  # Fixed by the model
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.pipe = None
        
        if not DIFFUSERS_AVAILABLE:
            print("⚠ Warning: diffusers is not installed.")
            print("  Will use bicubic resize as fallback")
            print("  To use Diffusion Upscaler: pip install diffusers")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        print("Loading Stable Diffusion Upscaler pipeline...")
        try:
            self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.pipe.to(device)
            print("✓ Diffusion pipeline loaded successfully!")
        except Exception as e:
            print(f"⚠ Warning: Could not load Diffusion Upscaler: {e}")
            print(f"  Will use bicubic resize as fallback")
            self.pipe = None
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the input image using Diffusion-based upscaling.
        
        Args:
            image (np.ndarray): Input image (BGR or grayscale).
        
        Returns:
            np.ndarray: Enhanced image.
        """
        if self.pipe is None:
            # Fallback to bicubic resize (4x upscaling)
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
        
        # Convert BGR to RGB for diffusers (expects PIL Image or RGB numpy)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert numpy to PIL Image
        from PIL import Image
        pil_image = Image.fromarray(img_rgb.astype(np.uint8))
        
        # Perform upscaling
        with torch.no_grad():
            result_pil = self.pipe(
                prompt="a high quality, detailed image",  # Generic prompt to guide upscaling
                image=pil_image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images[0]
        
        # Convert back to BGR numpy
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        
        # Convert back to original format if input was grayscale
        if is_grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        return result
