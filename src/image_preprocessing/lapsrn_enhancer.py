import cv2
import numpy as np
import os
import urllib.request
from .base_enhancer import BaseEnhancer

class LapSRNEnhancer(BaseEnhancer):
    """
    LapSRN (Laplacian Pyramid Super-Resolution Network) Enhancer using OpenCV DNN.
    Another lightweight super-resolution model suitable for real-time applications.
    Note: Falls back to bicubic resize if model download fails.
    """
    
    # Model URLs (multiple sources for reliability)
    MODEL_URLS = {
        2: [
            "https://github.com/opencv/opencv_contrib/raw/4.x/samples/data/sr_model/LapSRN_x2.pb",
        ],
        4: [
            "https://github.com/opencv/opencv_contrib/raw/4.x/samples/data/sr_model/LapSRN_x4.pb",
        ],
        8: [
            "https://github.com/opencv/opencv_contrib/raw/4.x/samples/data/sr_model/LapSRN_x8.pb",
        ],
    }
    
    def __init__(self, scale_factor=2.0, model_path=None):
        """
        Initialize LapSRN Enhancer.
        
        Args:
            scale_factor (float): Upscaling factor. Default is 2.0.
            model_path (str): Path to the LapSRN model file. If None, downloads to ~/.cache/model_weights/
        """
        self.scale_factor = int(scale_factor)
        self.model = None
        self.use_fallback = False
        
        if self.scale_factor not in [2, 4, 8]:
            raise ValueError(f"LapSRN only supports scale factors of 2, 4, or 8. Got {scale_factor}")
        
        # Determine model path
        if model_path is None:
            cache_dir = os.path.expanduser("~/.cache/model_weights")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, f"LapSRN_x{self.scale_factor}.pb")
        
        # Try to download and load model
        if not os.path.exists(model_path):
            self._download_model(model_path)
        
        # Try to load the model
        if os.path.exists(model_path):
            try:
                self.model = cv2.dnn_superres.DnnSuperResImpl_create()
                self.model.readModel(model_path)
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"✓ LapSRN model loaded successfully from {model_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not load LapSRN model: {e}")
                print(f"  Falling back to bicubic resize")
                self.use_fallback = True
                self.model = None
        else:
            print(f"⚠ Warning: LapSRN model not found")
            print(f"  Falling back to bicubic resize")
            self.use_fallback = True
    
    def _download_model(self, model_path):
        """Try to download model from multiple sources."""
        urls = self.MODEL_URLS.get(self.scale_factor, [])
        
        for url in urls:
            print(f"Attempting to download LapSRN_x{self.scale_factor} from {url}...")
            try:
                urllib.request.urlretrieve(url, model_path)
                print(f"✓ Model downloaded successfully!")
                return
            except Exception as e:
                print(f"  Failed: {e}")
        
        print(f"⚠ Could not download LapSRN_x{self.scale_factor} model from any source")
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the input image using LapSRN.
        
        Args:
            image (np.ndarray): Input image (BGR or grayscale).
        
        Returns:
            np.ndarray: Enhanced image.
        """
        if self.use_fallback or self.model is None:
            # Fallback to bicubic resize
            h, w = image.shape[:2]
            new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
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
        
        # LapSRN expects 8-bit BGR
        if img_bgr.dtype != np.uint8:
            img_bgr = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
        
        # Perform upscaling
        result = self.model.upsample(img_bgr)
        
        # Convert back to original format if input was grayscale
        if is_grayscale:
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        return result
