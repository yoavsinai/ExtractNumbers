import numpy as np
from .base_enhancer import BaseEnhancer

class NoneEnhancer(BaseEnhancer):
    """
    An enhancer that does nothing, returning the original image.
    Used as a baseline for comparison.
    """
    def enhance(self, image: np.ndarray) -> np.ndarray:
        return image.copy()