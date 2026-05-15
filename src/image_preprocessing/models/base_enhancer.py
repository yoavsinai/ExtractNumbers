import numpy as np

class BaseEnhancer:
    """
    Base class for all image enhancement models.
    """
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance the input image. Must be implemented by subclasses.
        """
        raise NotImplementedError("Each enhancer must implement the 'enhance' method.")