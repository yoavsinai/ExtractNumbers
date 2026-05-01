from .base_enhancer import BaseEnhancer
from .none_enhancer import NoneEnhancer
from .unsharp_mask_enhancer import UnsharpMaskEnhancer
from .clahe_enhancer import CLAHEEnhancer
from .esrgan_enhancer import ESRGANEnhancer

def get_enhancer(method: str, **kwargs) -> BaseEnhancer:
    """
    Factory function to get an image enhancer instance.

    Args:
        method (str): The name of the enhancement method.
                      Supported: 'none', 'unsharp_mask', 'clahe', 'esrgan', 'opencv'.
        **kwargs: Additional arguments to pass to the enhancer's constructor.
                  For example: scale_factor=2.0, clip_limit=2.0.

    Returns:
        An instance of a BaseEnhancer subclass.
    """
    if method == "none":
        return NoneEnhancer()
    elif method == "unsharp_mask":
        return UnsharpMaskEnhancer(**kwargs)
    elif method == "clahe":
        return CLAHEEnhancer(**kwargs)
    elif method == "esrgan":
        return ESRGANEnhancer(**kwargs)
    # The old 'opencv' option is mapped to unsharp_mask for compatibility.
    elif method == "opencv":
        return UnsharpMaskEnhancer(**kwargs)
    else:
        raise ValueError(f"Unknown enhancement method: '{method}'")