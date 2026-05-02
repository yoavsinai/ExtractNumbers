from .base_enhancer import BaseEnhancer
from .none_enhancer import NoneEnhancer
from .unsharp_mask_enhancer import UnsharpMaskEnhancer
from .clahe_enhancer import CLAHEEnhancer
from .esrgan_enhancer import ESRGANEnhancer
from .edsr_enhancer import EDSREnhancer
from .lapsrn_enhancer import LapSRNEnhancer
from .realcugan_enhancer import RealCUGANEnhancer
from .bsrgan_enhancer import BSRGANEnhancer
from .swiniR_enhancer import SwinIREnhancer
from .diffusion_upscaler_enhancer import DiffusionUpscalerEnhancer

def get_enhancer(method: str, **kwargs) -> BaseEnhancer:
    """
    Factory function to get an image enhancer instance.

    Args:
        method (str): The name of the enhancement method.
                      Supported: 'none', 'unsharp_mask', 'clahe', 'esrgan', 'opencv',
                                'edsr', 'lapsrn', 'realcugan', 'bsrgan', 'swiniR', 'diffusion'.
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
    elif method == "edsr":
        return EDSREnhancer(**kwargs)
    elif method == "lapsrn":
        return LapSRNEnhancer(**kwargs)
    elif method == "realcugan":
        return RealCUGANEnhancer(**kwargs)
    elif method == "bsrgan":
        return BSRGANEnhancer(**kwargs)
    elif method == "swiniR":
        return SwinIREnhancer(**kwargs)
    elif method == "diffusion":
        return DiffusionUpscalerEnhancer(**kwargs)
    # The old 'opencv' option is mapped to unsharp_mask for compatibility.
    elif method == "opencv":
        return UnsharpMaskEnhancer(**kwargs)
    else:
        raise ValueError(f"Unknown enhancement method: '{method}'. "
                        f"Supported methods: none, unsharp_mask, clahe, esrgan, edsr, lapsrn, "
                        f"realcugan, bsrgan, swiniR, diffusion, opencv")