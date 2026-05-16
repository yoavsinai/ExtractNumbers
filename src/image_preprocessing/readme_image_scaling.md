# Image Enhancement (Stage 2)

AI-powered image enhancement module using Real-ESRGAN.

## 4-Step Enhancement Pipeline
1. **AI Upscaling (Real-ESRGAN)**: 2x magnification with detail recovery and edge sharpening.
2. **Bilateral Filtering**: Edge-preserving denoising.
3. **Grayscale Conversion**: Consistent single-channel representation.
4. **Otsu Thresholding**: Optimal binary conversion for the Stage 4 classifier.

## Key Files
- `enhancer_factory.py`: Central factory for instantiating different enhancement models.
- `digit_preprocessor.py`: Core implementation of the enhancement functions and legacy pipeline.
- `models/`: Directory containing individual enhancer implementations.

## 🛠 Enhancer Factory
The module uses a factory pattern to easily switch between different enhancement methods.

### Usage Example
```python
from image_preprocessing import get_enhancer

# Initialize an enhancer (e.g., Real-ESRGAN)
enhancer = get_enhancer('esrgan', scale_factor=2)

# Enhance an image
enhanced_img = enhancer.enhance(image)
```

### Supported Methods
| Method | Description |
| :--- | :--- |
| `none` | Returns the image as is. |
| `unsharp_mask` | Traditional sharpening using unsharp masking. |
| `clahe` | Contrast Limited Adaptive Histogram Equalization. |
| `esrgan` | AI Super-Resolution (Real-ESRGAN). |
| `edsr` | Enhanced Deep Residual Networks. |
| `lapsrn` | Laplacian Pyramid Super-Resolution Network. |
| `realcugan` | Real-CUGAN (lightweight GAN). |
| `bsrgan` | Blind Super-Resolution GAN. |
| `swiniR` | SwinIR (Transformer-based restoration). |
| `diffusion` | Stable Diffusion 4x upscaler. |
| `opencv` | Alias for `unsharp_mask`. |
