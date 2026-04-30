# Image Enhancement (Stage 2)

AI-powered image enhancement module using Real-ESRGAN.

## 4-Step Enhancement Pipeline
1. **AI Upscaling (Real-ESRGAN)**: 2x magnification with detail recovery and edge sharpening.
2. **Bilateral Filtering**: Edge-preserving denoising.
3. **Grayscale Conversion**: Consistent single-channel representation.
4. **Otsu Thresholding**: Optimal binary conversion for the Stage 4 classifier.

## Key Files
- `digit_preprocessor.py`: Core implementation of the enhancement functions.
