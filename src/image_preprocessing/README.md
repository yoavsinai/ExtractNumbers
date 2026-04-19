# ImagePreprocessing Module

This module provides efficient, reusable functions for digit image enhancement and preprocessing using **Real-ESRGAN** for AI-powered image enhancement. The preprocessing pipeline optimizes cropped digit images for classification through a 4-step enhancement process.

## Pipeline Overview

The preprocessing pipeline transforms raw digit crops into optimized images suitable for classification and detection:

### Input → 4-Step Pipeline → Output
**Raw digit crop** → AI Upscaling → Denoising → Grayscale → Binary → **Optimized digit**

## Enhancement Pipeline Steps

### Step 1: AI Upscaling with Real-ESRGAN (2x/4x)
- **Method**: Real-ESRGAN super-resolution AI model
- **Purpose**: Increases resolution with superior quality and detail enhancement
- **Benefits**:
  - Better than traditional cubic interpolation
  - Automatic detail enhancement and noise reduction
  - Consistent quality across different input types
- **Parameters**: Scale factor = 2.0 (default), model = RealESRGAN_x2plus or x4plus

### Step 2: Bilateral Filtering
- **Method**: Edge-preserving denoising
- **Purpose**: Additional noise reduction while maintaining digit edges
- **Parameters**:
  - Diameter: 9 pixels
  - Sigma color: 75.0
  - Sigma space: 75.0

### Step 3: Grayscale Conversion
- **Method**: Color to grayscale conversion
- **Purpose**: Simplifies to single-channel representation
- **Formula**: Standard RGB to grayscale: 0.299R + 0.587G + 0.114B

### Step 4: Otsu Thresholding
- **Method**: Automatic binary conversion
- **Purpose**: Creates clean black-and-white digit images
- **Algorithm**: Otsu's method for optimal threshold selection

## Real-ESRGAN Integration

### What is Real-ESRGAN?
Real-ESRGAN (Real-Time Super-Resolution Generative Adversarial Network) is a state-of-the-art AI model that provides:
- **Superior Upscaling**: 2x and 4x magnification with better quality than traditional methods
- **Detail Enhancement**: Automatically improves texture and edge sharpness
- **Noise Reduction**: Intelligent denoising while preserving important features
- **Real-Time Performance**: Optimized for fast processing on standard hardware

### Usage in Digit Preprocessing
```python
# Automatic AI enhancement
enhanced = enhance_digit(image, upscale_factor=2.0)  # For model training

# Complete preprocessing pipeline
binary = sharpen_digit(image, target_size=128)  # For classification
```

### Model Management
- **Automatic Download**: Model weights are downloaded automatically on first use
- **Caching**: Weights are cached locally for subsequent runs (~200MB)
- **Hardware Support**: Works on both CPU and GPU (GPU recommended for speed)

## Usage Examples

### Single Image Preprocessing
```python
from image_preprocessing.digit_preprocessor import sharpen_digit, enhance_digit
import cv2

# Load digit crop
img = cv2.imread('digit_crop.png')

# For classification (binary output)
binary = sharpen_digit(img, target_size=64)

# For model training/inference (enhanced but not binary)
enhanced = enhance_digit(img, upscale_factor=2.0)

# Get intermediate steps for debugging
processed, steps = preprocess_digit(img, return_intermediate=True)
cv2.imshow('Original', steps['original'])
cv2.imshow('Enhanced', steps['upscaled'])
cv2.imshow('Binary', processed)
```

### Batch Processing
```python
from image_preprocessing.digit_preprocessor import batch_sharpen_digits

# Process multiple digit crops
digit_crops = [crop1, crop2, crop3]
processed_batch = batch_sharpen_digits(digit_crops, target_size=128)
# Returns: shape (N, 128, 128)
```

### Individual Pipeline Steps (Internal Use)
```python
from image_preprocessing.digit_preprocessor import (
    upscale_image, apply_bilateral_filter,
    convert_to_grayscale, apply_otsu_threshold
)

# Apply individual steps (not recommended for end users)
upscaled = upscale_image(img, scale_factor=2.0)  # Uses Real-ESRGAN if available
denoised = apply_bilateral_filter(upscaled)
gray = convert_to_grayscale(denoised)
binary, threshold = apply_otsu_threshold(gray)
```
upscaled = upscale_image(img, scale_factor=2.0)
denoised = apply_bilateral_filter(upscaled)
sharpened = apply_unsharp_mask(denoised, strength=1.5)
gray = convert_to_grayscale(sharpened)
binary, threshold = apply_otsu_threshold(gray)
```

## Testing and Validation

### Preprocessing Validation
```bash
# Test preprocessing on sample images
python src/test_files/test_preprocessing.py

# Test specific images
python src/test_files/test_preprocessing.py path/to/digit1.png path/to/digit2.png
```

### Enhanced Pipeline Testing
```bash
# Test complete YOLO + preprocessing pipeline
python src/test_files/test_enhanced_pipeline_health_check.py

# Test specific number of images
python src/test_files/test_enhanced_pipeline_health_check.py --num 10
```

## Output Visualization

The enhanced pipeline test generates 3-panel visualizations showing:

1. **Original Image with YOLO Boxes**: Input image with detected digit bounding boxes
2. **Original Digit Crops**: Raw cropped regions extracted from bounding boxes
3. **Enhanced Digit Crops**: Same crops after preprocessing enhancement

Visualizations are saved to `outputs/preprocessing_enhanced_test/`

## Technical Specifications

- **Target Size**: Configurable (default: 64×64 pixels)
- **Input Format**: Any image format supported by OpenCV
- **Output Format**: Binary uint8 (0-255)
- **Dependencies**: OpenCV (cv2), NumPy
- **Performance**: Optimized for real-time processing

## Integration with YOLO Pipeline

This preprocessing module is designed to work seamlessly with the YOLO digit detection pipeline:

1. YOLO detects digit bounding boxes
2. Digit crops are extracted
3. **Preprocessing enhancement is applied**
4. Enhanced digits are fed to classification model
5. Final digit predictions are generated

## Benefits

- **Improved Accuracy**: Enhanced contrast and clarity boost classification performance
- **Noise Reduction**: Bilateral filtering removes artifacts while preserving edges
- **Consistent Quality**: Standardized preprocessing ensures reliable results
- **Real-time Performance**: Optimized algorithms suitable for live processing
- **Configurable Parameters**: Adjustable settings for different use cases