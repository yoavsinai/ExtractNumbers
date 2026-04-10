"""
Digit Preprocessing Module

Provides efficient, reusable functions for digit image enhancement and preprocessing.
Optimizes cropped digit images for classification through a pipeline of:
1. Upscaling (cubic interpolation)
2. Denoising (bilateral filtering)
3. Sharpening (unsharp masking)
4. Grayscale conversion
5. Binary thresholding (Otsu's method)
"""

import cv2
import numpy as np
from typing import Tuple, Union, Optional


def upscale_image(
    image: np.ndarray,
    scale_factor: float = 2.0,
    interpolation: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    Upscale image using specified interpolation method.
    
    Args:
        image: Input image (numpy array)
        scale_factor: Scaling factor (e.g., 2.0 for 2x upscaling)
        interpolation: OpenCV interpolation method (default: INTER_CUBIC)
    
    Returns:
        Upscaled image
    
    Example:
        >>> img = cv2.imread('digit.png')
        >>> upscaled = upscale_image(img, scale_factor=2.0)
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    
    if scale_factor == 1.0:
        return image.copy()
    
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)


def apply_bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Apply bilateral filtering for edge-preserving denoising.
    
    Bilateral filter is effective at reducing noise while preserving edges,
    which is crucial for digit recognition.
    
    Args:
        image: Input image
        diameter: Diameter of each pixel neighborhood (default: 9)
        sigma_color: Filter sigma in the color space (default: 75.0)
        sigma_space: Filter sigma in the coordinate space (default: 75.0)
    
    Returns:
        Denoised image
    
    Example:
        >>> img = cv2.imread('digit.png')
        >>> denoised = apply_bilateral_filter(img, diameter=9)
    """
    if diameter % 2 == 0 or diameter < 3:
        raise ValueError("diameter must be odd and >= 3")
    
    return cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )


def apply_unsharp_mask(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    strength: float = 1.5,
    threshold: int = 0
) -> np.ndarray:
    """
    Apply unsharp masking for sharpening.
    
    Unsharp masking enhances edges and details in the digit image,
    improving feature distinctiveness for classification.
    
    Args:
        image: Input image (grayscale or color)
        kernel_size: Gaussian blur kernel size (must be odd)
        sigma: Standard deviation for Gaussian blur
        strength: Strength of sharpening effect (1.0 = original, >1.0 = sharper)
        threshold: Intensity threshold for blending
    
    Returns:
        Sharpened image
    
    Example:
        >>> img = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
        >>> sharpened = apply_unsharp_mask(img, kernel_size=(5, 5), strength=1.5)
    """
    if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
        raise ValueError("kernel_size values must be odd")
    
    if strength < 0:
        raise ValueError("strength must be non-negative")
    
    # Create blurred version
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Compute the mask
    sharpened = cv2.addWeighted(
        image,
        1.0 + strength,
        blurred,
        -strength,
        0
    )
    
    # Apply intensity threshold for selective sharpening
    if threshold > 0:
        diff = cv2.absdiff(image, blurred)
        mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else mask
        sharpened = np.where(mask > 0, sharpened, image)
    
    # Clip values to valid range
    return np.clip(sharpened, 0, 255).astype(image.dtype)


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if not already.
    
    Args:
        image: Input image (BGR or grayscale)
    
    Returns:
        Grayscale image (single channel)
    
    Example:
        >>> img = cv2.imread('digit.png')  # BGR
        >>> gray = convert_to_grayscale(img)
    """
    if image.ndim == 2:
        # Already grayscale
        return image
    elif image.ndim == 3:
        # Color image - convert from BGR to gray
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image dimensions: {image.ndim}")


def apply_otsu_threshold(
    image: np.ndarray,
    binary: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Apply Otsu's thresholding for automatic binary conversion.
    
    Otsu's method automatically determines the optimal threshold value,
    converting the digit to pure black and white for clean classification.
    
    Args:
        image: Input grayscale image
        binary: If True, returns binary image; if False, returns threshold value too
    
    Returns:
        Tuple of (thresholded_image, threshold_value)
        If binary=False, returns (binary_image, threshold_value)
    
    Example:
        >>> img = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
        >>> binary, thresh_val = apply_otsu_threshold(img)
        >>> print(f"Threshold value: {thresh_val}")
    """
    if image.ndim != 2:
        raise ValueError("apply_otsu_threshold requires grayscale image")
    
    threshold_value, binary_image = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary_image, float(threshold_value)


def preprocess_digit(
    image: np.ndarray,
    target_size: Optional[int] = None,
    upscale_factor: float = 2.0,
    bilateral_diameter: int = 9,
    unsharp_strength: float = 1.5,
    return_intermediate: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Complete preprocessing pipeline for digit enhancement.
    
    Applies all preprocessing steps in sequence:
    1. Upscale (cubic interpolation for quality)
    2. Bilateral filter (preserve edges, reduce noise)
    3. Unsharp masking (sharpen features)
    4. Grayscale conversion
    5. Otsu thresholding (binary conversion)
    
    Args:
        image: Input digit image (numpy array, BGR or grayscale)
        target_size: Optional target size to resize to after upscaling
        upscale_factor: Factor to upscale image (default: 2.0)
        bilateral_diameter: Bilateral filter diameter (default: 9)
        unsharp_strength: Unsharp mask strength (default: 1.5)
        return_intermediate: If True, return intermediate processing steps
    
    Returns:
        Processed binary image (uint8)
        Or if return_intermediate=True: Tuple of (final_image, intermediate_dict)
        
        intermediate_dict contains keys:
        - 'original': Original input
        - 'upscaled': After upscaling
        - 'denoised': After bilateral filter
        - 'sharpened': After unsharp masking
        - 'grayscale': After grayscale conversion
        - 'binary': Final binary image
        - 'threshold_value': Otsu threshold value
    
    Example:
        >>> img = cv2.imread('digit.png')
        >>> processed = preprocess_digit(img, target_size=128)
        
        >>> # Get intermediate steps for debugging
        >>> processed, steps = preprocess_digit(img, return_intermediate=True)
        >>> cv2.imshow('Original', steps['original'])
        >>> cv2.imshow('Sharpened', steps['sharpened'])
        >>> cv2.imshow('Binary', steps['binary'])
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid")
    
    intermediate = {}
    
    # Step 1: Store original
    original = image.copy()
    intermediate['original'] = original
    
    # Step 2: Upscale
    upscaled = upscale_image(image, scale_factor=upscale_factor)
    intermediate['upscaled'] = upscaled
    
    # Step 3: Apply bilateral filter (denoise)
    denoised = apply_bilateral_filter(upscaled, diameter=bilateral_diameter)
    intermediate['denoised'] = denoised
    
    # Step 4: Apply unsharp masking (sharpen)
    sharpened = apply_unsharp_mask(denoised, strength=unsharp_strength)
    intermediate['sharpened'] = sharpened
    
    # Step 5: Convert to grayscale
    grayscale = convert_to_grayscale(sharpened)
    intermediate['grayscale'] = grayscale
    
    # Step 6: Apply Otsu threshold
    binary, threshold_value = apply_otsu_threshold(grayscale)
    intermediate['binary'] = binary
    intermediate['threshold_value'] = threshold_value
    
    # Optional: Resize to target size
    if target_size is not None:
        binary = cv2.resize(binary, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        intermediate['binary'] = binary
    
    if return_intermediate:
        return binary, intermediate
    
    return binary


def batch_preprocess_digits(
    images: list,
    **kwargs
) -> np.ndarray:
    """
    Preprocess multiple digit images in batch.
    
    Args:
        images: List of digit images (numpy arrays)
        **kwargs: Arguments to pass to preprocess_digit()
    
    Returns:
        Numpy array of preprocessed images (shape: [N, height, width])
    
    Example:
        >>> images = [cv2.imread(f'digit_{i}.png') for i in range(10)]
        >>> processed = batch_preprocess_digits(images, target_size=128)
        >>> print(processed.shape)  # (10, 128, 128)
    """
    processed_images = []
    
    for img in images:
        processed = preprocess_digit(img, **kwargs)
        processed_images.append(processed)
    
    return np.array(processed_images)


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        
        # Read image
        img = cv2.imread(image_path)
        
        # Preprocess with intermediate tracking
        processed, steps = preprocess_digit(
            img,
            target_size=128,
            return_intermediate=True
        )
        
        print("Preprocessing completed!")
        print(f"Original shape: {steps['original'].shape}")
        print(f"Processed shape: {processed.shape}")
        print(f"Otsu threshold value: {steps['threshold_value']:.2f}")
        
        # Display results (if GUI available)
        try:
            cv2.imshow("Original", steps['original'])
            cv2.imshow("Upscaled", steps['upscaled'])
            cv2.imshow("Denoised", steps['denoised'])
            cv2.imshow("Sharpened", steps['sharpened'])
            cv2.imshow("Grayscale", steps['grayscale'])
            cv2.imshow("Binary", processed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Cannot display images: {e}")
    
    else:
        print("Usage: python digit_preprocessor.py <image_path>")
        print("\nExample functions:")
        print("  - preprocess_digit(image): Full preprocessing pipeline")
        print("  - upscale_image(image, scale_factor=2.0)")
        print("  - apply_bilateral_filter(image)")
        print("  - apply_unsharp_mask(image, strength=1.5)")
        print("  - convert_to_grayscale(image)")
        print("  - apply_otsu_threshold(image)")
        print("  - batch_preprocess_digits(images)")
