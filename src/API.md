# ExtractNumbers Source Code API Documentation

This document describes the purpose and usage of each script in the `src` folder.

---

## Root Level Scripts

### `prep_data.py`
**Purpose:** Automated data preparation pipeline orchestrator  
**Description:** Runs all data processing scripts in the correct order to prepare raw datasets into formatted classification and segmentation datasets.

**Usage:**
```bash
# Run the full pipeline
python src/prep_data.py

# Run with cleanup - removes existing output directories before processing
python src/prep_data.py --clean
```

**Arguments:**
- `--clean` (optional): Deletes existing `data/classification` and `data/segmentation` directories before running

**Outputs:**
- Processed classification data in `data/classification/single_digits/` (10 directories for digits 0-9)
- Segmentation data in `data/segmentation/` (handwritten, synthetic, natural)

---

## bounding_box/ - YOLO Object Detection

### `yolo_detector.py`
**Purpose:** Core YOLO digit detection utilities  
**Description:** Provides utility functions for digit bounding box detection and coordinate conversions.

**Key Functions:**
- `bbox_from_mask(mask: np.ndarray)` → `(x1, y1, x2, y2)`: Extract bounding box from binary mask
- `xyxy_to_yolo_bbox(xyxy, w, h)` → `(cx, cy, w, h)`: Convert bounding box format from xyxy (pixel coordinates) to YOLO normalized format
- `ensure_dir(path: str)`: Create directory if it doesn't exist
- `iter_samples(dataset_root, categories)` → `List[Dict]`: Iterate through dataset samples

**Usage Example:**
```python
from bounding_box.yolo_detector import bbox_from_mask, xyxy_to_yolo_bbox
import cv2
import numpy as np

# Read a mask
mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)

# Get bounding box from mask
bbox = bbox_from_mask(mask)  # Returns (x1, y1, x2, y2)

# Convert to YOLO format
yolo_bbox = xyxy_to_yolo_bbox(bbox, img_width, img_height)  # Returns normalized coordinates
```

---

### `run_yolo_flow.py`
**Purpose:** Complete YOLO training and detection workflow  
**Description:** Orchestrates the entire YOLO pipeline including model training, predictions, and comparison with digit classifier.

**Usage:**
```bash
# Run the full YOLO workflow
python src/bounding_box/run_yolo_flow.py
```

**What it does:**
1. Trains a YOLO model on digit detection dataset
2. Generates predictions on test images
3. Compares YOLO detections with digit classifier predictions
4. Saves results to `outputs/bbox_comparison/`

**Key Functions:**
- `run_quiet_script(script_name, args=[])`: Execute scripts with error handling
- `analyze_epochs(csv_path)`: Analyze YOLO training progress and provide recommendations

**Outputs:**
- YOLO model weights and training logs
- Predictions CSV: `outputs/bbox_comparison/yolo_predictions.csv`
- Digit predictions CSV: `outputs/bbox_comparison/digit_predictions.csv`

---

### `visualize_yolo_results.py`
**Purpose:** Visualize YOLO detection results  
**Description:** Creates visual comparisons of YOLO detections with ground truth or digit classifier predictions.

**Usage:**
```bash
python src/bounding_box/visualize_yolo_results.py
```

**Outputs:**
- Visualization images in `outputs/bbox_comparison/`

---

## digit_recognizer/ - Digit Classification

### `digit_recognizer.py`
**Purpose:** Digit classification model (ResNet18-based)  
**Description:** Provides a PyTorch-based digit classifier using a fine-tuned ResNet18 model. Automatically trains from classification data if weights don't exist.

**Key Functions:**
- `get_device()` → `torch.device`: Get CUDA device if available, else CPU
- `build_digit_model()` → `nn.Module`: Create ResNet18 model with 10 output classes (digits 0-9)
- `load_classifier(model_path, data_dir, epochs=3, batch_size=64)` → `nn.Module`: 
  - Loads existing model if present
  - Otherwise trains a new model from classification data
  - Returns model in evaluation mode
- `preprocess_crop(image, crop_size=64)`: Preprocess digit crop for classification

**Usage Example:**
```python
from digit_recognizer.digit_recognizer import load_classifier, get_device
import torch

# Load or train classifier
model = load_classifier(
    model_path="outputs/bbox_comparison/digit_classifier.pth",
    data_dir="data/classification",
    epochs=3,
    batch_size=64
)

# Use model for inference
device = get_device()
with torch.no_grad():
    logits = model(preprocessed_images)  # outputs shape (batch, 10)
    predictions = torch.softmax(logits, dim=1)
    class_ids = torch.argmax(predictions, dim=1)
```

**Parameters:**
- `model_path`: Path to save/load model weights
- `data_dir`: Path to classification dataset (expects `single_digits/0-9/` subdirectories)
- `epochs`: Training epochs (default: 3)
- `batch_size`: Batch size for training (default: 64)

**Outputs:**
- Trained model weights at `model_path`

---

## image_preprocessing/ - Digit Image Enhancement

### `digit_preprocessor.py`
**Purpose:** Preprocessing module for digit image enhancement  
**Description:** Provides efficient, reusable functions for improving cropped digit images through a multi-step pipeline optimized for classification. Handles upscaling, denoising, sharpening, grayscale conversion, and binary thresholding.

**Pipeline Steps:**
1. **Upscale** - Cubic interpolation for high-quality enlargement
2. **Bilateral Filter** - Edge-preserving denoising
3. **Unsharp Masking** - Sharpening for feature enhancement
4. **Grayscale Conversion** - Convert to single channel
5. **Otsu Thresholding** - Automatic binary conversion

**Key Functions:**

- `upscale_image(image, scale_factor=2.0, interpolation=cv2.INTER_CUBIC)` → `np.ndarray`
  - Upscale image by factor using cubic interpolation
  
- `apply_bilateral_filter(image, diameter=9, sigma_color=75.0, sigma_space=75.0)` → `np.ndarray`
  - Apply bilateral filter for noise reduction while preserving edges
  
- `apply_unsharp_mask(image, kernel_size=(5,5), sigma=1.0, strength=1.5, threshold=0)` → `np.ndarray`
  - Sharpen image by enhancing edges and details
  
- `convert_to_grayscale(image)` → `np.ndarray`
  - Convert BGR or already grayscale image to grayscale
  
- `apply_otsu_threshold(image, binary=True)` → `Tuple[np.ndarray, float]`
  - Apply Otsu's automatic thresholding, returns (binary_image, threshold_value)

- `preprocess_digit(image, target_size=None, upscale_factor=2.0, bilateral_diameter=9, unsharp_strength=1.5, return_intermediate=False)` → `Union[np.ndarray, Tuple[np.ndarray, dict]]`
  - Complete preprocessing pipeline combining all steps
  - Returns processed binary image or (image, intermediate_steps_dict) if return_intermediate=True
  - intermediate_steps dict contains: 'original', 'upscaled', 'denoised', 'sharpened', 'grayscale', 'binary', 'threshold_value'

- `batch_preprocess_digits(images, **kwargs)` → `np.ndarray`
  - Preprocess multiple digit images at once, returns stacked array

**Usage Example:**
```python
from image_preprocessing.digit_preprocessor import preprocess_digit, batch_preprocess_digits
import cv2

# Single image preprocessing
img = cv2.imread('digit_crop.png')
processed = preprocess_digit(img, target_size=128, upscale_factor=2.0)
# Returns: (128, 128) binary image

# With intermediate steps for debugging
final, steps = preprocess_digit(img, return_intermediate=True)
cv2.imshow("Original", steps['original'])
cv2.imshow("Sharpened", steps['sharpened'])
cv2.imshow("Binary", steps['binary'])

# Batch processing
digit_crops = [crop1, crop2, crop3, ...]
processed_batch = batch_preprocess_digits(digit_crops, target_size=128)
# Returns: shape (N, 128, 128)

# Run as script with visualization
# python src/image_preprocessing/digit_preprocessor.py path/to/digit.png
```

**Parameters:**
- `upscale_factor`: How much to enlarge the image (2.0 = 2x larger)
- `bilateral_diameter`: Size of bilateral filter kernel (larger = more smoothing)
- `unsharp_strength`: Sharpening intensity (>1.0 = sharper)
- `target_size`: Final image size (e.g., 128 for 128×128)
- `return_intermediate`: If True, get all intermediate processing steps

**Outputs:**
- Binary preprocessed image (uint8, 0-255)
- Optional: Dictionary with all intermediate steps for analysis/debugging

---

## tests/ - Testing & Validation

### `test_preprocessing.py`
**Purpose:** Validate digit preprocessing module  
**Description:** Tests the preprocessing pipeline by loading sample images, applying all processing steps, and saving intermediate outputs for visual inspection. Provides both standalone function and CLI interface.

**Key Functions:**
- `test_preprocessing(image_path, output_dir=None)` → `bool`
  - Test preprocessing on single image
  - Saves all 6 intermediate steps to output directory (defaults to outputs/preprocessing_test_output)
  - Returns True if test passed, False otherwise

**Usage:**
```bash
# Auto-detect and test sample images from data folder
python tests/test_preprocessing.py

# Test specific images
python src/test_files/test_preprocessing.py path/to/digit1.png path/to/digit2.png
```

**Test Output:** Generates 6 PNG images showing each pipeline stage (saved to outputs/preprocessing_test_output/):
- `01_original.png` - Input image
- `02_upscaled.png` - After upscaling (2x)
- `03_denoised.png` - After bilateral filtering
- `04_sharpened.png` - After unsharp masking
- `05_grayscale.png` - After grayscale conversion
- `06_binary.png` - Final binary image after Otsu thresholding

**Python Usage:**
```python
from test_files.test_preprocessing import test_preprocessing

# Run test and save outputs to outputs/preprocessing_test_output
success = test_preprocessing(image_path='path/to/digit.png')

if success:
    print("✓ Preprocessing validation passed!")
    # Check outputs/preprocessing_test_output/ folder for intermediate images
```

**Typical Output:**
```
✓ Loaded image: shape (100, 100, 3)
✓ Preprocessing completed successfully
  - Original shape: (100, 100, 3)
  - Processed shape: (128, 128)
  - Otsu threshold value: 127.00
✓ Saved test images to: outputs/preprocessing_test_output/...
```

---

### `test_visual_enhancement.py`
**Purpose:** Test YOLO detection + preprocessing enhancement sub-pipeline  
**Description:** Demonstrates the complete digit detection and enhancement workflow. Uses YOLO to find digit bounding boxes, then applies preprocessing enhancement to each cropped digit. Creates visual comparison showing original detections, original crops, and enhanced crops.

**Key Functions:**
- `visualize_enhanced_pipeline(image_path, output_dir, file_index)` → `bool`
  - Creates 3-panel visualization of the enhanced pipeline
  - Panel 1: Original image with YOLO bounding boxes
  - Panel 2: Original cropped digits
  - Panel 3: Enhanced cropped digits (after preprocessing)

- `test_enhanced_pipeline(image_path, output_dir=None)` → `bool`
  - Test single image through the enhanced pipeline
  - Saves visualization to outputs/preprocessing_enhanced_test/

**Usage:**
```bash
# Auto-test on sample images
python tests/test_enhanced_pipeline.py

# Test specific images
python src/test_files/test_enhanced_pipeline.py path/to/image1.jpg path/to/image2.png
```

**Output:** Saves visualization images to `outputs/preprocessing_enhanced_test/` showing:
- **Panel 1:** Original image with red bounding boxes around detected digits
- **Panel 2:** Side-by-side original digit crops (before enhancement)
- **Panel 3:** Side-by-side enhanced digit crops (after preprocessing pipeline)

**Pipeline Flow:**
1. Load YOLO model and detect digit bounding boxes
2. Extract original digit crops from bounding boxes
3. Apply preprocessing enhancement (upscale → denoise → sharpen → grayscale → threshold)
4. Create 3-panel visualization comparing all stages

**Python Usage:**
```python
from test_files.test_enhanced_pipeline import test_enhanced_pipeline

# Test enhanced pipeline on image
success = test_enhanced_pipeline('path/to/image.jpg')

if success:
    print("✓ Enhanced pipeline test passed!")
    # Check outputs/preprocessing_enhanced_test/ for visualization
```

---

### `test_health_check.py`
**Purpose:** Fast health check for enhanced YOLO + preprocessing pipeline  
**Description:** Quickly validates the YOLO detection + preprocessing pipeline by randomly selecting and testing images. Provides rapid feedback on pipeline health with minimal setup.

**Usage:**
```bash
# Test 6 random images (default)
python tests/test_health_check.py

# Test custom number of images
python src/test_files/test_enhanced_pipeline_health_check.py --num 10

# Reproducible random selection
python src/test_files/test_enhanced_pipeline_health_check.py --seed 42
```

**Arguments:**
- `--num, -n`: Number of random images to test (default: 6)
- `--seed, -s`: Random seed for reproducible results

**What it does:**
1. Scans data folders for available test images
2. Randomly selects specified number of images
3. Runs enhanced pipeline test on each
4. Provides summary with pass/fail statistics
5. Saves visualizations to `outputs/preprocessing_enhanced_test/`

**Output Example:**
```
Enhanced Pipeline Health Check - 6 Images
============================================================
Found 3376 total images
Randomly selected 6 images for testing:
  1. house_1491_1.png
  2. mnist_4123_0.png
  3. image.jpg
  ...

[1/6] Testing: house_1491_1.png
✓ Enhanced pipeline visualization saved: outputs/preprocessing_enhanced_test/enhanced_pipeline_0_house_1491_1.png
  - Detected 1 digits
  - Applied preprocessing enhancement to each crop
  ✓ PASSED

...

Results: 6/6 passed (100.0%)
Visualizations saved to: outputs/preprocessing_enhanced_test
🎉 All tests PASSED! Pipeline is working correctly.
```

---

## test_files/ - Existing Tests

### `download_datasets.py`
**Purpose:** Download raw datasets  
**Description:** Downloads MNIST, SVHN, and handwritten digit datasets from public sources.

**Usage:**
```bash
python src/data/download_datasets.py
```

**Datasets Downloaded:**
- **MNIST**: Training and test sets from torchvision
- **SVHN**: Street View House Numbers (training and test) from torchvision
- **SVHN Extra**: Test set bounding box annotations (digitStruct.mat)
- **Handwritten Digits**: 0-9 dataset from Kaggle (requires Kaggle API setup)

**Outputs:**
- Raw data in `data/raw/MNIST/`, `data/raw/svhn/`
- Kaggle data in `~/.cache/kagglehub/`

---

### `process_dataset.py`
**Purpose:** Process raw datasets into standardized format  
**Description:** Converts MNIST and SVHN datasets into normalized, augmented 64×64 images with optional noise injection.

**Usage:**
```bash
python src/data/process_dataset.py
```

**Processing Steps:**
1. Loads MNIST training data (first 5000 samples)
2. Loads SVHN training data (first 5000 samples)
3. Resizes images to 64×64
4. Converts to RGB
5. Optionally adds random letter noise (50% chance)
6. Saves to `data/classification/single_digits/0-9/`

**Outputs:**
- Processed classification images in `data/classification/single_digits/`

---

### `create_handwritten_seg.py`
**Purpose:** Create handwritten segmentation dataset  
**Description:** Processes handwritten digit segmentation data for training object detection models.

**Usage:**
```bash
python src/data/create_handwritten_seg.py
```

**Outputs:**
- Segmentation data in `data/segmentation/handwritten/`

---

### `create_synthetic_seg.py`
**Purpose:** Create synthetic segmentation dataset  
**Description:** Generates synthetic digit segmentation data (composite images with digits and masks).

**Usage:**
```bash
python src/data/create_synthetic_seg.py
```

**Outputs:**
- Synthetic segmentation data in `data/segmentation/synthetic/`

---

### `process_svhn_seg.py`
**Purpose:** Extract bounding boxes from SVHN dataset  
**Description:** Processes SVHN data to extract digit bounding boxes and creates segmentation dataset from house number images.

**Usage:**
```bash
python src/data/process_svhn_seg.py
```

**Outputs:**
- SVHN segmentation data in `data/segmentation/natural/`

---

## full_pipelines/ - End-to-End Solutions

### `single_photo_pipeline.py`
**Purpose:** Complete digit detection and recognition for a single image  
**Description:** Given an image path, detects all digit bounding boxes (YOLO) and classifies each digit. Returns annotated image and prediction data.

**Usage:**
```bash
python src/full_pipelines/single_photo_pipeline.py <image_path>
```

**Arguments:**
- `image_path`: Path to input image

**Key Functions:**
- `load_yolo_model(weights_path)`: Load trained YOLO detection model
- `load_digit_model(model_path)`: Load trained digit classifier
- `run_yolo_on_image(yolo_model, image_path, conf_thres=0.25, iou_thres=0.7)`: Detect digits
- `recognize_digits(digit_model, image_path, bboxes)`: Classify detected digit regions

**Outputs:**
- Annotated image with bounding boxes and predictions
- CSV with predictions: `outputs/fullpipelines_predictions/`

**Pipeline Flow:**
1. Load YOLO detection model
2. Load digit classification model
3. Run YOLO inference to find digit bounding boxes
4. Extract digit crops and preprocess
5. Run classifier on each crop
6. Save annotated result image

---

### `all_photos_pipeline.py`
**Purpose:** Batch process multiple images  
**Description:** Applies the full detection and recognition pipeline to all images in an input directory.

**Usage:**
```bash
python src/full_pipelines/all_photos_pipeline.py <input_dir>
```

**Arguments:**
- `input_dir`: Path to directory containing images

**Outputs:**
- Annotated images in `outputs/fullpipelines_predictions/`
- Combined predictions CSV

---

## test_files/ - Testing & Validation

### `test_digit_recognizer.py`
**Purpose:** Evaluate digit classifier accuracy  
**Description:** Tests the trained digit classifier on test dataset and reports accuracy metrics.

**Usage:**
```bash
python tests/test_digit_recognizer.py
```

**Outputs:**
- Accuracy report
- Per-class metrics

---

### `test_full_pipeline.py`
**Purpose:** Evaluate end-to-end pipeline performance  
**Description:** Tests the complete detection + recognition pipeline on images with ground truth labels.

**Usage:**
```bash
python tests/test_full_pipeline.py
```

**Outputs:**
- Pipeline accuracy metrics
- Detection accuracy (YOLO)
- Classification accuracy

---

### `test_svhn.py`
**Purpose:** Test on SVHN dataset  
**Description:** Evaluates the full pipeline on SVHN test set with house number images.

**Usage:**
```bash
python tests/test_svhn.py
```

**Outputs:**
- SVHN-specific accuracy metrics
- Natural image performance comparison

---

## Quick Start

### Setup and Data Preparation
```bash
# 1. Download datasets
python src/data/download_datasets.py

# 2. Process datasets
python src/prep_data.py --clean

# 3. Train models (automatic in pipeline)
python src/bounding_box/run_yolo_flow.py
```

### Use the Pipeline
```bash
# Process a single image
python src/full_pipelines/single_photo_full_pipeline_not_up_to_date.py path/to/image.jpg

# Process multiple images
python src/full_pipelines/all_photos_full_pipeline_not_up_to_date.py path/to/image/directory/

# Test and evaluate
python tests/test_digit_recognizer.py
python tests/test_full_pipeline.py
```

---

## Dependencies

Core dependencies (see `requirements.txt`):
- `torch`, `torchvision`: Deep learning framework
- `ultralytics`: YOLO models
- `opencv-python (cv2)`: Image processing
- `pandas`, `numpy`: Data manipulation
- `Pillow`: Image processing
- `kagglehub`: Kaggle dataset download
- `tqdm`: Progress bars

---

## Output Structure

```
outputs/
├── bbox_comparison/          # YOLO detection results
│   ├── digit_classifier.pth # Trained digit model
│   ├── yolo_predictions.csv # YOLO detections
│   ├── digit_predictions.csv # Classifier predictions
│   └── yolo_runs/           # YOLO training logs
├── fullpipelines_predictions/ # Single/batch pipeline results
└── paddle_pipeline_predictions/ # Alternative pipeline results (if used)
```
