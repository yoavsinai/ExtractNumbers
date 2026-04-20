# ExtractNumbers Source Code API Documentation

This document describes the purpose and usage of each module and script in the `src` folder.

---

## Root Level Scripts

### `prep_data.py`
**Purpose:** Automated data preparation pipeline orchestrator  
**Description:** Runs data processing scripts for SVHN, Race Numbers, and Handwritten datasets. It converts raw data into a unified structure and applies high-level augmentations.

**Usage:**
```bash
# Run the full pipeline
python src/prep_data.py

# Run with cleanup - removes existing data/digits_data before processing
python src/prep_data.py --clean

# Limit samples for quick testing
python src/prep_data.py --limit 100
```

**Key Parameters:**
- `--clean`: Deletes `data/digits_data` before running.
- `--limit`: Max samples per dataset.
- `--datasets`: List of datasets to process (default: svhn race_numbers handwritten).
- `--no-augment`: Skip the high-level augmentation phase.

---

## `data/` - Data Loading and Preparation

### `svhn.py`, `race_numbers.py`, `handwritten.py`
**Purpose:** Dataset-specific loaders.  
**Description:** Standardize raw data from different sources into the unified `data/digits_data` format. Each module provides a `prepare(output_dir, limit)` function.

### `apply_augmentations.py`
**Purpose:** Apply high-level image augmentations.  
**Description:** Uses `augmentations.py` to create augmented versions of the original samples, increasing dataset diversity.

---

## `bounding_box/` - YOLO Object Detection

### `globalbb_detector.py` (Stage 1 & 2)
**Purpose:** Multi-stage bounding box detector using YOLOv8.  
**Description:** 
- **Stage 1:** Detects the "Global" bounding box containing the entire number sequence.
- **Stage 2:** (Optional) Detects individual digits within the full image.
- Automates dataset conversion from masks to YOLO format and manages training.

**Key Functions:**
- `make_globalbb_dataset()`: Converts ground truth masks to YOLO global boxes.
- `make_digit_globalbb_dataset()`: Converts masks to per-digit YOLO boxes.
- `globalbb_predict_all()`: Runs GPU-accelerated batch inference on the full dataset.

### `individualbb_detector.py` (Stage 4)
**Purpose:** Individual digit detector on sharpened crops.  
**Description:** Trains a YOLO model specifically to find digits within the high-resolution sharpened crops produced in Stage 3.

**Key Functions:**
- `make_individualbb_dataset()`: Creates a dataset of sharpened crops with mapped digit coordinates.
- `train_individualbb()`: Trains the Stage 4 detector.

### `run_globalbb_flow.py` & `visualize_globalbb_results.py`
**Purpose:** Workflow orchestration and visualization for the global detection phase.

---

## `digit_recognizer/` - Digit Classification

### `digit_recognizer.py`
**Purpose:** ResNet18-based digit classification model.  
**Description:** Provides the core classification model (0-9) and data loading utilities.

**Key Components:**
- `DigitDataset`: PyTorch Dataset class for loading pre-cropped digits.
- `build_digit_model()`: Returns a ResNet18 model modified for 10-class classification.
- `load_classifier()`: Loads or trains the model weights.
- `preprocess_crop(img, bbox)`: Prepares a specific region of an image for classification.

---

## `image_preprocessing/` - Image Enhancement

### `digit_preprocessor.py` (Stage 3)
**Purpose:** Digit image enhancement using Real-ESRGAN and filtering.  
**Description:** Provides functions to upscale and sharpen low-resolution crops to improve classification accuracy.

**Key Functions:**
- `enhance_digit(image, upscale_factor=2.0)`: AI-powered super-resolution (Real-ESRGAN) + Bilateral filtering.
- `sharpen_digit()`: Full sharpening pipeline (upscale -> denoise -> grayscale -> threshold).
- `batch_sharpen_digits()`: Process multiple crops in parallel.

---

## `full_pipelines/` - End-to-End Extraction

### `predict_single.py`
**Purpose:** Run the full pipeline on a single image.  
**Description:** Executes Detection (Global) -> Sharpening -> Detection (Individual) -> Classification.

**Usage:**
```bash
python src/full_pipelines/predict_single.py path/to/image.jpg
```

### `batch_pipeline.py`
**Purpose:** High-performance batch processing for full datasets.  
**Description:** Processes entire directories or the `digits_data` folder using GPU batching for both YOLO and ResNet18.

---

## `evaluation/` - Metrics and Analysis

### `evaluate_classifier.py`
**Purpose:** Benchmarking the ResNet18 classifier.  
**Description:** Generates detailed classification reports and compares different sharpening methods' impact on accuracy.

### `evaluate_pipeline.py`
**Purpose:** End-to-end pipeline evaluation.  
**Description:** Benchmarks the full sequence (Stages 1-5) on ground truth data, calculating Sequence Accuracy, Mean Digit Accuracy, and IoU for detection stages. Generates a performance dashboard.

---

## `utils/` - Shared Utilities

### `data_utils.py`
**Purpose:** Unified data access.  
**Key Functions:**
- `iter_new_samples(data_root)`: Iterates through the `data/digits_data` structure.
- `get_gt_from_anno(anno_path)`: Parses `annotations.json` for global and digit-level ground truth.

### `metrics.py`
**Purpose:** Performance calculation.  
**Key Functions:**
- `calculate_iou(boxA, boxB)`: Standard Intersection over Union.
- `print_metrics_report()`: Formats and prints classification metrics via scikit-learn.

---

## `tests/` - Validation & Health Checks

- `test_health_check.py`: Quick validation of the YOLO + Preprocessing pipeline.
- `test_visual_enhancement.py`: Creates 3-panel visualizations (Original -> Crop -> Enhanced).
- `test_preprocessing.py`: Validates individual steps of the sharpening pipeline.
- `test_digit_recognizer.py`: Tests the classifier on a held-out set.
- `test_full_pipeline.py`: Validates the end-to-end logic on sample images.

---

## Output Structure

```
outputs/
├── bbox_comparison/          # Core pipeline outputs
│   ├── digit_classifier.pth  # Trained ResNet18 weights
│   ├── globalbb_run/         # Stage 1 YOLO weights
│   ├── individualbb_run/     # Stage 4 YOLO weights
│   ├── globalbb_predictions.csv
│   └── individualbb_predictions.csv
├── evaluation/               # Reports and metrics
└── full_pipeline_dashboard.png # Visualization from evaluate_pipeline.py
```
