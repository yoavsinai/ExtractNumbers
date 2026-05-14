# ExtractNumbers Source Code API Documentation

This document describes the purpose and usage of each module and script in the `src` folder.

---

## Root Level Scripts

### `prep_data.py`
**Purpose:** Automated data preparation pipeline orchestrator  
**Description:** Runs data processing scripts for SVHN, Race Numbers, and Handwritten datasets. It converts raw data into a unified structure and applies high-level augmentations.

**Usage:**
```bash
python src/prep_data.py [--clean] [--limit N] [--datasets svhn handwritten ...]
```

---

## `data/` - Data Loading and Preparation

### Unified Data Structure
Processed data is saved in `data/digits_data/` with the following structure:
- `data/digits_data/<dataset>/sample_<id>/original.png`
- `data/digits_data/<dataset>/sample_<id>/annotations.json`

### Metadata Schema (`annotations.json`)
Standardized JSON format for localization and recognition:
- `image_metadata`: Dimensions and index.
- `detected_numbers`: Array of number sequences, each with a `full_bounding_box` and individual `digits` (label + bbox).

### Modules
- `svhn.py`, `race_numbers.py`, `handwritten.py`: Dataset-specific loaders.
- `apply_augmentations.py`: Applies Gaussian noise, blur, and geometric stretches (auto-adjusting bboxes).

---

## `training/` - Model Orchestration & Training

### `train_pipeline.py`
**Purpose:** Master orchestrator for training the multi-stage system.
**Description:** Manages dependencies between stages. It ensures that Stage 3 (IndividualBB) is trained on the sharpened crops produced by Stage 2.

---

## `inference/` - Single Sample Prediction

### `predict_single.py`
**Purpose:** Full 4-stage prediction on an image.
**Description:** Detection -> Sharpening -> Localization -> Classification. Returns the final number string.

---

## `bounding_box/` - YOLO Object Detection

### `globalbb_detector.py` (Stage 1)
**Purpose:** Global number sequence detector.

### `individualbb_detector.py` (Stage 3)
**Purpose:** Individual digit detector on sharpened crops.

---

## `digit_recognizer/` - Digit Classification (Stage 4)

### `digit_recognizer.py`
**Purpose:** ResNet18-based digit classification (0-9).

---

## `image_preprocessing/` - Image Enhancement (Stage 2)

### `digit_preprocessor.py`
**Purpose:** AI-powered image enhancement using Real-ESRGAN.

### 4-Step Enhancement Pipeline
1. **AI Upscaling (Real-ESRGAN)**: 2x magnification with detail recovery.
2. **Bilateral Filtering**: Edge-preserving denoising.
3. **Grayscale Conversion**: Standard luminance-based conversion.
4. **Otsu Thresholding**: Optimal binary conversion for classification.

---

## `evaluation/` - Metrics and Analysis

### `eval_all.py`
**Purpose:** Master evaluation orchestrator. Runs all stages + full pipeline benchmark.

### `eval_global_bbox.py`, `eval_sharpening.py`, `eval_individual_bbox.py`, `eval_digit_recog.py`
**Purpose:** Detailed stage-specific metrics with category breakdown.

### `eval_pipeline.py`
**Purpose:** Full End-to-End benchmark. Generates dashboard and error analysis.

---

## `utils/` - Shared Utilities
- `data_utils.py`: Dataset iteration and annotation parsing.
- `metrics.py`: IoU and classification report generation.
