# ExtractNumbers

A comprehensive image recognition and segmentation dataset generation pipeline for digit extraction from noisy environments.

## Initial Setup

1. **Install Dependencies**:
   Ensure you have Python 3.12+ installed. Create a virtual environment and install the requirements:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Data Preparation**:
   The entire data fetching and processing pipeline is automated. Just run the following command from the project root:

   ```bash
   python src/prep_data.py
   ```

---

## 📂 Project Structure
The source code is organized into specialized modules:

* **[`src/training/`](src/training/README.md)**: Full pipeline training orchestrators.
* **[`src/inference/`](src/inference/README.md)**: Production prediction scripts.
* **[`src/data/`](src/data/README.md)**: Dataset loading and normalization.
* **[`src/bounding_box/`](src/bounding_box/README.md)**: Stage 1 & 3 YOLO detection.
* **[`src/image_preprocessing/`](src/image_preprocessing/README.md)**: Stage 2 Real-ESRGAN enhancement.
* **[`src/digit_recognizer/`](src/digit_recognizer/README.md)**: Stage 4 ResNet18 classification.
* **[`src/evaluation/`](src/evaluation/README.md)**: Multi-stage benchmarking suite.
* **[`src/utils/`](src/utils/README.md)**: Shared helper functions.

For a comprehensive technical reference of all scripts, see the **[Source API Documentation](src/API.md)**.

---

## Pipeline Workflow

The extraction process is divided into four main stages:

1.  **Global Bounding-Box Detection (GlobalBB):** Localizes the entire number sequence within the noisy source image.
2.  **Super-Resolution & Sharpening:** Implements **Real-ESRGAN** to enhance visual quality, recovery of fine details, and edge sharpening.
3.  **Individual Digit Localization (IndividualBB):** Detects and segments each digit individually within the sharpened crops.
4.  **Neural Character Recognition (Classification):** ResNet18-based classification of localized digits into final values (0-9).

![Process Pipeline](assets/diagram.PNG)

---

### Core Pipeline Execution

The system is designed for high-performance batch processing and seamless model synchronization.

**To train and run the full batch pipeline:**
```bash
python src/training/train_pipeline.py
```

**To run prediction on a single image:**
```bash
python src/inference/predict_single.py path/to/image.png
```

#### Control Flags
- `--skip-train`: Automatically skips training if valid weights already exist.
- `--force-train`: Forces a fresh training cycle for both YOLO stages.
- `--analyze-only`: Skips heavy detection/training and generates reports from previous results.
- `--viz-only`: Regenerates the progression visualizations from existing predictions.

---

## Evaluation & Insights

The pipeline is evaluated across four isolated stages and one comprehensive end-to-end benchmark.

### 🔍 Metric Definitions
To ensure clarity across all reports, the following metrics are used:
*   **Mean IoU (Intersection over Union)**: Measures the spatial overlap between the predicted bounding box and the ground truth. A score of 1.0 is a perfect match.
*   **Detection Rate**: The percentage of samples where the model successfully proposed at least one bounding box.
*   **mAP@0.5**: "Mean Average Precision" at a 50% IoU threshold. This is the standard accuracy metric for object detection.
*   **Precision**: The percentage of positive predictions that were actually correct (Quality).
*   **Recall**: The percentage of actual ground truth objects that were successfully detected (Quantity).
*   **Full Sequence Accuracy**: The percentage of images where the **entire** predicted number string exactly matches the ground truth.
*   **Mean Digit Accuracy (Pos)**: The percentage of digits correctly identified at their specific index in the sequence.
*   **Succession Rate**: The probability that a digit is correct given that the *previous* digit was correct. This measures the model's ability to maintain consistency across a sequence.

### 📊 Stage 1: Global Bounding Box Detection
*Evaluates the ability to localize the entire number sequence.*

| Category | Mean IoU | Detection Rate | mAP@0.5 |
| :--- | :--- | :--- | :--- |
| **Overall** | 0.7943 | 94.47% | 84.19% |
| **Natural** | - | - | - |
| **Handwritten**| - | - | - |

### 📊 Stage 2: Image Sharpening Comparison
*Compares AI-powered enhancement against traditional methods.*

| Method | Classification Accuracy |
| :--- | :--- |
| **Real-ESRGAN** | **98.2%** 🏆 |
| **Traditional** | 91.0% |
| **No-Sharpen** | 89.6% |

### 📊 Stage 3: Individual Digit Localization
*Evaluates digit segmentation within sharpened crops.*

| Category | Mean IoU | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **Overall** | 0.7390 | 98.45% | 98.22% |
| **Natural** | - | - | - |
| **Handwritten**| - | - | - |

### 📊 Stage 4: Digit Classification
*Isolated classification performance (ResNet18).*

| Digit | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0-9 Avg** | **0.97** | **0.97** | **0.97** | **76,803** |

### 🏆 Full End-to-End Pipeline Performance
*Master benchmark: Raw pixels → Final predicted string.*

| Metric | Overall | Natural (SVHN) | Handwritten |
| :--- | :--- | :--- | :--- |
| **Full Sequence Accuracy** | **79.20%** | **80.33%** | 47.06% |
| **Mean Digit Accuracy (Pos)**| **88.86%** | **89.25%** | **76.44%** |
| **Succession Rate** | **-** | **-** | **-** |

---

### How to Run Evaluations
The suite is divided into scripts for isolated performance analysis:

```bash
# Run ALL evaluations (Stages 1-4 + Full End-to-End Pipeline)
python src/evaluation/evaluate_all.py --max-samples 100

# Full End-to-End pipeline benchmark with error analysis dashboard
python src/evaluation/eval_pipeline.py --max-samples 500 --save-viz --analyze-errors
```

### Pipeline Progression
![Full Pipeline Dashboard](assets/full_pipeline_progression.png)

### Error Analysis
Detailed breakdown of how the model succeeds or fails at each individual step:
![Detailed Error Analysis](assets/detailed_error_analysis.png)
