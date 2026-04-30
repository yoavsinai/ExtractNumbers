# Evaluation Suite

Comprehensive benchmarking for every stage of the extraction pipeline.

---

## 📂 Evaluation Stages

### Stage 1: Global Bounding Box (`eval_global_bbox.py`)
**Purpose**: Evaluate the model's ability to localize the entire number sequence.
**Metrics**:
- **Mean IoU**: Average Intersection over Union with ground truth.
- **Detection Rate**: Percentage of images where a box was found.
- **mAP@0.5 / mAP@0.75**: Accuracy at specific IoU thresholds.

### Stage 2: Image Sharpening (`eval_sharpening.py`)
**Purpose**: Compare Real-ESRGAN enhancement against traditional methods.
**Metrics**:
- **Visual Comparison**: Side-by-side analysis of sharpened crops.
- **Downstream Accuracy**: How much the enhancement improves Stage 4 classification.

### Stage 3: Individual Bounding Box (`eval_individual_bbox.py`)
**Purpose**: Evaluate digit localization within sharpened crops.
**Metrics**:
- **Precision/Recall**: Per-digit localization accuracy.
- **IoU per Category**: Performance breakdown for Natural vs. Handwritten data.

### Stage 4: Digit Recognition (`eval_digit_recog.py`)
**Purpose**: Isolated classification performance of the ResNet18 model.
**Metrics**:
- **Precision/Recall/F1**: Standard classification metrics for digits 0-9.
- **Confusion Matrix**: Identifying common digit misidentifications.

---

## 🚀 End-to-End Pipeline Evaluation (`eval_pipeline.py`)

The master benchmark that tests the full sequence from raw pixels to final string.

### Specialized Metrics

#### 1. Mean Digit Accuracy (CER)
Measures the percentage of correctly identified digits at each position across the entire dataset.

#### 2. Single Digit Succession Rate
**New Metric**: Measures the conditional probability that a digit is correctly identified given that the *previous* digit in the sequence was correct.
- **Formula**: $P(D_{i+1} \text{ correct} | D_i \text{ correct})$
- **Significance**: High succession rates indicate that the model maintains positional consistency. Low rates suggest that a single error (like a shift in the bounding box) tends to cascade through the rest of the sequence.

---

## 📝 Evaluation Script Template
All evaluation scripts should follow this standardized structure:

```python
import os
import sys
import pandas as pd
from tqdm import tqdm

# 1. Setup paths and load models
def setup():
    # Load model weights, initialize results list
    pass

# 2. Main Evaluation Loop
def evaluate():
    # Iterate through data/digits_data/
    # Run inference
    # Calculate metrics (IoU, Accuracy, etc.)
    pass

# 3. Aggregation & Reporting
def report(results_df):
    # Group by category (Handwritten, Natural, Synthetic)
    # Print summary table
    # Save to outputs/reports/
    pass

if __name__ == "__main__":
    setup()
    evaluate()
```
