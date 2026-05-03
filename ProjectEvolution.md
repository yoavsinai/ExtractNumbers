# Project Evolution & Development Milestones

This section documents the iterative improvements made to the **ExtractNumbers** pipeline, focusing on the transition from simple global detection to a sophisticated, AI-powered multi-stage OCR system.

---

## 🟢 Stage 1: Foundation & Global Bounding-Box Detection
**Focus:** Establishing the automated data pipeline and baseline detection metrics.

*   **BB Strategy:** **Global Bounding Box (GlobalBB).** The model focused on identifying the **entire number** sequence as a single entity within noisy source images.
*   **Initial Setup:** Automated data fetching and processing pipeline (MNIST, SVHN, and Synthetic data).
*   **Key Model:** YOLOv8n (Initial training for 20 epochs).

### 📈 Results
| Metric | Value |
| :--- | :--- |
| **Overall mAP50** | **92.15%** |
| Precision | 89.58% |
| Recall | 81.04% |

| Category | Accuracy (Avg. Confidence) |
| :--- | :--- |
| Handwritten | 88.79% |
| **Natural (SVHN)** | **48.08%** (Baseline) |
| Synthetic | 64.08% |

**Example Result:**


<img src="assets/example1.png" width="400">


> **Conclusion:** While the pipeline was functional, the model struggled significantly with "Natural" images and lacked the precision to isolate overlapping digits.
---
## 🟢 Stage 2: Hierarchical 3-Step Process & Basic Sharpening 
**Focus:** Realizing that global detection isn't enough, we introduced a hierarchical flow to isolate individual digits.

**Architecture Diagram (The 3-Step Flow):**
![Stage 2 Architecture](assets/architecture2.png)

### 🔄 The 3-Step Workflow:
1.  **Global Detection:** Identify the Bounding Box (BB) for the **entire number sequence**.
2.  **Preprocessing:** Crop the global BB and apply **Basic Sharpening** (Unsharp Masking) and upscaling to separate tight digits (NEW).
3.  **Individual Localization:** Perform a second detection (IndividualBB) to find the **BB of each specific digit** within the sharpened crop (NEW).

*   **Augmentations:** Added White Noise, Blur, and Stretching/Pixelation to improve robustness against various image qualities.


### 📈 Results
#### 1. Global Bounding Box Detection
| Metric | Value |
| :--- | :--- |
| **Overall mAP50** | **94.47%** ⬆️ |
| Precision | 84.19% ⬇️|
| Recall | 92.34% ⬆️|

**Accuracy per Category (Average Confidence):**
| Category | Accuracy |
| :--- | :--- |
| **Handwritten** | 74.85% ⬇️|
| **Natural (SVHN)** | 59.16% ⬆️|
| **Synthetic** | 75.41% ⬆️|

#### 2. Individual Digit Detection (Stage 3 of Flow)
| Metric | Value |
| :--- | :--- |
| **Overall mAP50** | **92.86%** |
| Precision | 88.54% |
| Recall | 94.09% |


**Example Result:**
![Stage 2 Step-by-Step](assets/example2.png)

> **Conclusion:** While the split helped, we noticed that **basic sharpening was insufficient** for complex noise, and the dataset still lacked "character labels" for actual recognition.

---

## 🟢 Stage 3: AI Restoration & End-to-End OCR
**Focus:** Overhauling the data structure and integrating AI-powered restoration for production-grade accuracy.


**Final System Architecture:**
![Stage 3 Final Architecture](assets/architecture3.png)

*   **Data Overhaul:** **Complete Data Deletion & Replacement.** The dataset was transitioned to a **Unified Metadata Schema** (`annotations.json`). This new structure links every image to its actual numeric value (Labels), enabling full OCR capabilities.
*   **Sharpening (Stage 2):** Upgraded from basic unsharp/sharpen filters to the project’s AI image enhancement pipeline in `src/image_preprocessing/`, improving low-quality crop clarity before digit detection.
*   **Final Classification (Stage 4):** Implemented a **ResNet18** classifier to convert isolated digit crops into final numeric strings.
*   **Succession Rate Metric:** Introduced a conditional probability metric to measure sequence consistency:
  $$P(D_{i+1} \text{ correct} | D_i \text{ correct})$$


### 📈 Results
#### 1. Individual Digit Localization

| Metric | Value |
| :--- | :--- |
| Mean IoU (all digits) | 0.8267 |
| Mean Confidence | 0.7983 |
| Overall Recall | 101.24% |
| Total GT Digits | 4514 |
| Total Pred Digits | 4570 |

**Performance by Category:**
| Category | Mean IoU | Recall |
| :--- | :--- | :--- |
| Handwritten | 0.8748 | 100.77% |
| SVHN | 0.8389 | 101.27% |

> **Note:** The integration of AI enhancement improved digit localization precision and overall individual box coverage, as reflected by the 0.8267 mean IoU and very high recall across both categories.



#### 2. Image Sharpening Comparison

**Pipeline Performance Comparison:**

| Metric | With Sharpening | Without Sharpening (Baseline) |
| :--- | :--- | :--- |
| Full Sequence Accuracy | 68.20% | 68.00% |
| Mean Digit Accuracy (Pos) | 81.61% | 80.62% |
| Stage 1 (Global) Mean IoU | 0.7516 | 0.7612 |
| Stage 3 (Individual) Mean IoU | 0.7573 | 0.7585 |

**Sharpening Enhancement Metrics:**

| Metric | Result |
| :--- | :--- |
| Total Processed | 2000 |
| Average Duration | 0.0006s |
| Throughput | 5,135,519 pixels/sec |
| Avg Upscale Factor | 2.00x |
#### 3. Digit Classification

| Metric | Value |
| :--- | :--- |
| Overall Accuracy | 93.75% |
| Handwritten Accuracy | 98.85% |
| SVHN Accuracy | 93.44% |

Detailed per-digit classification scores are available in the evaluation report, with overall support of 4514 digits across both categories.

#### 4. Full End-to-End Pipeline Performance

| Metric | Overall | Handwritten | SVHN |
| :--- | :--- | :--- | :--- |
| Full Sequence Accuracy | 70.25% | 58.50% | 82.00% |
| Mean Digit Accuracy (Pos) | 81.89% | 74.33% | 89.46% |
| Stage 1 (Global) Mean IoU | 0.7635 | 0.7226 | 0.8044 |
| Stage 3 (Individual) Mean IoU | 0.7629 | 0.7810 | 0.7461 |

###
> **Conclusion:** The pipeline now produces a working end-to-end OCR workflow, with full-sequence success at 70.25% and strong digit-level accuracy across both handwritten and SVHN samples. Performance improved with the AI-powered sharpening enhancement in Stage 2.

> **Note:** The reported Stage 3 recall can exceed 100% because it is currently calculated as the ratio of total predicted digit boxes to total ground-truth digit boxes, not as the standard TP/(TP+FN) recall. This means extra predicted boxes can push the metric above 100%.

**Full Pipeline Progression:**

The generated dashboard and detailed error analysis are captured in the evaluation reports under `outputs/reports`.


example for image sharpening (not cherry picked. if you want you can alter the visualize_enhancement.py to run in a loop and cherry pick the best result)
![Stage 3 Step-by-Step Visualization](assets/visualization_unsharp_mask.png)
