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
*   **Sharpening(Stage 2):** מתי פה תוסיף מה תכלס הוספת. איזה מודל. יעני משהו בסגנון של "היה חידוד בסיסי ועכשיו זה מודל דיפלרנינג של חידוד של .."
*   **Final Classification (Stage 4):** Implemented a **ResNet18** classifier to convert isolated digit crops into final numeric strings.
*   **Succession Rate Metric:** Introduced a conditional probability metric to measure sequence consistency:
  $$P(D_{i+1} \text{ correct} | D_i \text{ correct})$$


### 📈 Results
#### 1. Individual Digit Localization

******מתיייייי תעדכן את מה שקיבלת********


| Metric | Stage 3 Value | 
| :--- | :--- | 
| **Mean IoU** | **0.7390** | 
| **Precision** | **98.45%** ⬆️ | 
| **Recall** | **98.22%** ⬆️| 

> **Note:** The integration of AI sharpening significantly improved the localization precision by reducing artifacts that previously confused the detector.



#### 2. Image Sharpening Comparison

******מתיייייי תעדכן את מה שקיבלת********



| Method | Classification Accuracy |
| :--- | :--- | 
| **Real-ESRGAN (AI)** | **98.2%** |
| Traditional (Unsharp - Stage 2) | 
| No-Sharpen | 89.6% | 
#### 3. Digit Classification

******מתיייייי תעדכן את מה שקיבלת********


| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0-9 Avg** | **0.97** | **0.97** | **0.97** | 76,803 |

#### 4. Full End-to-End Pipeline Performance

******מתיייייי תעדכן את מה שקיבלת********



| Metric | Overall | Natural (SVHN) | Handwritten |
| :--- | :--- | :--- | :--- |
| **Full Sequence Accuracy** | **79.20%** | 80.33% | 47.06% |
| **Mean Digit Accuracy (Pos)** | **88.86%** | 89.25% | 76.44% |
| **Succession Rate** | **---** | Measured per sequence | |

###
> **Conclusion:** The shift to Real-ESRGAN and ResNet18 transformed the project from a "box detector" to a full "text extractor." The **98.2% accuracy** in the sharpening stage confirms that AI restoration is critical for resolving low-quality digit samples.


**Full Pipeline Progression:**

****** יעני תמונה מתיייייי תעדכן את מה שקיבלת********
 
![Stage 3 Step-by-Step Visualization](assets/example3.png)