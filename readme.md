# ExtractNumbers

A comprehensive image recognition and segmentation dataset generation pipeline.

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

## Dataset Structure

After running the preparation script, your `data/` directory will be structured as follows:

* **Classification** (`data/classification/`)
  * `single_digits/`: 5,000+ images per digit (0-9) from MNIST, SVHN, and Handwritten sources.
  * `multi_digits/`: 2,000 synthesized multi-digit sequences with surrounding letter noise.
* **Segmentation** (`data/segmentation/`)
  * `natural/`: 500 house number images (SVHN Format 1) with paired binary masks.
  * `synthetic/`: 500 high-noise synthetic images with paired binary masks.
  * `handwritten/`: 500 high-contrast handwritten digit samples with randomized color palettes and large distractor letters.
  * #### Data Augmentation & Noise Summary
      The segmentation dataset underwent various augmentation processes to improve model robustness, including White Noise, Blur, and Stretching/Pixelation:
      
      | Dataset Type | White Noise | Blur | Stretching / Pixelation |
      | :--- | :--- | :--- | :--- |
      | **Synthetic** | ✅ Applied globally to the entire image. | ✅ Applied globally to the entire image. | ✅ Applied globally to the entire image. |
      | **Handwritten** | ⚠️ Only on digits (from classification stage). | ⚠️ Only on digits (from classification stage). | ⚠️ Only on digits (from classification stage). |
      | **Natural (SVHN)** | ❌ Not applied; uses original quality. | ❌ Not applied; uses original quality. | ❌ Not applied; uses original quality. |


Each segmentation sample is isolated in its own numeric folder (e.g., `data/segmentation/synthetic/0/image.jpg` and `data/segmentation/synthetic/0/mask.png`).


## Extraction Pipeline

The project follows a multi-stage pipeline to ensure high accuracy in digit extraction and recognition:

![Process Pipeline](assets/diagram.PNG)

### Stage 1: Global Bounding-Box Detection (GlobalBB)

**How it works:** This stage identifies the entire number sequence as a single entity. The script scans the ground-truth masks, extracts bounding box coordinates for each valid digit blob, and builds a YOLO-compatible dataset. It performs an 80/20 train-validation split and trains a YOLOv8n model for 20 epochs.

**To run the GlobalBB pipeline:**
```bash
python "src/bounding_box/run_globalbb_flow.py"
```

> [!TIP]
> If you have already trained the GlobalBB model, you can skip the training phase and run inference only by appending the `--skip-train` flag:
> ```bash
> python "src/bounding_box/run_globalbb_flow.py" --skip-train
> ```

#### **Current Evaluation Results (Stage 1)**
The GlobalBB detection model achieves high accuracy across various noise levels:
* **Overall mAP50**: 94.47%
* **Precision**: 84.19%
* **Recall**: 92.34%

**Accuracy per Category (Average Confidence):**
* **Handwritten**: 74.85%
* **Natural**: 59.16%
* **Synthetic**: 75.41%

---

### Stage 2: Individual Digit Detection (IndividualBB)

**How it works:** This stage focuses on isolation and precision. We utilize a second YOLOv8 model trained specifically on **sharpened crops** of the number sequences detected in Stage 1. By upscaling and applying unsharp masking, the model can more accurately distinguish between tightly packed or overlapping digits.

**To generate the sharpened dataset:**
```bash
python "src/bounding_box/individualbb_detector.py" --prepare-only
```

**To run IndividualBB training:**
```bash
python "src/bounding_box/individualbb_detector.py" --train-only --epochs 20
```

#### **Evaluation Results (Stage 2)**
The IndividualBB model is trained to detect a single class ("digit") across all sharpened crops. The current model achieves high precision in isolating individual digits:
* **Overall mAP50**: 92.86%
* **Precision**: 88.54%
* **Recall**: 94.09%


---

### Full Automated Extraction Process

For a seamless experience, the entire multi-stage flow (Detection → Sharpening → Individual Localization) can be executed via a single command. The script handles model synchronization and dataset handoffs automatically.

**To run the full pipeline:**
```bash
python "src/full_pipelines/run_full_enhanced_flow.py"
```

#### **Control Flags**
The pipeline script offers granular control over the process:
* **`--skip-train`**: (Default behavior) Automatically skips training for any stage where valid model weights already exist.
* **`--force-train`**: Clears previous runs and forces a fresh training cycle for both GlobalBB and IndividualBB.
* **`--analyze-only`**: Skips the heavy detection and training phases entirely, generating reports from previous results.
* **`--viz-only`**: Quickly regenerates the 4-panel progression visualization using existing predictions.

#### **Pipeline Progression Visualization**
Below is the complete demonstration of the extraction process, showing the transformation from raw input to finalized digit localization:

![Full Pipeline Progression](assets/full_pipeline_progression.png)
