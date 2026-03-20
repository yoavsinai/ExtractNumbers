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

Each segmentation sample is isolated in its own numeric folder (e.g., `data/segmentation/synthetic/0/image.jpg` and `data/segmentation/synthetic/0/mask.png`).
