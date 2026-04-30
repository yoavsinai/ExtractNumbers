# Data Preparation Module

Handles fetching, parsing, and normalization of datasets into a unified format.

## Unified Data Structure

Processed data is saved in `data/digits_data/`:

- `data/digits_data/<dataset>/sample_<id>/original.png`
- `data/digits_data/<dataset>/sample_<id>/annotations.json`

## Metadata Schema (`annotations.json`)

Standardized JSON format:

- `image_metadata`: Dimensions and index.
- `detected_numbers`: Full value, sequence bbox, and individual digit labels/bboxes.

## Key Files

- `svhn.py`, `race_numbers.py`, `handwritten.py`: Dataset-specific loaders.
- `apply_augmentations.py`: Applies Gaussian noise, blur, and geometric stretches.
- `augmentations.py`: Low-level augmentation implementations.
