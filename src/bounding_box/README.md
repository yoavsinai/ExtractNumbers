# Bounding Box Detection

This module implements Stage 1 and Stage 3 of the extraction pipeline using YOLOv8.

## Stage 1: Global Detection
The `globalbb_detector.py` script trains a model to localize the entire number sequence area in noisy source images.

## Stage 3: Individual Detection
The `individualbb_detector.py` script trains a model to localize individual digits within the high-resolution sharpened crops produced in Stage 2.

## Key Files
- `globalbb_detector.py`: Training and inference logic for Stage 1.
- `individualbb_detector.py`: Training and inference logic for Stage 3.
- `run_globalbb_flow.py`: Orchestrates Stage 1 training and data conversion.
- `visualize_globalbb_results.py`: Visualizes detection performance.
