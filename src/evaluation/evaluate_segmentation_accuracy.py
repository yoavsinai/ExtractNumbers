"""
Evaluate Enhancement Methods on Full Segmentation Datasets

This script is designed to evaluate how different image enhancement methods (Real-ESRGAN, 
No-Sharpen, Traditional, and Both) affect digit recognition accuracy across various 
segmentation datasets (handwritten, synthetic, natural). It extracts ground truth from 
masks, compares it with model predictions, and generates a comprehensive accuracy report 
to validate the full preprocessing pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from digit_recognizer.digit_recognizer import build_digit_model, get_device
from image_preprocessing.digit_preprocessor import (
    enhance_digit, enhance_without_sharpening, enhance_with_traditional_methods, enhance_with_both
)
from bounding_box.globalbb_detector import _read_mask_grayscale, extract_digit_bboxes
from utils.metrics import calculate_metrics, print_metrics_report


def load_segmentation_samples(data_root, categories=None, max_samples_per_category=50):
    """Load samples from segmentation datasets."""
    if categories is None:
        categories = ['handwritten', 'synthetic', 'natural']

    samples = []

    for category in categories:
        category_path = os.path.join(data_root, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} not found in {data_root}")
            continue

        # Get all subdirectories (numbered 0,1,2,... etc.)
        subdirs = [d for d in os.listdir(category_path)
                  if os.path.isdir(os.path.join(category_path, d))]

        # Check if the dataset has labels.txt
        has_labels = False
        for s in subdirs[:5]:
            if os.path.exists(os.path.join(category_path, s, 'labels.txt')):
                has_labels = True
                break
                
        if not has_labels:
            print(f"Skipping category '{category}' because it lacks ground truth 'labels.txt'")
            continue

        # Limit samples per category
        subdirs = subdirs[:max_samples_per_category]

        for subdir in subdirs:
            subdir_path = os.path.join(category_path, subdir)
            image_path = os.path.join(subdir_path, 'image.jpg')
            mask_path = os.path.join(subdir_path, 'mask.png')
            labels_path = os.path.join(subdir_path, 'labels.txt')

            if os.path.exists(image_path) and os.path.exists(mask_path) and os.path.exists(labels_path):
                samples.append({
                    'category': category,
                    'sample_id': subdir,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'labels_path': labels_path
                })

    print(f"Loaded {len(samples)} samples from {len(categories)} categories")
    return samples


def extract_ground_truth_labels(labels_path):
    """Extract individual digit bounding boxes and true labels from labels.txt."""
    digits_info = []
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # SVHN created labels as: left top right bottom label
                l, t, r, b, label = map(int, parts[:5])
                digits_info.append({
                    'bbox': (l, t, r, b),
                    'digit': label
                })
    # Sort by x-coordinate (left to right)
    digits_info = sorted(digits_info, key=lambda d: d['bbox'][0])
    for i, d in enumerate(digits_info):
        d['position'] = i
    return digits_info


def preprocess_image_for_model(image, bbox, enhancement_method=None):
    """Preprocess cropped digit image for model input."""

    # Crop the digit
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    # Apply enhancement method
    if enhancement_method == 'realesrgan':
        processed = enhance_digit(crop, upscale_factor=2.0)
    elif enhancement_method == 'no_sharpening':
        processed = enhance_without_sharpening(crop, target_size=64)
    elif enhancement_method == 'traditional':
        processed = enhance_with_traditional_methods(crop, target_size=64)
    elif enhancement_method == 'both':
        processed = enhance_with_both(crop, target_size=64)
    else:
        # Default: basic preprocessing
        processed = enhance_without_sharpening(crop, target_size=64)

    # Convert to RGB and PIL for torchvision transforms
    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    from torchvision.transforms import ToPILImage, Resize, ToTensor, Normalize, Compose
    pil_image = ToPILImage()(processed)

    # Apply model transforms
    transform = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(pil_image)


def evaluate_on_segmentation_datasets(model_path, data_root, max_samples_per_category=50):
    """Evaluate model accuracy on segmentation datasets with different enhancement methods."""

    print("🔬 EVALUATING ENHANCEMENT METHODS ON SEGMENTATION DATASETS")
    print("=" * 70)

    # Load model
    device = get_device()
    model = build_digit_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model from: {model_path}")
    print(f"✓ Using device: {device}")

    # Load samples
    samples = load_segmentation_samples(data_root, max_samples_per_category=max_samples_per_category)

    # Enhancement methods to test
    methods = ['realesrgan', 'no_sharpening', 'traditional', 'both']

    results = {}

    for method in methods:
        print(f"\n📊 Testing method: {method.upper()}")
        print("-" * 50)

        # Track results
        total_predictions = 0
        correct_predictions = 0
        category_results = defaultdict(lambda: {'total': 0, 'correct': 0})
        all_predictions = []

        # Process each sample
        for sample in tqdm(samples, desc=f"{method} Evaluation"):
            image_path = sample['image_path']
            labels_path = sample['labels_path']
            category = sample['category']

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue

            # Extract ground truth digit positions and labels
            try:
                gt_digits = extract_ground_truth_labels(labels_path)
            except Exception as e:
                print(f"Warning: Could not extract GT for {labels_path}: {e}")
                continue

            # For each ground truth digit, make prediction
            for gt_digit in gt_digits:
                bbox = gt_digit['bbox']
                true_digit = gt_digit['digit']

                # Preprocess the cropped digit
                processed_tensor = preprocess_image_for_model(image, bbox, method)
                if processed_tensor is None:
                    continue

                processed_tensor = processed_tensor.unsqueeze(0).to(device)

                # Get prediction
                with torch.no_grad():
                    outputs = model(processed_tensor)
                    probs = torch.softmax(outputs, dim=-1)
                    pred_digit = int(probs.argmax(dim=-1).item())
                    confidence = float(probs.max().item())

                # Track metrics
                total_predictions += 1
                category_results[category]['total'] += 1
                if pred_digit == true_digit:
                    correct_predictions += 1
                    category_results[category]['correct'] += 1

                all_predictions.append({
                    'category': category,
                    'sample_id': sample['sample_id'],
                    'method': method,
                    'pred_digit': pred_digit,
                    'true_digit': true_digit,
                    'confidence': confidence,
                    'bbox': bbox
                })

        # Store results for this method
        results[method] = {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'category_results': dict(category_results),
            'all_predictions': all_predictions
        }

        print(f"   Total predictions: {total_predictions}")
        print(f"   Accuracy: {correct_predictions / total_predictions if total_predictions > 0 else 0:.4f}")
        print("   Per-category breakdown:")
        for cat, stats in category_results.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"     {cat}: {acc:.4f} ({stats['correct']}/{stats['total']})")

    return results, samples


def analyze_predictions(results, samples):
    """Analyze predictions and create comprehensive report."""

    print(f"\n{'='*70}")
    print("📊 PREDICTION ANALYSIS REPORT (FULL PIPELINE)")
    print(f"{'='*70}")

    # Basic statistics
    print("\n🔢 Basic Statistics:")
    for method, data in results.items():
        total = data['total_predictions']
        correct = data['correct_predictions']
        acc = correct / total if total > 0 else 0
        print(f"   {method.upper()}: {total} predictions, Accuracy: {acc:.4f}")

    # Category breakdown
    print("\n📂 Accuracy by Category:")
    categories = ['handwritten', 'synthetic', 'natural']
    print("Category     | Real-ESRGAN | No-Sharpen | Traditional | Both")
    print("-------------|------------|------------|------------|------------")

    for cat in categories:
        def get_acc(method):
            stats = results[method]['category_results'].get(cat, {'total': 0, 'correct': 0})
            if stats['total'] == 0: return "-"
            return f"{stats['correct']/stats['total']:.4f}"

        realesrgan_acc = get_acc('realesrgan')
        no_sharp_acc = get_acc('no_sharpening')
        trad_acc = get_acc('traditional')
        both_acc = get_acc('both')

        print(f"{cat:12} | {realesrgan_acc:10} | {no_sharp_acc:10} | {trad_acc:10} | {both_acc:10}")

    # Metrics report per method
    for method, data in results.items():
        print(f"\n\n📈 Precision, Recall, F1-Score for {method.upper()}:")
        y_true = [p['true_digit'] for p in data['all_predictions']]
        y_pred = [p['pred_digit'] for p in data['all_predictions']]
        if len(y_true) > 0:
            print_metrics_report(y_true, y_pred, title=f"Metrics for {method.upper()}")
        else:
            print("   No predictions to evaluate.")

    # Confidence analysis
    print("\n📈 Confidence Analysis:")
    for method in results.keys():
        confidences = [p['confidence'] for p in results[method]['all_predictions']]
        if confidences:
            avg_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)

            print(f"   {method.upper()}:")
            print(f"     Average: {avg_conf:.4f}")
            print(f"     Min: {min_conf:.4f}")
            print(f"     Max: {max_conf:.4f}")

    return results


def save_results(results, samples, output_dir):
    """Save detailed results to files."""

    os.makedirs(output_dir, exist_ok=True)

    # Save all predictions
    all_predictions = []
    for method, data in results.items():
        for pred in data['all_predictions']:
            pred['method'] = method
            all_predictions.append(pred)

    predictions_df = pd.DataFrame(all_predictions)
    predictions_path = os.path.join(output_dir, "all_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    # Save summary statistics
    summary_data = []
    for method, data in results.items():
        summary_data.append({
            'method': method,
            'total_predictions': data['total_predictions'],
            'categories_covered': len(data['category_results'])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Save samples info
    samples_df = pd.DataFrame(samples)
    samples_path = os.path.join(output_dir, "samples.csv")
    samples_df.to_csv(samples_path, index=False)

    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - All predictions: {predictions_path}")
    print(f"   - Summary: {summary_path}")
    print(f"   - Samples: {samples_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate digit recognition accuracy on segmentation datasets")
    parser.add_argument("--model-path", type=str,
                       default=os.path.join(BASE_DIR, "outputs", "bbox_comparison", "digit_classifier.pth"),
                       help="Path to trained digit classifier model")
    parser.add_argument("--data-root", type=str,
                       default=os.path.join(BASE_DIR, "data", "segmentation"),
                       help="Path to segmentation datasets")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Maximum samples per category")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("Please train a digit classifier first.")
        sys.exit(1)

    # Check if data exists
    if not os.path.exists(args.data_root):
        print(f"❌ Data directory not found: {args.data_root}")
        print("Please run data preparation first.")
        sys.exit(1)

    # Run evaluation
    results, samples = evaluate_on_segmentation_datasets(
        args.model_path,
        args.data_root,
        max_samples_per_category=args.max_samples
    )

    # Analyze results
    analyze_predictions(results, samples)

    # Save results
    output_dir = os.path.join(BASE_DIR, "outputs", "segmentation_accuracy_evaluation")
    save_results(results, samples, output_dir)

    print(f"\n✅ Evaluation complete!")
    print("Results saved to segmentation datasets evaluation.")
if __name__ == "__main__":
    main()