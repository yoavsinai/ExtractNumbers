"""
Evaluate Enhancement Methods Accuracy

Tests digit recognition accuracy on classification dataset using different enhancement methods.
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from digit_recognizer import build_digit_model, get_device
from image_preprocessing.digit_preprocessor import (
    enhance_digit, enhance_without_sharpening, enhance_with_traditional_methods
)


class DigitDataset(Dataset):
    """Dataset for loading digit images with ground truth labels."""

    def __init__(self, root_dir, transform=None, max_samples_per_class=50):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_names = []

        # Load samples from each digit class
        for class_idx in range(10):
            class_dir = os.path.join(root_dir, str(class_idx))
            if os.path.exists(class_dir):
                self.class_names.append(str(class_idx))
                image_files = [f for f in os.listdir(class_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                # Limit samples per class for faster testing
                image_files = image_files[:max_samples_per_class]

                for img_file in image_files:
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, class_idx))

        print(f"Loaded {len(self.samples)} samples from {len(self.class_names)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def preprocess_image_for_model(image, enhancement_method=None):
    """Preprocess image for model input using specified enhancement method."""

    # Apply enhancement method
    if enhancement_method == 'realesrgan':
        processed = enhance_digit(image, upscale_factor=2.0)
    elif enhancement_method == 'no_sharpening':
        processed = enhance_without_sharpening(image, target_size=64)
    elif enhancement_method == 'traditional':
        processed = enhance_with_traditional_methods(image, target_size=64)
    else:
        # Default: basic preprocessing
        processed = enhance_without_sharpening(image, target_size=64)

    # Convert to RGB and PIL for torchvision transforms
    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    pil_image = T.ToPILImage()(processed)

    # Apply model transforms
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform(pil_image)


def evaluate_model_on_enhancement_methods(model_path, data_dir, max_samples_per_class=50):
    """Evaluate model accuracy with different enhancement methods."""

    print("🔬 EVALUATING ENHANCEMENT METHODS ACCURACY")
    print("=" * 60)

    # Load model
    device = get_device()
    model = build_digit_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded model from: {model_path}")
    print(f"✓ Using device: {device}")

    # Enhancement methods to test
    methods = ['realesrgan', 'no_sharpening', 'traditional']

    results = {}

    for method in methods:
        print(f"\n📊 Testing method: {method.upper()}")
        print("-" * 40)

        # Create dataset
        dataset = DigitDataset(data_dir, max_samples_per_class=max_samples_per_class)

        # Track results
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(10)}
        class_total = {i: 0 for i in range(10)}
        predictions = []

        # Evaluate
        with torch.no_grad():
            for image, true_label, img_path in tqdm(dataset, desc=f"{method} Evaluation"):
                # Apply enhancement method
                processed_tensor = preprocess_image_for_model(image, method)
                processed_tensor = processed_tensor.unsqueeze(0).to(device)

                # Get prediction
                outputs = model(processed_tensor)
                probs = torch.softmax(outputs, dim=-1)
                pred_label = int(probs.argmax(dim=-1).item())
                confidence = float(probs.max().item())

                # Track results
                total += 1
                if pred_label == true_label:
                    correct += 1
                    class_correct[true_label] += 1
                class_total[true_label] += 1

                predictions.append({
                    'image_path': img_path,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'method': method
                })

        # Calculate accuracy
        accuracy = correct / total if total > 0 else 0

        # Calculate per-class accuracy
        class_accuracies = {}
        for i in range(10):
            if class_total[i] > 0:
                class_accuracies[i] = class_correct[i] / class_total[i]
            else:
                class_accuracies[i] = 0.0

        results[method] = {
            'accuracy': accuracy,
            'total_samples': total,
            'correct_predictions': correct,
            'class_accuracies': class_accuracies,
            'predictions': predictions
        }

        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Samples per class: {max_samples_per_class}")
        print("   Per-class accuracies:")
        for i in range(10):
            count = class_total[i]
            acc = class_accuracies[i]
            print(f"     Digit {i}: {acc:.4f} ({count} samples)")
    return results


def create_accuracy_comparison(results):
    """Create a comparison report of accuracy results."""

    print(f"\n{'='*60}")
    print("📊 ACCURACY COMPARISON REPORT")
    print(f"{'='*60}")

    # Overall accuracy comparison
    print("\n🎯 Overall Accuracy:")
    for method, data in results.items():
        acc = data['accuracy']
        total = data['total_samples']
        correct = data['correct_predictions']
        print(f"   {method.upper()}: {acc:.4f} ({correct}/{total} correct)")
    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_method]['accuracy']
    print(f"\n🏆 Best method: {best_method.upper()} with {best_accuracy:.4f} accuracy")

    # Per-class comparison
    print("\n📈 Per-Class Accuracy Comparison:")
    print("Class | Real-ESRGAN | No-Sharpen | Traditional")
    print("------|------------|------------|------------")

    for i in range(10):
        realesrgan_acc = results['realesrgan']['class_accuracies'][i]
        no_sharp_acc = results['no_sharpening']['class_accuracies'][i]
        trad_acc = results['traditional']['class_accuracies'][i]

        # Highlight best for each class
        best_acc = max(realesrgan_acc, no_sharp_acc, trad_acc)
        realesrgan_str = f"{realesrgan_acc:.3f}"
        no_sharp_str = f"{no_sharp_acc:.3f}"
        trad_str = f"{trad_acc:.3f}"

        print(f"  {i}   | {realesrgan_str} | {no_sharp_str} | {trad_str}")

    # Save detailed results
    output_dir = os.path.join(BASE_DIR, "outputs", "accuracy_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary_data = []
    for method, data in results.items():
        summary_data.append({
            'method': method,
            'accuracy': data['accuracy'],
            'total_samples': data['total_samples'],
            'correct_predictions': data['correct_predictions']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "accuracy_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Save detailed predictions
    all_predictions = []
    for method, data in results.items():
        all_predictions.extend(data['predictions'])

    predictions_df = pd.DataFrame(all_predictions)
    predictions_path = os.path.join(output_dir, "detailed_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Summary: {summary_path}")
    print(f"   - Detailed predictions: {predictions_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate digit recognition accuracy with different enhancement methods")
    parser.add_argument("--model-path", type=str,
                       default=os.path.join(BASE_DIR, "outputs", "bbox_comparison", "digit_classifier.pth"),
                       help="Path to trained digit classifier model")
    parser.add_argument("--data-dir", type=str,
                       default=os.path.join(BASE_DIR, "data", "classification", "single_digits"),
                       help="Path to classification dataset")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="Maximum samples per digit class for testing")

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model not found: {args.model_path}")
        print("Please train a digit classifier first.")
        sys.exit(1)

    # Check if data exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Please run data preparation first.")
        sys.exit(1)

    # Run evaluation
    results = evaluate_model_on_enhancement_methods(
        args.model_path,
        args.data_dir,
        max_samples_per_class=args.max_samples
    )

    # Create comparison report
    create_accuracy_comparison(results)

    print(f"\n✅ Evaluation complete!")
    print("Use the results to determine which enhancement method works best for your use case.")


if __name__ == "__main__":
    main()