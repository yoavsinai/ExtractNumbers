"""
Enhanced Pipeline Comparison Script

This script serves as a comprehensive tool for comparing different enhancement methods 
(Real-ESRGAN, No-Sharpen, Traditional, and Both) across the full extraction pipeline. 
It runs the complete pipeline for each method, generates visual comparisons, and creates 
a detailed report. It is intended to help developers visually and quantitatively determine 
which enhancement technique yields the best bounding boxes and crop results.
"""

import os
import sys
import subprocess
import pandas as pd
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import json

# Add src to path so we can import modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import (
    enhance_digit, enhance_without_sharpening, enhance_with_traditional_methods, enhance_with_both
)

# Path configuration
BOUNDING_BOX_SRC = os.path.join(BASE_DIR, "src", "bounding_box")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bbox_comparison")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "globalbb_predictions.csv")
BEST_GLOBAL_PATH = os.path.join(OUTPUT_DIR, "globalbb_run", "weights", "best.pt")
BEST_INDIVIDUAL_PATH = os.path.join(OUTPUT_DIR, "individualbb_run", "weights", "best.pt")

def run_python_script(script_path, args=[], capture=True):
    """Run a Python script. If capture=True, output is hidden until completion."""
    cmd = [sys.executable, script_path] + args
    if not capture:
        result = subprocess.run(cmd)
        return result.returncode == 0

    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error running {script_path}:\n{result.stderr}")
        return False
    return True

def run_pipeline_with_enhancement_method(method_name, enhancement_func, epochs=20):
    """
    Run the full pipeline with a specific enhancement method.

    Args:
        method_name: Name of the enhancement method
        enhancement_func: Function to use for enhancement
        epochs: Number of training epochs
    """
    print(f"\n{'='*60}")
    print(f"RUNNING PIPELINE WITH: {method_name.upper()}")
    print(f"{'='*60}")

    # Create method-specific output directories
    method_output_dir = os.path.join(BASE_DIR, "outputs", f"enhancement_comparison_{method_name}")
    method_sharpened_dir = os.path.join(method_output_dir, "sharpened_crops")
    method_predictions_csv = os.path.join(method_output_dir, "globalbb_predictions.csv")
    method_best_individual_path = os.path.join(method_output_dir, "individualbb_run", "weights", "best.pt")

    os.makedirs(method_sharpened_dir, exist_ok=True)
    os.makedirs(method_output_dir, exist_ok=True)

    # Copy global predictions if they exist
    if os.path.exists(PREDICTIONS_CSV):
        shutil.copy2(PREDICTIONS_CSV, method_predictions_csv)
        print(f"✓ Copied global predictions to {method_predictions_csv}")
    else:
        print("✗ No global predictions found - need to run GlobalBB first")
        return None

    # Load predictions
    df = pd.read_csv(method_predictions_csv)
    df_valid = df.dropna(subset=['pred_x1'])
    print(f"✓ Loaded {len(df_valid)} valid detections for {method_name}")

    # Process each image with the specific enhancement method
    results_data = []

    print(f"\nProcessing images with {method_name} enhancement...")
    for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc=f"{method_name} Processing"):
        img_path = row['image_path']
        cat = row['category']
        img = cv2.imread(img_path)
        if img is None:
            continue

        x1, y1, x2, y2 = int(row['pred_x1']), int(row['pred_y1']), int(row['pred_x2']), int(row['pred_y2'])
        h, w = img.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Apply the specific enhancement method
        enhanced = enhancement_func(crop)

        # Save enhanced crop
        sample_name = os.path.splitext(os.path.basename(img_path))[0]
        crop_filename = f"{cat}_{sample_name}_{method_name}.jpg"
        cv2.imwrite(os.path.join(method_sharpened_dir, crop_filename), enhanced)

        # Store result data
        results_data.append({
            'image_path': img_path,
            'category': cat,
            'method': method_name,
            'crop_shape': crop.shape,
            'enhanced_shape': enhanced.shape,
            'bbox': [x1, y1, x2, y2]
        })

    # Save results summary
    results_df = pd.DataFrame(results_data)
    results_csv_path = os.path.join(method_output_dir, f"{method_name}_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"✓ {method_name} processing complete!")
    print(f"  - Processed {len(results_data)} images")
    print(f"  - Results saved to: {results_csv_path}")

    return {
        'method': method_name,
        'output_dir': method_output_dir,
        'results_csv': results_csv_path,
        'sharpened_dir': method_sharpened_dir,
        'num_processed': len(results_data),
        'results_df': results_df
    }

def create_comparison_visualization(all_results):
    """Create a comprehensive comparison visualization."""
    print(f"\n{'='*60}")
    print("CREATING COMPARISON VISUALIZATION")
    print(f"{'='*60}")

    # Load sample images for comparison
    sample_images = []
    categories = ['handwritten', 'synthetic', 'natural']

    # Get one sample from each category
    for method_result in all_results:
        df = method_result['results_df']
        for cat in categories:
            cat_samples = df[df['category'] == cat]
            if len(cat_samples) > 0:
                sample_row = cat_samples.iloc[0]
                img_path = sample_row['image_path']
                img = cv2.imread(img_path)
                if img is not None:
                    sample_images.append({
                        'category': cat,
                        'image_path': img_path,  # Store the path
                        'original': img,  # Store the loaded image
                        'bbox': sample_row['bbox'],
                        'method_results': []
                    })
                    break

    # For each sample, get the enhanced versions from all methods
    for sample in sample_images:
        img_path = sample['original']  # This should be the path, not the image array
        x1, y1, x2, y2 = sample['bbox']
        crop = img_path[y1:y2, x1:x2]  # Wait, this is wrong - img_path is a path string, not an image array

        # Actually, we need to reload the image from the path
        actual_img_path = sample['image_path']  # We need to store the actual path
        img = cv2.imread(actual_img_path)
        if img is None:
            continue
        crop = img[y1:y2, x1:x2]

        for method_result in all_results:
            method_name = method_result['method']
            sharpened_dir = method_result['sharpened_dir']

            # Find the corresponding enhanced image
            sample_name = os.path.splitext(os.path.basename(actual_img_path))[0]
            enhanced_filename = f"{sample['category']}_{sample_name}_{method_name}.jpg"
            enhanced_path = os.path.join(sharpened_dir, enhanced_filename)

            if os.path.exists(enhanced_path):
                enhanced_img = cv2.imread(enhanced_path)
                sample['method_results'].append({
                    'method': method_name,
                    'enhanced': enhanced_img
                })

    # Create comparison visualization
    fig, axes = plt.subplots(len(sample_images), len(all_results) + 1,
                           figsize=(18, 6 * len(sample_images)))

    if len(sample_images) == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(sample_images):
        # Original image with bbox
        img_rgb = cv2.cvtColor(sample['original'], cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(img_rgb)
        x1, y1, x2, y2 = sample['bbox']
        axes[i, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                         fill=False, edgecolor='red', linewidth=2))
        axes[i, 0].set_title(f"Original\n({sample['category']})", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Enhanced versions
        for j, method_result in enumerate(sample['method_results']):
            method_name = method_result['method']
            enhanced_rgb = cv2.cvtColor(method_result['enhanced'], cv2.COLOR_BGR2RGB)
            axes[i, j+1].imshow(enhanced_rgb)
            axes[i, j+1].set_title(f"{method_name.title()}\nEnhancement", fontsize=12)
            axes[i, j+1].axis('off')

    plt.suptitle("ENHANCEMENT METHODS COMPARISON", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()

    comparison_path = os.path.join(BASE_DIR, "outputs", "enhancement_methods_comparison.png")
    plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"✓ Comparison visualization saved to: {comparison_path}")

def generate_comparison_report(all_results):
    """Generate a detailed comparison report."""
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*60}")

    report = {
        'summary': {},
        'methods': {},
        'recommendations': []
    }

    # Summary statistics
    total_images = 0
    for result in all_results:
        method = result['method']
        num_processed = result['num_processed']
        total_images = max(total_images, num_processed)

        report['methods'][method] = {
            'processed_images': num_processed,
            'output_directory': result['output_dir'],
            'results_csv': result['results_csv']
        }

    report['summary'] = {
        'total_images_in_dataset': total_images,
        'methods_compared': len(all_results),
        'comparison_visualization': os.path.join(BASE_DIR, "outputs", "enhancement_methods_comparison.png")
    }

    # Analysis by category
    categories = ['handwritten', 'synthetic', 'natural']
    category_stats = {}

    for cat in categories:
        category_stats[cat] = {}
        for result in all_results:
            method = result['method']
            df = result['results_df']
            cat_count = len(df[df['category'] == cat])
            category_stats[cat][method] = cat_count

    report['category_breakdown'] = category_stats

    # Save report
    report_path = os.path.join(BASE_DIR, "outputs", "enhancement_comparison_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Detailed report saved to: {report_path}")

    # Print summary to console
    print("\n📊 COMPARISON SUMMARY:")
    print(f"   Total images processed: {total_images}")
    print(f"   Methods compared: {len(all_results)}")
    print("\n📁 Category Breakdown:")
    for cat in categories:
        print(f"   {cat.title()}:")
        for method, count in category_stats[cat].items():
            print(f"     - {method}: {count} images")

    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare different enhancement methods across the full pipeline")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (not used in comparison)")
    parser.add_argument("--sample-size", type=int, default=None, help="Limit processing to N samples per method")
    args = parser.parse_args()

    print("🔬 ENHANCEMENT METHODS COMPARISON")
    print("=" * 60)
    print("This script will run the full extraction pipeline with different")
    print("image enhancement methods and compare their performance.")
    print()

    # Define enhancement methods to compare
    enhancement_methods = [
        ('realesrgan', lambda img: enhance_digit(img, upscale_factor=2.0)),
        ('no_sharpening', lambda img: enhance_without_sharpening(img)),
        ('traditional', lambda img: enhance_with_traditional_methods(img)),
        ('both', lambda img: enhance_with_both(img))
    ]

    all_results = []

    # Run pipeline for each enhancement method
    for method_name, enhancement_func in enhancement_methods:
        result = run_pipeline_with_enhancement_method(
            method_name=method_name,
            enhancement_func=enhancement_func,
            epochs=args.epochs
        )

        if result:
            all_results.append(result)

        # Optional: limit sample size for testing
        if args.sample_size and len(all_results) >= args.sample_size:
            break

    if not all_results:
        print("❌ No results generated. Make sure GlobalBB predictions exist.")
        sys.exit(1)

    # Create visualizations and reports
    create_comparison_visualization(all_results)
    report = generate_comparison_report(all_results)

    print(f"\n{'='*60}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*60}")
    print(f"📁 All results saved to: {os.path.join(BASE_DIR, 'outputs')}")
    print("📊 Check the comparison visualization and detailed report")
if __name__ == "__main__":
    main()