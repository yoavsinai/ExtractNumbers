import os
import sys
import cv2
import time
import pandas as pd
from tqdm import tqdm

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, "src"))

from image_preprocessing.digit_preprocessor import enhance_digit
from utils.data_utils import iter_new_samples, get_gt_from_anno

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2: Sharpening/Enhancement Evaluation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--save-crops", action="store_true", help="Save sharpened crops for inspection")
    args = parser.parse_args()

    # Paths
    DATA_ROOT = os.path.join(BASE_DIR, "data", "digits_data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "evaluation_sharpening")
    REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("\n--- Stage 2: Sharpening/Enhancement Evaluation ---")
    
    samples = iter_new_samples(DATA_ROOT)
    import random
    random.seed(42)
    random.shuffle(samples)
    eval_samples = samples[:args.max_samples]
    
    results = []
    
    for s in tqdm(eval_samples, desc="Processing Sharpening"):
        img = cv2.imread(s['image_path'])
        if img is None: continue
        
        # Use Ground Truth Global BB for sharpening evaluation to isolate the stage
        gt_global_boxes, _, _, _ = get_gt_from_anno(s['anno_path'])
        if not gt_global_boxes: continue
        
        gx1, gy1, gx2, gy2 = map(int, gt_global_boxes[0])
        h, w = img.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        crop = img[gy1:gy2, gx1:gx2]
        
        if crop.size == 0: continue
        
        start_time = time.time()
        sharp = enhance_digit(crop, upscale_factor=2.0)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if args.save_crops:
            sample_name = os.path.splitext(os.path.basename(s['image_path']))[0]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{sample_name}_original.jpg"), crop)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{sample_name}_sharpened.jpg"), sharp)
        
        results.append({
            'sample_id': s['sample_id'],
            'category': s['category'],
            'orig_w': crop.shape[1],
            'orig_h': crop.shape[0],
            'sharp_w': sharp.shape[1],
            'sharp_h': sharp.shape[0],
            'duration_sec': duration,
            'pixels_processed': crop.shape[0] * crop.shape[1]
        })

    df = pd.DataFrame(results)
    
    # Global Metrics
    avg_duration = df['duration_sec'].mean()
    total_pixels = df['pixels_processed'].sum()
    throughput = total_pixels / df['duration_sec'].sum() if df['duration_sec'].sum() > 0 else 0
    
    print("\n" + "="*40)
    print("📊 STAGE 2: SHARPENING METRICS")
    print("="*40)
    print(f"Total Processed:    {len(df)}")
    print(f"Average Duration:   {avg_duration:.4f}s")
    print(f"Throughput:         {throughput:,.0f} pixels/sec")
    print(f"Avg Upscale Factor: {df['sharp_w'].mean() / df['orig_w'].mean():.2f}x")
    
    print("\n📈 PERFORMANCE BY CATEGORY:")
    cat_metrics = df.groupby('category').agg({
        'duration_sec': ['mean', 'std'],
        'pixels_processed': 'mean'
    })
    print(cat_metrics)

    # Save report
    csv_path = os.path.join(REPORTS_DIR, "stage2_sharpening_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Detailed report saved to: {csv_path}")
    if args.save_crops:
        print(f"🖼️  Sharpened crops saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
