import os
import subprocess
import sys
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# הגדרת נתיבים בסיסיים
BASE_DIR = r"C:\Users\user\OneDrive - Bar-Ilan University - Students\Bar Ilan\C\ExtractNumbers\ExtractNumbers"
SRC_DIR = os.path.join(BASE_DIR, "src", "Bounding Box")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bbox_comparison")
YOLO_RUN_DIR = os.path.join(OUTPUT_DIR, "yolo_runs", "run1")

def run_quiet_script(script_name, args=[]):
    """מריץ סקריפט פייתון ומדפיס רק אם יש שגיאה"""
    script_path = os.path.join(SRC_DIR, script_name)
    cmd = [sys.executable, script_path] + args
    
    # הוספנו encoding='utf-8' כדי לפתור את השגיאה של ה- UnicodeDecodeError
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"Error running {script_name}:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

def analyze_epochs(csv_path):
    """מנתח את קובץ ה-results.csv של YOLO כדי לראות שיפור בין אפוקים"""
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    map_col = 'metrics/mAP50(B)'
    
    print("\n--- Epoch-by-Epoch Improvement (mAP50) ---")
    map_vals = df[map_col].values
    for i in range(len(map_vals)):
        diff = map_vals[i] - map_vals[i-1] if i > 0 else 0
        trend = "▲" if diff > 0.005 else "≈"
        print(f"Epoch {i+1:02d}: {map_vals[i]:.4f} (Change: {diff:+.4f}) {trend}")
    
    # המלצה
    last_5_avg_diff = np.mean(np.diff(map_vals[-5:]))
    if last_5_avg_diff < 0.002:
        print("\nRecommendation: The model has plateaued. You can likely use 10-15 epochs.")
    else:
        print("\nRecommendation: The model is still improving. 20 epochs is appropriate.")



def preview_ground_truth():
    """מייצר תמונת תצוגה מקדימה של הלייבלים כדי לוודא שהם נחתכו נכון לפני האימון"""
    print("\nGenerating Ground Truth preview...")
    dataset_root = os.path.join(BASE_DIR, "data", "segmentation")
    preview_img_path = os.path.join(OUTPUT_DIR, "preview_labels_before_training.png")
    categories = ['natural', 'synthetic', 'handwritten']
    samples_per_cat = 2
    
    fig, axes = plt.subplots(len(categories), samples_per_cat, figsize=(10, 12))
    plt.subplots_adjust(hspace=0.4)
    
    for i, cat in enumerate(categories):
        cat_dir = os.path.join(dataset_root, cat)
        if not os.path.exists(cat_dir): continue
        
        folders = os.listdir(cat_dir)
        selected_folders = random.sample(folders, min(samples_per_cat, len(folders)))
        
        for j, folder in enumerate(selected_folders):
            img_path = os.path.join(cat_dir, folder, "image.jpg")
            mask_path = os.path.join(cat_dir, folder, "mask.png")
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path): continue
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            ax = axes[i, j]
            ax.imshow(img)
            
            # הלוגיקה המדויקת שלפיה המודל מייצר לייבלים
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h > 10:
                        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=3))
            
            ax.set_title(f"Preview: {cat}")
            ax.axis('off')
            
    plt.suptitle("SANITY CHECK: Ground Truth Labels Before Training", fontsize=16)
    plt.savefig(preview_img_path, bbox_inches='tight')
    plt.close()
    return preview_img_path



def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # -- תצוגה מקדימה ובדיקת שפיות --
    preview_path = preview_ground_truth()
    print(f"\n=> SANITY CHECK: Please open the image at:\n   {preview_path}")
    print("Check if the green bounding boxes surround individual digits correctly.")
    user_input = input("Press ENTER to continue with YOLO training, or type 'q' and Enter to abort: ")
    
    # ---------------------------------
    # 1. הרצת האימון והחיזוי של YOLO
    print("Step 1/3: Running YOLO Training & Inference...")
    run_quiet_script("yolo_detector.py", [
        "--dataset-root", os.path.join(BASE_DIR, "data", "segmentation"),
        "--output-dir", OUTPUT_DIR,
        "--epochs", "20",
        "--overwrite-conversion"
    ])

    # 2. הצגת סטטיסטיקות סופיות מהטרמינל (מתוך קובץ ה-results של YOLO)
    results_csv = os.path.join(YOLO_RUN_DIR, "results.csv")
    if os.path.exists(results_csv):
        rdf = pd.read_csv(results_csv)
        rdf.columns = rdf.columns.str.strip()
        last = rdf.iloc[-1]
        print("\n=== YOLO Final Performance Summary ===")
        print(f"Overall mAP50:  {last['metrics/mAP50(B)']:.2%}")
        print(f"Precision:      {last['metrics/precision(B)']:.2%}")
        print(f"Recall:         {last['metrics/recall(B)']:.2%}")
        
        analyze_epochs(results_csv)

    # 3. ניתוח לפי קטגוריות (מתוך קובץ הניבויים)
    pred_csv = os.path.join(OUTPUT_DIR, "yolo_predictions.csv")
    if os.path.exists(pred_csv):
        pdf = pd.read_csv(pred_csv)
        print("\n=== Accuracy per Category (Average Confidence) ===")
        stats = pdf.groupby('category')['pred_conf'].agg(['mean', 'count'])
        for cat, row in stats.iterrows():
            print(f"- {cat:12}: {row['mean']:.2%} (Total samples: {int(row['count'])})")

    # 4. הרצת ויזואליזציה
    print("\nStep 3/3: Generating Comparison Images...")
    run_quiet_script("visualize_yolo_results.py")
    print(f"Process Complete. Check: {OUTPUT_DIR}\yolo_comparison_summary.png")

if __name__ == "__main__":
    main()