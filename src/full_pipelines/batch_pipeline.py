import os
import sys
import subprocess
import pandas as pd

# Base path configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "bbox_comparison")

def run_script(script_path, args=[]):
    """Run a Python script and capture output."""
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error running {script_path}:\n{result.stderr}")
        sys.exit(1)
    return result.stdout

def extract_labeled_numbers():
    """Extract labeled numbers from digit predictions CSV."""
    digit_csv = os.path.join(OUTPUT_DIR, "digit_predictions.csv")
    if not os.path.exists(digit_csv):
        print(f"Digit predictions file not found: {digit_csv}")
        return

    df = pd.read_csv(digit_csv)
    grouped = df.groupby("image_path")

    print("\n--- Labeled Numbers ---")
    for image_path, group in grouped:
        # Sort by pred_x1 (left to right)
        sorted_group = group.sort_values("pred_x1")
        digits = sorted_group["digit"].tolist()
        number_str = "".join(map(str, digits))
        print(f"{image_path}: {number_str}")

def main():
    print("Starting Full Pipeline: Image to Labeled Numbers")

    # Step 1: Run YOLO detection to get bounding boxes
    print("Step 1: Running YOLO Bounding Box Detection...")
    run_script(os.path.join(BASE_DIR, "src", "bounding_box", "yolo_detector.py"), ["--skip-train"])  # Assuming weights exist, skip training

    # Step 2: Run Digit Recognition on the detected bounding boxes
    print("Step 2: Running Digit Recognition...")
    run_script(os.path.join(BASE_DIR, "src", "digit_recognizer", "digit_recognizer.py"))

    # Step 3: Visualize Results
    print("Step 3: Visualizing Results...")
    run_script(os.path.join(BASE_DIR, "src", "bounding_box", "visualize_yolo_results.py"))

    # Step 4: Extract and Print Labeled Numbers
    print("Step 4: Extracting Labeled Numbers...")
    extract_labeled_numbers()

    print("Full Pipeline Complete. Check outputs for results.")

if __name__ == "__main__":
    main()