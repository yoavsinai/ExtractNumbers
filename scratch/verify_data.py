import os
import json
import cv2
import pandas as pd
from pathlib import Path

def verify_datasets(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"Error: {data_root} does not exist.")
        return

    datasets = [d for d in data_root.iterdir() if d.is_dir()]
    print(f"Found {len(datasets)} datasets in {data_root}")
    
    summary = []
    for ds_path in datasets:
        samples = [s for s in ds_path.iterdir() if s.is_dir() and s.name.startswith("sample_")]
        num_samples = len(samples)
        
        if num_samples == 0:
            print(f"Dataset {ds_path.name} has 0 samples.")
            continue
            
        # Check first sample
        first_sample = samples[0]
        img_path = first_sample / "original.png"
        anno_path = first_sample / "annotations.json"
        
        img_size = "N/A"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img_size = f"{img.shape[1]}x{img.shape[0]}"
        
        label = "N/A"
        if anno_path.exists():
            with open(anno_path, 'r') as f:
                anno = json.load(f)
                if anno.get('detected_numbers'):
                    label = anno['detected_numbers'][0].get('full_value', 'N/A')
        
        summary.append({
            "Dataset": ds_path.name,
            "Samples": num_samples,
            "Example Size": img_size,
            "Example Label": label
        })
        
    df = pd.DataFrame(summary)
    print("\nDataset Summary:")
    print(df.to_string(index=False))
    
    # Check for potential clashing (duplicate sample paths across all datasets)
    all_sample_dirs = []
    for root, dirs, files in os.walk(data_root):
        if "annotations.json" in files:
            all_sample_dirs.append(root)
            
    print(f"\nTotal sample directories found: {len(all_sample_dirs)}")
    
    # Check for duplicate base names if they were at the same level (legacy check)
    base_names = [os.path.basename(d) for d in all_sample_dirs]
    if len(base_names) != len(set(base_names)):
        print("Warning: Some sample directories share the same base name (but are in different subfolders).")

if __name__ == "__main__":
    verify_datasets("data/digits_data")
