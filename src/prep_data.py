import os
import sys
import subprocess

def run_script(script_path):
    print(f"[{script_path}] Starting...")
    # Execute the python scripts in standard order
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"[{script_path}] ERROR! Exited with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[{script_path}] Finished successfully.\n")

def main():
    print("=== Automated Data Preparation Pipeline ===\n")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_data_dir = os.path.join(base_dir, "src", "data")
    
    # Clean previous arrays if running full pipeline
    processed_dir = os.path.join(base_dir, "data", "classification")
    seg_dir = os.path.join(base_dir, "data", "segmentation")
    if os.path.exists(processed_dir):
        print("Cleaning old classification directories...")
        import shutil
        shutil.rmtree(processed_dir)
    if os.path.exists(seg_dir):
        print("Cleaning old segmentation directories...")
        import shutil
        shutil.rmtree(seg_dir)
        
    print("\n--- PHASE 1: Fetching core classification sets ---")
    run_script(os.path.join(src_data_dir, "download_datasets.py"))
    
    print("--- PHASE 2: Compiling Classification Images ---")
    run_script(os.path.join(src_data_dir, "process_dataset.py"))
    
    print("--- PHASE 3: Downloading & Parsing SVHN Segmentation ---")
    run_script(os.path.join(src_data_dir, "process_svhn_seg.py"))
    
    print("--- PHASE 4: Generating Synthetic Segmentation Arrays ---")
    run_script(os.path.join(src_data_dir, "create_synthetic_seg.py"))

    print("\n=== All Data Datasets Successfully Fetched & Built! ===")

if __name__ == "__main__":
    main()
