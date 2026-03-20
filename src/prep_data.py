import argparse
import os
import shutil
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

def _safe_rmtree(path, base_dir, label):
    """Remove *path* only when it is a real directory inside *base_dir*."""
    real_path = os.path.realpath(path)
    real_base = os.path.realpath(base_dir)
    if os.path.islink(path):
        print(f"Skipping {label}: '{path}' is a symlink – will not remove.")
        return
    rel = os.path.relpath(real_path, real_base)
    if rel == "." or rel.startswith(".."):
        print(f"Skipping {label}: '{path}' is outside the project root – will not remove.")
        return
    if not os.path.exists(path):
        print(f"Skipping {label}: '{path}' does not exist – nothing to remove.")
        return
    if not os.path.isdir(path):
        print(f"Skipping {label}: '{path}' is not a directory – will not remove.")
        return
    print(f"Removing {label} directory: {path}")
    shutil.rmtree(path)

def main():
    parser = argparse.ArgumentParser(description="Automated data preparation pipeline.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing output directories (data/classification and data/segmentation) before running.",
    )
    args = parser.parse_args()

    print("=== Automated Data Preparation Pipeline ===\n")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_data_dir = os.path.join(base_dir, "src", "data")
    
    # Only clean previous output when the user explicitly requests it
    if args.clean:
        processed_dir = os.path.join(base_dir, "data", "classification")
        seg_dir = os.path.join(base_dir, "data", "segmentation")
        confirm = input(
            "Are you sure you want to delete existing output directories? [y/N] "
        ).strip().lower()
        if confirm == "y":
            if os.path.exists(processed_dir):
                _safe_rmtree(processed_dir, base_dir, "classification")
            if os.path.exists(seg_dir):
                _safe_rmtree(seg_dir, base_dir, "segmentation")
        else:
            print("Clean step skipped.")

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
