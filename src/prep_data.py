import argparse
import os
import shutil
import sys
import subprocess
from dotenv import load_dotenv

# Ensure we can import the data loaders from the local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import svhn, race_numbers, handwritten, ocr_trains, download_icdar_svt, download_dstext_v2, download_bovtext, download_moving_mnist, download_roadtext

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
        return
    if not os.path.isdir(path):
        return
    print(f"Cleaning up {label} directory...")
    shutil.rmtree(path)

def main():
    parser = argparse.ArgumentParser(description="Automated data preparation pipeline.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing processed data before running.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per dataset (applies to all datasets except where a specific limit is set).")
    parser.add_argument("--handwritten-limit", type=int, default=None, help="Override sample limit specifically for the synthetic handwritten dataset (default: 10000). Decoupled from --limit so SVHN is not affected.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["svhn", "race_numbers", "handwritten", "ocr_trains", "icdar_svt", "dstext_v2", "bovtext", "moving_mnist", "roadtext"],
        help="Datasets to process.",
    )
    parser.add_argument("--no-augment", action="store_true", help="Skip the high-level augmentation phase.")
    args, unknown = parser.parse_known_args()

    # Load environment variables for Kaggle authentication
    load_dotenv()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(base_dir, "data", "digits_data")
    
    if args.clean:
        _safe_rmtree(output_dir, base_dir, "digits_data")

    os.makedirs(output_dir, exist_ok=True)

    if "svhn" in args.datasets:
        svhn.prepare(output_dir, limit=args.limit)

    if "race_numbers" in args.datasets:
        race_numbers.prepare(output_dir, limit=args.limit)

    if "handwritten" in args.datasets:
        # Use --handwritten-limit if provided, else fall back to --limit.
        # This decouples synthetic generation from the SVHN (and other real) dataset limits.
        hw_limit = args.handwritten_limit if args.handwritten_limit is not None else args.limit
        handwritten.prepare(output_dir, limit=hw_limit)



    if "ocr_trains" in args.datasets:
        ocr_trains.prepare(output_dir, limit=args.limit)

    if "icdar_svt" in args.datasets:
        download_icdar_svt.main()

    if "dstext_v2" in args.datasets:
        download_dstext_v2.main()

    if "bovtext" in args.datasets:
        download_bovtext.main()

    if "moving_mnist" in args.datasets:
        download_moving_mnist.main()

    if "roadtext" in args.datasets:
        download_roadtext.main()


    if not args.no_augment:
        print("\n--- PHASE 2: Applying High-Level Augmentations ---")
        aug_script = os.path.join(base_dir, "src", "data", "apply_augmentations.py")
        subprocess.run([sys.executable, aug_script])
    else:
        print("\n--- PHASE 2 Skipped (Augmentations Disabled) ---")

    print("\n--- PHASE 3: Converting Video Datasets ---")
    video_conversion_script = os.path.join(base_dir, "src", "data", "convert_to_video_data.py")
    if os.path.exists(video_conversion_script):
        result = subprocess.run(
            [sys.executable, video_conversion_script],
            capture_output=False,  # Show output in real-time
            text=True
        )
        if result.returncode != 0:
            print("\n[!] Warning: Video conversion script exited with non-zero status")
    else:
        print(f"[!] Video conversion script not found: {video_conversion_script}")

    print("\n=== All Data Datasets Successfully Fetched & Built! ===")

if __name__ == "__main__":
    main()
