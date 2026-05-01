import os
import sys
import subprocess
import argparse

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Run enhancement benchmark for all available methods")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to run per method")
    args = parser.parse_args()

    methods = ["none", "unsharp_mask", "clahe", "esrgan"]
    script_path = os.path.join(BASE_DIR, "src", "evaluation", "eval_pipeline_for_enhancement.py")

    print("="*60)
    print("🚀 STARTING FULL ENHANCEMENT BENCHMARK")
    print("="*60)

    for method in methods:
        print(f"\n\n{'='*40}")
        print(f"▶️  RUNNING EVALUATION FOR METHOD: {method.upper()}")
        print(f"{'='*40}")
        
        cmd = [
            sys.executable, 
            script_path, 
            "--enhancement", method,
            "--max-samples", str(args.max_samples)
        ]
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Error occurred while running method: {method}")

if __name__ == "__main__":
    main()