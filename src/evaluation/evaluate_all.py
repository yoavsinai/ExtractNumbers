import os
import sys
import subprocess

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_script(script_name, args=[]):
    script_path = os.path.join(BASE_DIR, "src", "evaluation", script_name)
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script_path] + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"❌ Error running {script_name}")
    return result.returncode == 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all evaluation stages")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples for each stage")
    args = parser.parse_args()

    stages = [
        ("eval_global_bbox.py", ["--max-samples", str(args.max_samples)]),
        ("eval_sharpening.py", ["--max-samples", str(args.max_samples)]),
        ("eval_individual_bbox.py", ["--max-samples", str(args.max_samples)]),
        ("eval_digit_recog.py", ["--max-samples", str(args.max_samples)]),
        ("eval_pipeline.py", ["--max-samples", str(args.max_samples), "--save-viz", "--analyze-errors"])
    ]
    
    success_count = 0
    for script, script_args in stages:
        if run_script(script, script_args):
            success_count += 1
            
    print(f"\n{'='*60}")
    print(f"✅ EVALUATION COMPLETE: {success_count}/{len(stages)} stages succeeded")
    print(f"{'='*60}")
    print(f"Reports are available in: {os.path.join(BASE_DIR, 'outputs', 'reports')}")

if __name__ == "__main__":
    main()
