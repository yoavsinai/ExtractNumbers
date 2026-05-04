import os
import sys
import subprocess
import argparse

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_comparison_summary(methods, reports_dir):
    """Read each method's summary file and print a side-by-side comparison table."""
    import re

    print("\n\n" + "="*70)
    print("📊 FINAL COMPARISON SUMMARY — ALL ENHANCEMENT METHODS")
    print("="*70)

    header = f"{'Method':<16} {'Seq Acc':>10} {'Digit Acc':>10} {'S1 IoU':>10} {'S3 IoU':>10}"
    print(header)
    print("-" * len(header))

    patterns = {
        "Seq Acc":   r"Full Sequence Accuracy:\s+([\d.]+%)",
        "Digit Acc": r"Mean Digit Accuracy \(Pos\):\s+([\d.]+%)",
        "S1 IoU":    r"Stage 1 \(Global\) Mean IoU:\s+([\d.]+)",
        "S3 IoU":    r"Stage 3 \(Indiv\)  Mean IoU:\s+([\d.]+)",
    }

    rows = []
    for method in methods:
        path = os.path.join(reports_dir, f"enhancement_summary_{method}.txt")
        if not os.path.exists(path):
            rows.append((method, "N/A", "N/A", "N/A", "N/A"))
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        vals = {}
        for key, pat in patterns.items():
            m = re.search(pat, text)
            vals[key] = m.group(1) if m else "N/A"
        rows.append((method, vals["Seq Acc"], vals["Digit Acc"], vals["S1 IoU"], vals["S3 IoU"]))

    for row in rows:
        print(f"{row[0]:<16} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")

    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Run enhancement benchmark for all available methods")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to run per method")
    args = parser.parse_args()

    methods = ["none", "unsharp_mask", "clahe", "esrgan", "edsr", "lapsrn", "realcugan", "bsrgan", "swiniR", "diffusion"]
    script_path = os.path.join(BASE_DIR, "src", "evaluation", "eval_pipeline_for_enhancement.py")
    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")

    print("="*60)
    print("STARTING FULL ENHANCEMENT BENCHMARK")
    print("="*60)

    for method in methods:
        print(f"\n\n{'='*40}")
        print(f"RUNNING EVALUATION FOR METHOD: {method.upper()}")
        print(f"{'='*40}")

        cmd = [
            sys.executable,
            script_path,
            "--enhancement", method,
            "--max-samples", str(args.max_samples)
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error occurred while running method: {method}")

    print_comparison_summary(methods, reports_dir)

if __name__ == "__main__":
    main()