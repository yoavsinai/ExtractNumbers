"""
run_all_postenhancement.py
--------------------------
Runs eval_postenhancement_pipeline.py for all 10 enhancement methods and
prints a side-by-side comparison table.

Enhancement is applied AFTER the IndividualBB detection (on each digit crop).
"""
import os
import sys
import subprocess
import argparse
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_comparison_summary(methods, reports_dir):
    """Read each method's post-enhancement summary file and print a comparison table."""
    print("\n\n" + "="*80)
    print("📊 FINAL COMPARISON — POST-ENHANCEMENT (Digit Crop Enhanced Before Classification)")
    print("="*80)

    header = f"{'Method':<16} {'Seq Acc':>10} {'Digit Acc':>10} {'S1 IoU':>8} {'S3 IoU':>8} {'Succ Rate':>10}"
    print(header)
    print("-" * len(header))

    patterns = {
        "Seq Acc":   r"Full Sequence Accuracy:\s+([\d.]+%)",
        "Digit Acc": r"Mean Digit Accuracy \(Pos\):\s+([\d.]+%)",
        "S1 IoU":    r"Stage 1 \(Global\) Mean IoU:\s+([\d.]+)",
        "S3 IoU":    r"Stage 3 \(Indiv\)  Mean IoU:\s+([\d.]+)",
        "Succ Rate": r"Succession Rate:\s+([\d.]+%)",
    }

    rows = []
    for method in methods:
        path = os.path.join(reports_dir, f"postenhancement_summary_{method}.txt")
        if not os.path.exists(path):
            rows.append((method, "N/A", "N/A", "N/A", "N/A", "N/A"))
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        vals = {}
        for key, pat in patterns.items():
            m = re.search(pat, text)
            vals[key] = m.group(1) if m else "N/A"
        rows.append((
            method,
            vals["Seq Acc"], vals["Digit Acc"],
            vals["S1 IoU"], vals["S3 IoU"],
            vals["Succ Rate"]
        ))

    for row in rows:
        print(f"{row[0]:<16} {row[1]:>10} {row[2]:>10} {row[3]:>8} {row[4]:>8} {row[5]:>10}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run post-enhancement benchmark for all available methods"
    )
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples per method (split evenly across categories)")
    args = parser.parse_args()

    methods = ["none", "unsharp_mask", "clahe", "esrgan", "edsr",
               "lapsrn", "realcugan", "bsrgan", "swiniR", "diffusion"]
    script_path = os.path.join(BASE_DIR, "src", "evaluation", "eval_postenhancement_pipeline.py")
    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")

    print("="*70)
    print("STARTING FULL POST-ENHANCEMENT BENCHMARK (Individual Digit Crop → Enhance)")
    print("="*70)

    for method in methods:
        print(f"\n\n{'='*45}")
        print(f"RUNNING POST-ENHANCEMENT FOR METHOD: {method.upper()}")
        print(f"{'='*45}")

        cmd = [
            sys.executable, script_path,
            "--enhancement", method,
            "--max-samples", str(args.max_samples),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"⚠️  Error occurred while running method: {method}")

    print_comparison_summary(methods, reports_dir)


if __name__ == "__main__":
    main()
