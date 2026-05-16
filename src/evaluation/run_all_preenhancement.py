"""
run_all_preenhancement.py
--------------------------
Runs eval_preenhancement_pipeline.py for all 10 enhancement methods and
prints a side-by-side comparison table.

Enhancement is applied on the FULL IMAGE before GlobalBB detection.
Compare results against run_all_enhancements.py (which enhances the crop).
"""
import os
import sys
import subprocess
import argparse
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_comparison_summary(methods, reports_dir):
    """Read each method's pre-enhancement summary file and print a comparison table."""
    print("\n\n" + "="*80)
    print("📊 FINAL COMPARISON — PRE-ENHANCEMENT (Full Image Enhanced Before GlobalBB)")
    print("="*80)

    header = f"{'Method':<16} {'Seq Acc':>10} {'Digit Acc':>10} {'S1 IoU':>8} {'S3 IoU':>8} {'Succ Rate':>10} {'ms/img':>8}"
    print(header)
    print("-" * len(header))

    patterns = {
        "Seq Acc":   r"Full Sequence Accuracy:\s+([\d.]+%)",
        "Digit Acc": r"Mean Digit Accuracy \(Pos\):\s+([\d.]+%)",
        "S1 IoU":    r"Stage 1 \(Global\) Mean IoU:\s+([\d.]+)",
        "S3 IoU":    r"Stage 3 \(Indiv\)  Mean IoU:\s+([\d.]+)",
        "Succ Rate": r"Succession Rate:\s+([\d.]+%)",
        "ms/img":    r"Avg Time per Sample:\s+([\d.]+) ms",
    }

    rows = []
    for method in methods:
        path = os.path.join(reports_dir, f"pre_enhancement_summary_{method}.txt")
        if not os.path.exists(path):
            rows.append((method, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"))
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
            vals["Succ Rate"], vals["ms/img"],
        ))

    for row in rows:
        print(f"{row[0]:<16} {row[1]:>10} {row[2]:>10} {row[3]:>8} {row[4]:>8} {row[5]:>10} {row[6]:>8}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run pre-enhancement benchmark for all available methods"
    )
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Max samples per method (split evenly across categories)")
    args = parser.parse_args()

    methods = ["none", "unsharp_mask", "clahe", "esrgan", "edsr",
               "lapsrn", "realcugan", "bsrgan", "swiniR", "diffusion"]
    script_path = os.path.join(BASE_DIR, "src", "evaluation", "eval_preenhancement_pipeline.py")
    reports_dir = os.path.join(BASE_DIR, "outputs", "reports")

    print("="*70)
    print("STARTING FULL PRE-ENHANCEMENT BENCHMARK (Full Image → GlobalBB)")
    print("="*70)

    for method in methods:
        print(f"\n\n{'='*45}")
        print(f"RUNNING PRE-ENHANCEMENT FOR METHOD: {method.upper()}")
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
