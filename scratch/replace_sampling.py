import re
import os
import glob

files = [
    "src/evaluation/eval_global_bbox.py",
    "src/evaluation/eval_postenhancement_pipeline.py",
    "src/evaluation/eval_pipeline_for_enhancement.py",
    "src/evaluation/eval_preenhancement_pipeline.py",
    "src/evaluation/eval_pipeline.py",
]

for fpath in files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r") as f:
        content = f.read()

    # Look for the pattern
    #    if samples_by_cat:
    #        samples_per_cat = args.max_samples // len(samples_by_cat)
    #        for cat, samps in samples_by_cat.items():
    #            random.shuffle(samps)
    #            eval_samples.extend(samps[:samples_per_cat])

    new_content = re.sub(
        r'(\w+)\s*=\s*args\.max_samples\s*//\s*len\(samples_by_cat\)\s*\n\s*for\s+(\w+),\s+(\w+)\s+in\s+samples_by_cat\.items\(\):\s*\n\s*random\.shuffle\(\3\)\s*\n\s*eval_samples\.extend\(\3\[:\1\]\)',
        r'total_samples = sum(len(s) for s in samples_by_cat.values())\n        for \2, \3 in samples_by_cat.items():\n            random.shuffle(\3)\n            \1 = int(args.max_samples * (len(\3) / total_samples))\n            eval_samples.extend(\3[:\1])',
        content
    )

    if new_content != content:
        with open(fpath, "w") as f:
            f.write(new_content)
        print(f"Updated {fpath}")
    else:
        print(f"No match in {fpath}")

