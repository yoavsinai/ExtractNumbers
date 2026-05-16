import re
import os

files_to_replace = {
    "src/evaluation/eval_individual_bbox.py": (
        r"filtered_samples = \[s for s in all_samples if s\['category'\] not in excluded_categories\]\s*\n\s*random\.shuffle\(filtered_samples\)\s*\n\s*eval_samples = filtered_samples\[:args\.max_samples\]",
        r"""from collections import defaultdict
    samples_by_cat = defaultdict(list)
    for s in all_samples:
        if s['category'] not in excluded_categories:
            samples_by_cat[s['category']].append(s)
            
    eval_samples = []
    if samples_by_cat:
        total_samples = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            num_samples = int(args.max_samples * (len(samps) / total_samples))
            eval_samples.extend(samps[:num_samples])
        random.shuffle(eval_samples)"""
    ),
    "src/evaluation/eval_digit_recog.py": (
        r"filtered_samples = \[s for s in all_samples if s\['category'\] not in excluded_categories\]\s*\n\s*random\.shuffle\(filtered_samples\)\s*\n\s*eval_samples = filtered_samples\[:args\.max_samples\]",
        r"""from collections import defaultdict
    samples_by_cat = defaultdict(list)
    for s in all_samples:
        if s['category'] not in excluded_categories:
            samples_by_cat[s['category']].append(s)
            
    eval_samples = []
    if samples_by_cat:
        total_samples = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            num_samples = int(args.max_samples * (len(samps) / total_samples))
            eval_samples.extend(samps[:num_samples])
        random.shuffle(eval_samples)"""
    ),
    "src/evaluation/eval_sharpening.py": (
        r"random\.shuffle\(all_samples\)\s*\n\s*eval_samples = all_samples\[:args\.max_samples\]",
        r"""from collections import defaultdict
    samples_by_cat = defaultdict(list)
    for s in all_samples:
        samples_by_cat[s['category']].append(s)
        
    eval_samples = []
    if samples_by_cat:
        total_samples = sum(len(s) for s in samples_by_cat.values())
        for cat, samps in samples_by_cat.items():
            random.shuffle(samps)
            num_samples = int(args.max_samples * (len(samps) / total_samples))
            eval_samples.extend(samps[:num_samples])
        random.shuffle(eval_samples)"""
    )
}

for fpath, (pattern, replacement) in files_to_replace.items():
    if not os.path.exists(fpath):
        continue
    with open(fpath, "r") as f:
        content = f.read()

    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        with open(fpath, "w") as f:
            f.write(new_content)
        print(f"Updated {fpath}")
    else:
        print(f"No match in {fpath}")

