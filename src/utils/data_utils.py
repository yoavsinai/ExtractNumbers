import os
import json
from typing import List, Dict, Tuple

def iter_new_samples(data_root: str) -> List[Dict[str, str]]:
    """Iterate through the unified data structure: data/digits_data/<dataset>/sample_<id>/"""
    samples = []
    if not os.path.exists(data_root):
        return samples
        
    # Sort for deterministic behavior
    datasets = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        samples_in_dataset = sorted([s for s in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, s))])
        
        for sample_folder in samples_in_dataset:
            sample_path = os.path.join(dataset_path, sample_folder)
            img_path = os.path.join(sample_path, "original.png")
            anno_path = os.path.join(sample_path, "annotations.json")
            
            if os.path.exists(img_path) and os.path.exists(anno_path):
                samples.append({
                    "category": dataset,
                    "sample_id": f"{dataset}/{sample_folder}",
                    "image_path": img_path,
                    "anno_path": anno_path
                })
    return samples

def get_gt_from_anno(anno_path: str) -> Tuple[List[Tuple[float, float, float, float]], List[Dict], bool, str]:
    """
    Extract GT boxes and labels from annotations.json.
    Returns:
        - List of global bounding boxes (x1, y1, x2, y2)
        - List of digit info dictionaries (bbox=(x1, y1, x2, y2), label=int)
        - bool: has_digit_boxes (True if individual digit boxes exist)
        - str: full_sequence_label (The complete number string, e.g. "123")
    """
    try:
        with open(anno_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], [], False, ""
    
    global_boxes = []
    digit_info = []
    sequence_parts = []
    
    for number in data.get('detected_numbers', []):
        # Global BB
        bb = number.get('full_bounding_box', {})
        if bb and all(k in bb for k in ['x', 'y', 'width', 'height']):
            x1, y1 = bb['x'], bb['y']
            global_boxes.append((x1, y1, x1 + bb['width'], y1 + bb['height']))
            
        # Full Value if exists (preferred for sequence label)
        if 'full_value' in number and number['full_value']:
            sequence_parts.append({'x': bb.get('x', 0), 'label': str(number['full_value'])})
            
        # Individual Digits
        num_digits = 0
        for digit in number.get('digits', []):
            dbb = digit.get('bounding_box', {})
            if dbb and all(k in dbb for k in ['x', 'y', 'width', 'height']) and 'label' in digit:
                dx1, dy1 = dbb['x'], dbb['y']
                digit_info.append({
                    'bbox': (dx1, dy1, dx1 + dbb['width'], dy1 + dbb['height']),
                    'label': digit.get('label')
                })
                num_digits += 1
                # If full_value was missing, we can build it from digits later
                if not number.get('full_value'):
                    sequence_parts.append({'x': dx1, 'label': str(digit['label'])})
                
    has_digit_boxes = len(digit_info) > 0
    
    # Sort sequence parts by x-coordinate to ensure correct reading order
    sequence_parts.sort(key=lambda x: x['x'])
    full_sequence_label = "".join([p['label'] for p in sequence_parts])
    
    return global_boxes, digit_info, has_digit_boxes, full_sequence_label
