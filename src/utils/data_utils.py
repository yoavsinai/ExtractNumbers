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

def get_gt_from_anno(anno_path: str) -> Tuple[List[Tuple[float, float, float, float]], List[Dict]]:
    """
    Extract GT boxes and labels from annotations.json.
    Returns:
        - List of global bounding boxes (x1, y1, x2, y2)
        - List of digit info dictionaries (bbox=(x1, y1, x2, y2), label=int)
    """
    with open(anno_path, 'r') as f:
        data = json.load(f)
    
    global_boxes = []
    digit_info = []
    
    for number in data.get('detected_numbers', []):
        # Global BB
        bb = number.get('full_bounding_box', {})
        if bb:
            x1, y1 = bb['x'], bb['y']
            global_boxes.append((x1, y1, x1 + bb['width'], y1 + bb['height']))
            
        # Individual Digits
        for digit in number.get('digits', []):
            dbb = digit.get('bounding_box', {})
            if dbb:
                dx1, dy1 = dbb['x'], dbb['y']
                digit_info.append({
                    'bbox': (dx1, dy1, dx1 + dbb['width'], dy1 + dbb['height']),
                    'label': digit.get('label')
                })
                
    return global_boxes, digit_info
