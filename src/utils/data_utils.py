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
def parse_anno_data(data: dict) -> Tuple[List[Tuple[float, float, float, float]], List[Dict], bool, str]:
    """
    Parse annotation dictionary.
    Returns:
        - List of global bounding boxes (x1, y1, x2, y2)
        - List of digit info dictionaries (bbox=(x1, y1, x2, y2), label=int)
        - bool: has_digit_boxes (True only if individual digit boxes with valid labels exist)
        - str: full_sequence_label (the complete number string, e.g. "123")
    """
    global_boxes = []
    digit_info = []
    sequence_parts = []

    # Labels that mean "unknown" — used by Moving MNIST which has no ground truth digit identity
    UNKNOWN_LABELS = {"digit", "digit_seq", None, -1, ""}
    UNKNOWN_FULL_VALUES = {"digit_seq", "", None}

    for number in data.get('detected_numbers', []):
        # --- Global bounding box ---
        bb = number.get('full_bounding_box', {})
        if bb and all(k in bb for k in ['x', 'y', 'width', 'height']):
            x1, y1 = bb['x'], bb['y']
            global_boxes.append((x1, y1, x1 + bb['width'], y1 + bb['height']))

        # --- Full sequence value ---
        # Only use it if it's a real transcription (not a placeholder like "digit_seq")
        full_value = number.get('full_value')
        if full_value not in UNKNOWN_FULL_VALUES:
            sequence_parts.append({
                'x': bb.get('x', 0),
                'label': str(full_value)
            })

        # --- Individual digit boxes ---
        for digit in number.get('digits', []):
            dbb = digit.get('bounding_box', {})
            label = digit.get('label')

            # Skip unknown/placeholder labels (Moving MNIST case)
            if label in UNKNOWN_LABELS:
                continue

            if not (dbb and all(k in dbb for k in ['x', 'y', 'width', 'height'])):
                continue

            dx1, dy1 = dbb['x'], dbb['y']

            # Cast label to int safely
            try:
                int_label = int(label)
            except (ValueError, TypeError):
                continue  # skip anything that can't be cast to a digit

            digit_info.append({
                'bbox': (dx1, dy1, dx1 + dbb['width'], dy1 + dbb['height']),
                'label': int_label
            })

            # If full_value was missing/unknown, build sequence from individual digits
            if full_value in UNKNOWN_FULL_VALUES:
                sequence_parts.append({'x': dx1, 'label': str(int_label)})

    has_digit_boxes = len(digit_info) > 0

    # Sort by x-coordinate for correct reading order left → right
    sequence_parts.sort(key=lambda p: p['x'])
    full_sequence_label = "".join([p['label'] for p in sequence_parts])

    return global_boxes, digit_info, has_digit_boxes, full_sequence_label

def get_gt_from_anno(anno_path: str) -> Tuple[List[Tuple[float, float, float, float]], List[Dict], bool, str]:
    """
    Extract GT boxes and labels from annotations.json.
    """
    try:
        with open(anno_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return [], [], False, ""
    return parse_anno_data(data)

def iter_video_samples(data_root: str) -> List[Dict[str, str]]:
    """Iterate through the video data structure: data/video_data/<dataset>/sample_<id>/"""
    samples = []
    if not os.path.exists(data_root):
        return samples
        
    datasets = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        samples_in_dataset = sorted([s for s in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, s))])
        
        for sample_folder in samples_in_dataset:
            sample_path = os.path.join(dataset_path, sample_folder)
            
            # Find any video file (.mp4) in the folder
            video_files = [f for f in os.listdir(sample_path) if f.endswith(".mp4")]
            anno_path = os.path.join(sample_path, "annotations.json")
            
            if video_files and os.path.exists(anno_path):
                samples.append({
                    "category": dataset,
                    "sample_id": f"{dataset}/{sample_folder}",
                    "video_path": os.path.join(sample_path, video_files[0]),
                    "anno_path": anno_path
                })
    return samples

from functools import lru_cache

@lru_cache(maxsize=16)
def _load_json_cached(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def get_video_gt_from_anno(anno_path: str, frame_idx: int) -> Tuple[List[Tuple[float, float, float, float]], List[Dict], bool, str]:
    """
    Extract GT boxes and labels for a specific frame from video annotations.json.
    """
    data = _load_json_cached(anno_path)
    if not data:
        return [], [], False, ""
        
    frames_data = data.get("frames", {})
    frame_data = frames_data.get(str(frame_idx)) or frames_data.get(int(frame_idx))
    
    if not frame_data:
        return [], [], False, ""
        
    return parse_anno_data(frame_data)

def create_mock_video_dataset(data_root: str):
    """
    Generate a mock video dataset with annotations for testing video pipelines.
    """
    import cv2
    import numpy as np
    
    dataset_dir = os.path.join(data_root, "mock_video")
    sample_dir = os.path.join(dataset_dir, "sample_001")
    os.makedirs(sample_dir, exist_ok=True)
    
    video_path = os.path.join(sample_dir, "video.mp4")
    anno_path = os.path.join(sample_dir, "annotations.json")
    
    if os.path.exists(video_path) and os.path.exists(anno_path):
        return
        
    print(f"Generating mock video dataset at {sample_dir}...")
    
    width, height = 256, 256
    num_frames = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
    
    # We will draw a moving sequence "789"
    # To make it realistic, the number will start at (50, 100) and move slowly to (100, 100)
    anno_data = {
        "video_metadata": {
            "sample_id": "mock_video/sample_001",
            "width": width,
            "height": height,
            "fps": 10.0
        },
        "frames": {}
    }
    
    for i in range(num_frames):
        # Create a frame with random noise background
        frame = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
        
        # Position of text
        tx = 50 + int(i * 1.5)
        ty = 120
        
        # Draw number "789"
        cv2.putText(frame, "789", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        out.write(frame)
        
        # Annotations for frames: annotate every 3rd frame to simulate sparse annotations
        if i % 3 == 0:
            # Approximate bounding box for the text "789"
            # Each character is about 24px wide, 35px high
            gx = tx
            gy = ty - 35
            gw = 80
            gh = 45
            
            # Sub-digits: "7", "8", "9"
            digits = []
            labels = [7, 8, 9]
            dw = gw / 3
            for idx, label in enumerate(labels):
                digits.append({
                    "label": label,
                    "bounding_box": {
                        "x": float(gx + idx * dw),
                        "y": float(gy),
                        "width": float(dw),
                        "height": float(gh)
                    }
                })
                
            anno_data["frames"][str(i)] = {
                "detected_numbers": [
                    {
                        "full_value": "789",
                        "full_bounding_box": {
                            "x": float(gx),
                            "y": float(gy),
                            "width": float(gw),
                            "height": float(gh)
                        },
                        "digits": digits
                    }
                ]
            }
            
    out.release()
    
    with open(anno_path, 'w') as f:
        json.dump(anno_data, f, indent=4)
    print("Mock video dataset generated successfully.")


