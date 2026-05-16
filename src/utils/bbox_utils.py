import numpy as np
from typing import List, Tuple, Union

def compute_iou(box1: Union[np.ndarray, Tuple[float, float, float, float]], 
                box2: Union[np.ndarray, Tuple[float, float, float, float]]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / float(area1 + area2 - inter)

def merge_global_boxes(boxes: np.ndarray) -> np.ndarray:
    """
    Merge multiple bounding boxes into a single encompassing bounding box.
    boxes: Nx4 array of (x1, y1, x2, y2)
    Returns: 1x4 array (x1, y1, x2, y2)
    """
    if boxes is None or len(boxes) == 0:
        return np.array([])
    
    x1 = np.min(boxes[:, 0])
    y1 = np.min(boxes[:, 1])
    x2 = np.max(boxes[:, 2])
    y2 = np.max(boxes[:, 3])
    
    return np.array([x1, y1, x2, y2])

def nms_individual_boxes(boxes: np.ndarray, confs: np.ndarray, iou_thresh: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Non-Maximum Suppression to choose one box among duplicates for individual digits.
    boxes: Nx4 array of (x1, y1, x2, y2)
    confs: N array of confidences
    iou_thresh: IoU threshold for suppression
    Returns: (kept_boxes, kept_confs)
    """
    if len(boxes) == 0:
        return np.array([]), np.array([])

    indices = np.argsort(confs)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        rest_indices = indices[1:]
        current_box = boxes[current]
        
        ious = np.array([compute_iou(current_box, boxes[idx]) for idx in rest_indices])
        
        # Keep boxes with IoU less than threshold (i.e. not duplicates)
        indices = rest_indices[ious <= iou_thresh]

    return boxes[keep], confs[keep]
