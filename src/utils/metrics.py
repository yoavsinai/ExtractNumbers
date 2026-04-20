from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        boxA, boxB: Bounding boxes in (x1, y1, x2, y2) format.
        
    Returns:
        float: IoU value between 0.0 and 1.0.
    """
    if boxA is None or boxB is None:
        return 0.0
        
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both bounding boxes
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def calculate_mean_iou(gt_boxes, pred_boxes):
    """
    Calculate the mean IoU for a list of GT and predicted boxes.
    This assumes a 1-to-1 mapping or matching. For multiple boxes,
    a matching algorithm should be used, but for simplicity we'll 
    handle the single-box case or assume they are ordered.
    """
    ious = []
    for gt, pred in zip(gt_boxes, pred_boxes):
        ious.append(calculate_iou(gt, pred))
    
    return np.mean(ious) if ious else 0.0

def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    Calculate common classification metrics.
    
    Args:
        y_true (list or np.array): Ground truth labels.
        y_pred (list or np.array): Predicted labels.
        average (str): Type of averaging to use for multi-class tasks ('weighted', 'macro', 'micro').
        
    Returns:
        dict: A dictionary containing metrics.
    """
    # Filter out samples where either true or pred is None if they are not strings
    filtered_true = []
    filtered_pred = []
    for t, p in zip(y_true, y_pred):
        if t is not None and p is not None:
            filtered_true.append(t)
            filtered_pred.append(p)
    
    if not filtered_true:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    metrics = {
        'accuracy': accuracy_score(filtered_true, filtered_pred),
        'precision': precision_score(filtered_true, filtered_pred, average=average, zero_division=0),
        'recall': recall_score(filtered_true, filtered_pred, average=average, zero_division=0),
        'f1': f1_score(filtered_true, filtered_pred, average=average, zero_division=0)
    }
    return metrics

def print_metrics_report(y_true, y_pred, title="Evaluation Report", average='weighted'):
    """
    Print a detailed metrics report.
    """
    print(f"\n{'='*20} {title} {'='*20}")
    
    # Filter out None values
    filtered_true = []
    filtered_pred = []
    for t, p in zip(y_true, y_pred):
        if t is not None and p is not None:
            filtered_true.append(t)
            filtered_pred.append(p)
            
    if not filtered_true:
        print("No valid samples to evaluate.")
        print('='*50)
        return {}

    metrics = calculate_metrics(filtered_true, filtered_pred, average=average)
    
    print(f"Total Samples: {len(filtered_true)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} ({average})")
    print(f"Recall:    {metrics['recall']:.4f} ({average})")
    print(f"F1-score:  {metrics['f1']:.4f} ({average})")
    
    print("\nDetailed Classification Report:")
    # We only show the report if the number of unique labels is not too large
    unique_labels = sorted(list(set(filtered_true) | set(filtered_pred)))
    if len(unique_labels) <= 50: # Avoid huge reports for string-based comparisons
        report = classification_report(filtered_true, filtered_pred, zero_division=0)
        print(report)
        return metrics, report
    else:
        print("Report too large to display (many unique classes).")
        return metrics, None
        
    print('='*50)
    return metrics
