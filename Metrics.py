"""
Evaluation metrics for object detection
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes
    
    Args:
        box1: Boxes [N, 4] (x1, y1, x2, y2)
        box2: Boxes [M, 4] (x1, y1, x2, y2)
        
    Returns:
        IoU matrix [N, M]
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision (AP)
    
    Args:
        recall: Recall values
        precision: Precision values
        
    Returns:
        Average Precision
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Calculate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def calculate_metrics(predictions: List[torch.Tensor],
                     ground_truths: List[torch.Tensor],
                     iou_threshold: float = 0.5,
                     conf_threshold: float = 0.25) -> Dict[str, float]:
    """
    Calculate detection metrics (mAP, Precision, Recall, F1)
    
    Args:
        predictions: List of prediction tensors [N, 6] (x1, y1, x2, y2, conf, cls)
        ground_truths: List of ground truth tensors [M, 5] (x1, y1, x2, y2, cls)
        iou_threshold: IoU threshold for matching
        conf_threshold: Confidence threshold
        
    Returns:
        Dictionary of metrics
    """
    all_detections = []
    all_ground_truths = []
    
    # Collect all predictions and ground truths
    for pred, gt in zip(predictions, ground_truths):
        if pred is not None and len(pred) > 0:
            # Filter by confidence
            pred = pred[pred[:, 4] >= conf_threshold]
            all_detections.append(pred)
        else:
            all_detections.append(torch.zeros((0, 6)))
        
        if gt is not None and len(gt) > 0:
            all_ground_truths.append(gt)
        else:
            all_ground_truths.append(torch.zeros((0, 5)))
    
    # Calculate per-class metrics
    num_classes = 1  # Strawberry only
    aps = []
    precisions = []
    recalls = []
    
    for cls_id in range(num_classes):
        # Collect detections and ground truths for this class
        cls_detections = []
        cls_gt = []
        
        for i, (dets, gts) in enumerate(zip(all_detections, all_ground_truths)):
            if len(dets) > 0:
                cls_mask = dets[:, 5] == cls_id
                if cls_mask.any():
                    cls_det = dets[cls_mask]
                    cls_det = torch.cat([
                        torch.full((len(cls_det), 1), i, device=cls_det.device),
                        cls_det
                    ], dim=1)
                    cls_detections.append(cls_det)
            
            if len(gts) > 0:
                cls_gt_mask = gts[:, 4] == cls_id
                if cls_gt_mask.any():
                    cls_gt_item = gts[cls_gt_mask]
                    cls_gt_item = torch.cat([
                        torch.full((len(cls_gt_item), 1), i, device=cls_gt_item.device),
                        cls_gt_item
                    ], dim=1)
                    cls_gt.append(cls_gt_item)
        
        if len(cls_detections) == 0:
            continue
        
        # Concatenate all detections
        cls_detections = torch.cat(cls_detections, dim=0)
        
        # Sort by confidence
        sorted_indices = torch.argsort(cls_detections[:, 5], descending=True)
        cls_detections = cls_detections[sorted_indices]
        
        # Count ground truths
        num_gt = sum(len(gt[gt[:, 5] == cls_id]) if len(gt) > 0 else 0 for gt in cls_gt)
        
        if num_gt == 0:
            continue
        
        # Match detections to ground truths
        tp = torch.zeros(len(cls_detections))
        fp = torch.zeros(len(cls_detections))
        
        for det_idx, detection in enumerate(cls_detections):
            img_idx = int(detection[0].item())
            det_box = detection[1:5]
            
            # Find ground truths for this image
            img_gt = [gt[gt[:, 0] == img_idx] if len(gt) > 0 else torch.zeros((0, 6)) 
                     for gt in cls_gt]
            img_gt = torch.cat(img_gt) if len(img_gt) > 0 else torch.zeros((0, 6))
            
            if len(img_gt) > 0:
                gt_boxes = img_gt[:, 1:5]
                ious = box_iou(det_box.unsqueeze(0), gt_boxes)[0]
                
                max_iou, max_idx = ious.max(0)
                
                if max_iou >= iou_threshold:
                    tp[det_idx] = 1
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1
        
        # Compute cumulative TP and FP
        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)
        
        # Compute precision and recall
        recall = tp_cumsum / (num_gt + 1e-6)
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Compute AP
        ap = compute_ap(recall.cpu().numpy(), precision.cpu().numpy())
        aps.append(ap)
        
        # Store final precision and recall
        if len(recall) > 0:
            recalls.append(recall[-1].item())
            precisions.append(precision[-1].item())
    
    # Calculate mean metrics
    mAP = np.mean(aps) if aps else 0.0
    precision = np.mean(precisions) if precisions else 0.0
    recall = np.mean(recalls) if recalls else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    metrics = {
        'mAP@0.5': mAP,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    return metrics


def calculate_confusion_matrix(predictions: List[torch.Tensor],
                               ground_truths: List[torch.Tensor],
                               iou_threshold: float = 0.5,
                               num_classes: int = 1) -> np.ndarray:
    """
    Calculate confusion matrix
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_threshold: IoU threshold
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes+1, num_classes+1]
    """
    # Include background as class 0
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)
    
    for pred, gt in zip(predictions, ground_truths):
        if gt is None or len(gt) == 0:
            continue
        
        gt_boxes = gt[:, :4]
        gt_classes = gt[:, 4].long()
        
        if pred is None or len(pred) == 0:
            # All ground truths are FN (missed detections)
            for cls in gt_classes:
                matrix[cls + 1, 0] += 1
            continue
        
        pred_boxes = pred[:, :4]
        pred_classes = pred[:, 5].long()
        
        # Compute IoU matrix
        ious = box_iou(pred_boxes, gt_boxes)
        
        # Match predictions to ground truths
        matched_gt = set()
        
        for pred_idx in range(len(pred)):
            max_iou, gt_idx = ious[pred_idx].max(0)
            
            if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
                matched_gt.add(gt_idx.item())
                pred_cls = pred_classes[pred_idx].item()
                gt_cls = gt_classes[gt_idx].item()
                matrix[gt_cls + 1, pred_cls + 1] += 1
            else:
                # False positive (background predicted as class)
                pred_cls = pred_classes[pred_idx].item()
                matrix[0, pred_cls + 1] += 1
        
        # Unmatched ground truths are false negatives
        for gt_idx in range(len(gt)):
            if gt_idx not in matched_gt:
                gt_cls = gt_classes[gt_idx].item()
                matrix[gt_cls + 1, 0] += 1
    
    return matrix


def print_metrics(metrics: Dict[str, float]):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:15s}: {value:.4f}")
    
    print("="*50 + "\n")
