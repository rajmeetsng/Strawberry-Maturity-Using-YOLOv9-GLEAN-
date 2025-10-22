"""
Visualization utilities for strawberry detection
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path


def draw_detections(image: np.ndarray,
                   detections: torch.Tensor,
                   class_names: List[str],
                   show_conf: bool = True,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image (BGR format)
        detections: Detections tensor [N, 6] (x1, y1, x2, y2, conf, cls)
        class_names: List of class names
        show_conf: Whether to show confidence scores
        color: Box color (BGR)
        thickness: Box line thickness
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    if detections is None or len(detections) == 0:
        return annotated
    
    # Convert to numpy if tensor
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
        if show_conf:
            label = f"{label} {conf:.2f}"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_width, label_height = label_size
        
        # Ensure label stays within image bounds
        y1_label = max(y1, label_height + 10)
        
        cv2.rectangle(annotated, 
                     (x1, y1_label - label_height - 10),
                     (x1 + label_width, y1_label),
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated, label,
                   (x1, y1_label - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 255), 1)
    
    return annotated


def draw_comparison(image: np.ndarray,
                   detections_yolo: torch.Tensor,
                   detections_glen: torch.Tensor,
                   class_names: List[str]) -> np.ndarray:
    """
    Draw side-by-side comparison of YOLOv9 and YOLOv9+Glen results
    
    Args:
        image: Input image
        detections_yolo: YOLOv9 detections
        detections_glen: YOLOv9+Glen detections
        class_names: List of class names
        
    Returns:
        Comparison image
    """
    # Create two copies
    img_yolo = draw_detections(image.copy(), detections_yolo, class_names,
                               color=(0, 0, 255))  # Red for YOLOv9
    img_glen = draw_detections(image.copy(), detections_glen, class_names,
                              color=(0, 255, 0))  # Green for Glen
    
    # Add labels
    cv2.putText(img_yolo, "YOLOv9", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img_glen, "YOLOv9 + Glen", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add detection counts
    count_yolo = len(detections_yolo) if detections_yolo is not None else 0
    count_glen = len(detections_glen) if detections_glen is not None else 0
    
    cv2.putText(img_yolo, f"Detections: {count_yolo}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_glen, f"Detections: {count_glen}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Stack horizontally
    comparison = np.hstack([img_yolo, img_glen])
    
    return comparison


def plot_detection_stats(detections_list: List[torch.Tensor],
                        save_path: Path = None):
    """
    Plot detection statistics
    
    Args:
        detections_list: List of detection results
        save_path: Path to save plot
    """
    # Extract statistics
    num_detections = [len(d) if d is not None else 0 for d in detections_list]
    confidences = []
    
    for detections in detections_list:
        if detections is not None and len(detections) > 0:
            if isinstance(detections, torch.Tensor):
                confidences.extend(detections[:, 4].cpu().numpy().tolist())
            else:
                confidences.extend(detections[:, 4].tolist())
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Detection counts
    axes[0].hist(num_detections, bins=max(num_detections) + 1 if num_detections else 10,
                edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Detections')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Detection Count Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence distribution
    if confidences:
        axes[1].hist(confidences, bins=20, edgecolor='black', alpha=0.7, color='green')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Confidence Score Distribution')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=np.mean(confidences), color='red', linestyle='--',
                       label=f'Mean: {np.mean(confidences):.3f}')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_detection_grid(images: List[np.ndarray],
                         detections_list: List[torch.Tensor],
                         class_names: List[str],
                         grid_size: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """
    Create a grid of detection results
    
    Args:
        images: List of images
        detections_list: List of detections
        class_names: List of class names
        grid_size: Grid dimensions (rows, cols)
        
    Returns:
        Grid image
    """
    rows, cols = grid_size
    num_images = min(len(images), rows * cols)
    
    # Resize images to same size
    target_size = (640, 640)
    resized_images = []
    
    for i in range(num_images):
        img = images[i].copy()
        detections = detections_list[i]
        
        # Draw detections
        annotated = draw_detections(img, detections, class_names)
        
        # Resize
        resized = cv2.resize(annotated, target_size)
        resized_images.append(resized)
    
    # Fill remaining slots with blank images
    while len(resized_images) < rows * cols:
        blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        resized_images.append(blank)
    
    # Create grid
    grid_rows = []
    for i in range(rows):
        row_images = resized_images[i*cols:(i+1)*cols]
        grid_row = np.hstack(row_images)
        grid_rows.append(grid_row)
    
    grid = np.vstack(grid_rows)
    
    return grid


def save_results(detections: torch.Tensor,
                save_path: Path,
                image_shape: Tuple[int, int],
                save_conf: bool = False):
    """
    Save detection results to text file in YOLO format
    
    Args:
        detections: Detections tensor [N, 6]
        save_path: Path to save file
        image_shape: Image shape (height, width)
        save_conf: Whether to save confidence scores
    """
    if detections is None or len(detections) == 0:
        # Create empty file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.touch()
        return
    
    # Convert to numpy
    if isinstance(detections, torch.Tensor):
        detections = detections.cpu().numpy()
    
    h, w = image_shape[:2]
    
    # Convert to YOLO format (class x_center y_center width height)
    lines = []
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        
        # Convert to normalized coordinates
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        
        cls_id = int(cls)
        
        if save_conf:
            line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}"
        else:
            line = f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
        lines.append(line)
    
    # Save to file
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))


def visualize_heatmap(image: np.ndarray,
                     detections: torch.Tensor,
                     save_path: Path = None) -> np.ndarray:
    """
    Create detection density heatmap
    
    Args:
        image: Input image
        detections: Detections tensor
        save_path: Path to save heatmap
        
    Returns:
        Heatmap overlay image
    """
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    if detections is not None and len(detections) > 0:
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, _ = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                # Add confidence-weighted detection
                heatmap[y1:y2, x1:x2] += conf
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8),
                                       cv2.COLORMAP_JET)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    if save_path:
        cv2.imwrite(str(save_path), overlay)
    
    return overlay
