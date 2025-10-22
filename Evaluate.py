"""
Evaluation script for strawberry detection model
"""

import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.yolov9_wrapper import YOLOv9Detector, YOLOv9WithGlen
from utils.metrics import calculate_metrics, calculate_confusion_matrix, print_metrics
from utils.general import load_config
from utils.visualization import plot_detection_stats


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Strawberry Detection Model')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--glen', action='store_true',
                       help='Use Glen algorithm')
    parser.add_argument('--config', type=str, default='config/glen_config.yaml',
                       help='Glen config file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for metrics')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results to text files')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save evaluation plots')
    parser.add_argument('--output', type=str, default='runs/evaluate',
                       help='Output directory')
    
    return parser.parse_args()


def load_ground_truths(label_path: Path, img_shape: tuple) -> torch.Tensor:
    """
    Load ground truth labels from YOLO format file
    
    Args:
        label_path: Path to label file
        img_shape: Image shape (height, width)
        
    Returns:
        Ground truth tensor [N, 5] (x1, y1, x2, y2, cls)
    """
    if not label_path.exists():
        return torch.zeros((0, 5))
    
    h, w = img_shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        cls = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        width = float(parts[3]) * w
        height = float(parts[4]) * h
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        boxes.append([x1, y1, x2, y2, cls])
    
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32)
    return torch.zeros((0, 5))


def evaluate(args):
    """Main evaluation function"""
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data config
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Initialize detector
    if args.glen:
        glen_config = load_config(args.config)
        detector = YOLOv9WithGlen(
            weights=args.weights,
            glen_config=glen_config.get('glen', {}),
            device=args.device,
            img_size=args.img_size
        )
        print("✓ Glen algorithm enabled")
    else:
        detector = YOLOv9Detector(
            weights=args.weights,
            device=args.device,
            img_size=args.img_size,
            conf_threshold=args.conf_threshold,
            iou_threshold=0.45
        )
    
    print(f"✓ Model loaded: {args.weights}")
    
    # Get test/validation images
    test_path = data_config.get('test', data_config.get('val'))
    if test_path is None:
        raise ValueError("No test or val path specified in data.yaml")
    
    test_dir = Path(test_path)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Get image files
    image_files = list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png'))
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {test_dir}")
    
    print(f"✓ Found {len(image_files)} test images")
    
    # Get labels directory
    labels_dir = test_dir.parent.parent / 'labels' / test_dir.name
    
    print("\nRunning evaluation...")
    
    all_predictions = []
    all_ground_truths = []
    
    # Process images
    for img_path in tqdm(image_files, desc="Evaluating"):
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        # Run detection
        detections = detector.detect(image)
        all_predictions.append(detections)
        
        # Load ground truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt = load_ground_truths(label_path, image.shape)
        all_ground_truths.append(gt)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )
    
    # Print metrics
    print_metrics(metrics)
    
    # Calculate confusion matrix
    conf_matrix = calculate_confusion_matrix(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        iou_threshold=args.iou_threshold,
        num_classes=data_config['nc']
    )
    
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Save metrics
    metrics_file = output_dir / 'metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f)
    print(f"\n✓ Metrics saved to {metrics_file}")
    
    # Save plots
    if args.save_plots:
        plot_path = output_dir / 'detection_stats.png'
        plot_detection_stats(all_predictions, save_path=plot_path)
        print(f"✓ Plots saved to {plot_path}")
    
    # Save detailed results
    if args.save_txt:
        results_file = output_dir / 'results.txt'
        with open(results_file, 'w') as f:
            for i, (pred, gt) in enumerate(zip(all_predictions, all_ground_truths)):
                f.write(f"Image {i}: ")
                f.write(f"Predictions: {len(pred) if pred is not None else 0}, ")
                f.write(f"Ground Truth: {len(gt) if gt is not None else 0}\n")
        print(f"✓ Results saved to {results_file}")
    
    return metrics


def main():
    """Main entry point"""
    args = parse_args()
    
    print("=" * 60)
    print("Model Evaluation - Strawberry Detection")
    print("=" * 60)
    
    metrics = evaluate(args)
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
