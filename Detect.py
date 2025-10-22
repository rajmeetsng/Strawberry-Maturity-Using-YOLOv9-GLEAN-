"""
Detection script for strawberry detection using YOLOv9 + Glen
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import yaml
import time
from typing import List
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.yolov9_wrapper import YOLOv9Detector, YOLOv9WithGlen
from utils.visualization import draw_detections, save_results
from utils.general import load_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Strawberry Detection using YOLOv9 + Glen')
    
    parser.add_argument('--source', type=str, required=True,
                       help='Source: image file, video file, directory, or webcam (0)')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to YOLOv9 weights file')
    parser.add_argument('--glen', action='store_true',
                       help='Enable Glen algorithm enhancement')
    parser.add_argument('--config', type=str, default='config/glen_config.yaml',
                       help='Path to Glen configuration file')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on (cuda/cpu)')
    parser.add_argument('--output', type=str, default='runs/detect',
                       help='Output directory')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results to text files')
    parser.add_argument('--save-conf', action='store_true',
                       help='Save confidence scores in labels')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save images/videos')
    parser.add_argument('--view-img', action='store_true',
                       help='Display results')
    parser.add_argument('--fps', action='store_true',
                       help='Show FPS in output')
    
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    """Load image from path"""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def load_video(path: str):
    """Load video capture"""
    if path.isdigit():
        cap = cv2.VideoCapture(int(path))
    else:
        cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")
    
    return cap


def process_image(image: np.ndarray, 
                 detector,
                 args) -> tuple:
    """
    Process single image
    
    Returns:
        (annotated_image, detections, inference_time)
    """
    start_time = time.time()
    
    # Run detection
    detections = detector.detect(image)
    
    inference_time = time.time() - start_time
    
    # Draw results
    annotated = draw_detections(
        image=image.copy(),
        detections=detections,
        class_names=['strawberry'],
        show_conf=True
    )
    
    return annotated, detections, inference_time


def run_detection(args):
    """Main detection function"""
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    if args.glen:
        config = load_config(args.config)
        glen_config = config.get('glen', {})
        
        # Initialize detector with Glen
        detector = YOLOv9WithGlen(
            weights=args.weights,
            glen_config=glen_config,
            device=args.device,
            img_size=args.img_size
        )
        print("✓ Glen algorithm enabled")
    else:
        # Initialize standard YOLOv9
        detector = YOLOv9Detector(
            weights=args.weights,
            device=args.device,
            img_size=args.img_size,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        print("✓ Using standard YOLOv9")
    
    print(f"✓ Model loaded: {args.weights}")
    print(f"✓ Device: {args.device}")
    
    # Determine source type
    source_path = Path(args.source)
    
    # Process based on source type
    if source_path.is_file() and source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Single image
        print(f"\nProcessing image: {args.source}")
        image = load_image(args.source)
        
        annotated, detections, inf_time = process_image(image, detector, args)
        
        print(f"Detected {len(detections)} strawberries")
        print(f"Inference time: {inf_time*1000:.1f}ms")
        
        # Save results
        if not args.no_save:
            output_path = output_dir / f"result_{source_path.name}"
            cv2.imwrite(str(output_path), annotated)
            print(f"✓ Saved to {output_path}")
        
        # Save labels
        if args.save_txt:
            save_results(detections, output_dir / f"result_{source_path.stem}.txt", 
                        image.shape, args.save_conf)
        
        # Display
        if args.view_img:
            cv2.imshow('Detection Result', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif source_path.is_file() and source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        # Video file
        print(f"\nProcessing video: {args.source}")
        process_video(args.source, detector, args, output_dir)
    
    elif source_path.is_dir():
        # Directory of images
        print(f"\nProcessing directory: {args.source}")
        image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
        
        total_detections = 0
        total_time = 0
        
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            image = load_image(str(img_path))
            
            annotated, detections, inf_time = process_image(image, detector, args)
            
            total_detections += len(detections)
            total_time += inf_time
            
            # Save results
            if not args.no_save:
                output_path = output_dir / f"result_{img_path.name}"
                cv2.imwrite(str(output_path), annotated)
            
            if args.save_txt:
                save_results(detections, output_dir / f"result_{img_path.stem}.txt",
                           image.shape, args.save_conf)
        
        print(f"\n✓ Processed {len(image_files)} images")
        print(f"✓ Total detections: {total_detections}")
        print(f"✓ Average inference time: {(total_time/len(image_files))*1000:.1f}ms")
    
    elif args.source.isdigit():
        # Webcam
        print(f"\nStarting webcam detection...")
        process_video(args.source, detector, args, output_dir)
    
    else:
        raise ValueError(f"Invalid source: {args.source}")


def process_video(source, detector, args, output_dir):
    """Process video file or webcam"""
    
    cap = load_video(source)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    if not args.no_save and not source.isdigit():
        output_path = output_dir / f"result_{Path(source).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    else:
        out = None
    
    frame_count = 0
    total_time = 0
    
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated, detections, inf_time = process_image(frame, detector, args)
            
            frame_count += 1
            total_time += inf_time
            
            # Add FPS overlay
            if args.fps:
                fps_text = f"FPS: {1/inf_time:.1f} | Detections: {len(detections)}"
                cv2.putText(annotated, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frame
            if out is not None:
                out.write(annotated)
            
            # Display
            if args.view_img or source.isdigit():
                cv2.imshow('Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_fps = frame_count / total_time
            print(f"\n✓ Processed {frame_count} frames")
            print(f"✓ Average FPS: {avg_fps:.1f}")
            if out is not None:
                print(f"✓ Saved to {output_path}")


def main():
    """Main entry point"""
    args = parse_args()
    
    print("=" * 60)
    print("Strawberry Detection - YOLOv9 + Glen Algorithm")
    print("=" * 60)
    
    run_detection(args)
    
    print("\n✓ Detection complete!")


if __name__ == '__main__':
    main()
