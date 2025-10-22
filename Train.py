"""
Training script for YOLOv9 on strawberry detection dataset
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path
import os

# Add YOLOv9 to path
yolov9_path = Path(__file__).parent / 'yolov9'
if yolov9_path.exists():
    sys.path.insert(0, str(yolov9_path))


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train YOLOv9 for Strawberry Detection')
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml file')
    parser.add_argument('--weights', type=str, default='yolov9-c.pt',
                       help='Initial weights path')
    parser.add_argument('--cfg', type=str, default='',
                       help='Model configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Save results to project/name')
    parser.add_argument('--name', type=str, default='exp',
                       help='Save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='Existing project/name ok, do not increment')
    parser.add_argument('--hyp', type=str, default='',
                       help='Hyperparameters path')
    parser.add_argument('--optimizer', type=str, default='SGD',
                       choices=['SGD', 'Adam', 'AdamW'],
                       help='Optimizer')
    parser.add_argument('--freeze', nargs='+', type=int, default=[],
                       help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--resume', action='store_true',
                       help='Resume most recent training')
    parser.add_argument('--nosave', action='store_true',
                       help='Only save final checkpoint')
    parser.add_argument('--notest', action='store_true',
                       help='Only test final epoch')
    parser.add_argument('--cache', action='store_true',
                       help='Cache images for faster training')
    parser.add_argument('--verbose', action='store_true',
                       help='Print all logging')
    
    return parser.parse_args()


def create_training_config(args):
    """Create training configuration"""
    config = {
        'data': args.data,
        'weights': args.weights,
        'cfg': args.cfg,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'optimizer': args.optimizer,
        'freeze': args.freeze,
        'resume': args.resume,
        'nosave': args.nosave,
        'notest': args.notest,
        'cache': args.cache,
        'verbose': args.verbose
    }
    
    return config


def train_yolov9(config):
    """
    Train YOLOv9 model
    
    Args:
        config: Training configuration dictionary
    """
    try:
        # Try importing YOLOv9 training module
        from train import train as yolov9_train
        
        print("Starting YOLOv9 training...")
        print(f"Data: {config['data']}")
        print(f"Weights: {config['weights']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Image size: {config['img_size']}")
        print(f"Device: {config['device']}")
        
        # Run training
        yolov9_train(**config)
        
        print("\n✓ Training complete!")
        
    except ImportError as e:
        print(f"Error: Could not import YOLOv9 training module: {e}")
        print("\nManual training command:")
        print(f"cd yolov9 && python train.py --data ../{config['data']} "
              f"--weights {config['weights']} --epochs {config['epochs']} "
              f"--batch-size {config['batch_size']} --img {config['img_size']} "
              f"--device {config['device']}")


def validate_data_config(data_path):
    """Validate data configuration file"""
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data config not found: {data_path}")
    
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    required_keys = ['train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in data_config:
            raise ValueError(f"Missing required key in data config: {key}")
    
    print("✓ Data configuration valid")
    return data_config


def main():
    """Main training entry point"""
    args = parse_args()
    
    print("=" * 60)
    print("YOLOv9 Training - Strawberry Detection")
    print("=" * 60)
    
    # Validate data configuration
    validate_data_config(args.data)
    
    # Create training config
    config = create_training_config(args)
    
    # Check for YOLOv9 installation
    yolov9_path = Path('yolov9')
    if not yolov9_path.exists():
        print("\nWarning: YOLOv9 directory not found!")
        print("Please clone YOLOv9 first:")
        print("git clone https://github.com/WongKinYiu/yolov9.git")
        return
    
    # Check for weights
    weights_path = Path(args.weights)
    if not weights_path.exists() and not (yolov9_path / args.weights).exists():
        print(f"\nWarning: Weights not found: {args.weights}")
        print("Please download pre-trained weights or train from scratch")
        print("Available weights: yolov9-c.pt, yolov9-e.pt, yolov9-s.pt")
        return
    
    # Start training
    train_yolov9(config)


if __name__ == '__main__':
    main()
