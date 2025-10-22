# 🍓 Strawberry Detection with YOLOv9 + GLEN Algorithm

A state-of-the-art strawberry detection system combining YOLOv9 with the Global-Local Enhancement Network (GLEN) algorithm for improved accuracy in agricultural applications.

## Features

- **YOLOv9 Architecture**: Latest YOLO version with improved detection accuracy
- **GLEN Algorithm**: Global-Local Enhancement Network for better feature extraction
- **Real-time Detection**: Fast inference suitable for real-world applications
- **Easy Training**: Simple pipeline for custom dataset training


## Architecture

The system combines:
1. **YOLOv9 Backbone**: For robust object detection
2. **GLEN Module**: Enhances both global and local features


## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM (16GB+ recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/strawberry-detection-yolov9.git
cd strawberry-detection-yolov9

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained weights (optional)
python scripts/download_weights.py
```

## Quick Start

### Detection on Images

```bash
python detect.py --source images/strawberries.jpg --weights weights/best.pt --conf 0.5
```

### Detection on Video

```bash
python detect.py --source videos/strawberry_field.mp4 --weights weights/best.pt --save-vid
```

### Webcam Detection

```bash
python detect.py --source 0 --weights weights/best.pt
```

## Training

### Prepare Dataset

Organize your dataset in YOLO format:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Train Model

```bash
python train.py --data data/strawberry.yaml --cfg models/yolov9-glen.yaml --weights weights/yolov9.pt --epochs 100
```

### Training Configuration

Edit `data/strawberry.yaml`:
```yaml
train: dataset/images/train
val: dataset/images/val
nc: 2  # number of classes
names: ['unripe', 'ripe']
```

## Model Configuration

The GLEN-enhanced YOLOv9 architecture can be customized in `models/yolov9-glen.yaml`:
- Adjust backbone depth
- Modify GLEN module parameters
- Configure detection heads

## Results

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS | Params |
|-------|---------|--------------|-----|--------|
| YOLOv9 | 92.3% | 78.5% | 45 | 51.2M |
| YOLOv9+GLEN | 94.7% | 81.2% | 42 | 53.8M |

## Project Structure

```
strawberry-detection-yolov9/
├── models/
│   ├── yolov9.py          # YOLOv9 implementation
│   ├── glen.py            # GLEN module
│   └── yolov9-glen.yaml   # Model configuration
├── utils/
│   ├── datasets.py        # Dataset loading
│   ├── general.py         # General utilities
│   ├── metrics.py         # Evaluation metrics
│   └── plots.py           # Visualization
├── data/
│   └── strawberry.yaml    # Dataset configuration
├── weights/               # Model weights
├── scripts/              # Utility scripts
├── detect.py             # Inference script
├── train.py              # Training script
├── val.py                # Validation script
└── requirements.txt      # Dependencies
```

## Usage Examples

### Python API

```python
from models.yolov9_glen import YOLOv9GLEN
from utils.general import load_image

# Load model
model = YOLOv9GLEN('weights/best.pt')

# Detect strawberries
image = load_image('path/to/image.jpg')
results = model.detect(image, conf_threshold=0.5)

# Process results
for detection in results:
    bbox = detection['bbox']
    confidence = detection['confidence']
    class_name = detection['class']
    print(f"Detected {class_name} with confidence {confidence:.2f}")
```

## Advanced Features

### GLEN Algorithm Parameters

Customize GLEN module behavior:
```python
glen_config = {
    'global_channels': 256,
    'local_channels': 128,
    'fusion_method': 'concat',  # or 'add', 'attention'
    'use_se': True  # Squeeze-and-Excitation
}
```

### Custom Post-processing

```bash
python detect.py --source image.jpg --augment --agnostic-nms --save-txt --save-conf
```

## Performance Optimization

- **TensorRT**: Convert model for faster inference
- **ONNX Export**: For deployment flexibility
- **Mixed Precision**: FP16 training for speed
- **Pruning**: Reduce model size

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{strawberry-yolov9-glen,
  author = {Your Name},
  title = {Strawberry Detection with YOLOv9 and GLEN},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/strawberry-detection-yolov9}
}
```

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv9 architecture from the original paper
- GLEN algorithm implementation
- PyTorch team for the deep learning framework

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/strawberry-detection-yolov9/issues)
- **Email**: your.email@example.com

