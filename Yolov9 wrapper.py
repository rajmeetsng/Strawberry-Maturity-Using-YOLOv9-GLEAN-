"""
YOLOv9 Wrapper for Strawberry Detection
Integrates YOLOv9 with Glen algorithm
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
import cv2
import sys

# Add YOLOv9 to path
yolov9_path = Path(__file__).parent.parent / 'yolov9'
if yolov9_path.exists():
    sys.path.insert(0, str(yolov9_path))


class YOLOv9Detector:
    """
    YOLOv9 Detector wrapper for strawberry detection
    """
    
    def __init__(self, 
                 weights: str,
                 device: str = 'cuda',
                 img_size: int = 640,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize YOLOv9 detector
        
        Args:
            weights: Path to model weights
            device: Device to run inference on ('cuda' or 'cpu')
            img_size: Input image size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        self.model = self._load_model(weights)
        self.model.to(self.device)
        self.model.eval()
        
        # Class names
        self.names = ['strawberry']
        
    def _load_model(self, weights: str):
        """
        Load YOLOv9 model
        
        Args:
            weights: Path to weights file
            
        Returns:
            Loaded model
        """
        try:
            # Try to load with official YOLOv9 loader
            from models.common import DetectMultiBackend
            model = DetectMultiBackend(weights, device=self.device, dnn=False, fp16=False)
            return model
        except ImportError:
            # Fallback: load as PyTorch checkpoint
            print("Loading model with PyTorch...")
            checkpoint = torch.load(weights, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'ema' in checkpoint:
                    model = checkpoint['ema']
                else:
                    raise ValueError("Cannot find model in checkpoint")
            else:
                model = checkpoint
            
            return model
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image for YOLOv9
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed tensor and original image shape
        """
        # Store original shape
        orig_shape = image.shape[:2]
        
        # Resize
        img = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).to(self.device)
        
        return img_tensor, orig_shape
    
    def postprocess(self, 
                   predictions: torch.Tensor,
                   orig_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Post-process YOLOv9 predictions
        
        Args:
            predictions: Raw model predictions
            orig_shape: Original image shape (h, w)
            
        Returns:
            Processed detections [N, 6] (x1, y1, x2, y2, conf, cls)
        """
        # Handle different output formats
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        
        # Apply confidence threshold
        predictions = predictions[predictions[..., 4] > self.conf_threshold]
        
        if len(predictions) == 0:
            return torch.zeros((0, 6), device=self.device)
        
        # Extract boxes, scores, and classes
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5]
        
        # Handle class predictions
        if predictions.shape[1] > 5:
            class_scores = predictions[:, 5:]
            class_ids = torch.argmax(class_scores, dim=1, keepdim=True).float()
            class_confs = torch.max(class_scores, dim=1, keepdim=True)[0]
            scores = scores * class_confs
        else:
            class_ids = torch.zeros((len(predictions), 1), device=self.device)
        
        # Scale boxes to original image size
        scale_h = orig_shape[0] / self.img_size
        scale_w = orig_shape[1] / self.img_size
        
        boxes[:, [0, 2]] *= scale_w
        boxes[:, [1, 3]] *= scale_h
        
        # Combine results
        detections = torch.cat([boxes, scores, class_ids], dim=1)
        
        # Apply NMS
        detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Non-Maximum Suppression
        
        Args:
            detections: Detections [N, 6]
            
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        boxes = detections[:, :4]
        scores = detections[:, 4]
        
        # Compute areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        order = scores.argsort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            
            i = order[0]
            keep.append(i.item())
            
            # Compute IoU
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            
            w = torch.maximum(torch.tensor(0.0, device=self.device), xx2 - xx1)
            h = torch.maximum(torch.tensor(0.0, device=self.device), yy2 - yy1)
            
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IoU less than threshold
            mask = iou <= self.iou_threshold
            order = order[1:][mask]
        
        return detections[keep]
    
    def detect(self, 
               image: np.ndarray,
               return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Run detection on image
        
        Args:
            image: Input image (BGR format)
            return_features: Whether to return intermediate features
            
        Returns:
            Detections [N, 6] or (Detections, Features)
        """
        # Preprocess
        img_tensor, orig_shape = self.preprocess(image)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
            
            # Extract features if needed
            features = None
            if return_features:
                try:
                    # Try to get intermediate features
                    features = self.model.model[-1].anchors if hasattr(self.model, 'model') else None
                except:
                    features = None
        
        # Postprocess
        detections = self.postprocess(predictions, orig_shape)
        
        if return_features:
            return detections, features
        return detections
    
    def batch_detect(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Run detection on batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of detections for each image
        """
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results


class YOLOv9WithGlen:
    """
    YOLOv9 detector with Glen algorithm enhancement
    """
    
    def __init__(self,
                 weights: str,
                 glen_config: dict,
                 device: str = 'cuda',
                 img_size: int = 640):
        """
        Initialize YOLOv9 + Glen detector
        
        Args:
            weights: Path to YOLOv9 weights
            glen_config: Glen algorithm configuration
            device: Device to run inference on
            img_size: Input image size
        """
        # Initialize YOLOv9
        self.yolo = YOLOv9Detector(
            weights=weights,
            device=device,
            img_size=img_size
        )
        
        # Initialize Glen algorithm
        from models.glen_algorithm import GlenAlgorithm
        self.glen = GlenAlgorithm(glen_config)
        
    def detect(self, image: np.ndarray) -> torch.Tensor:
        """
        Run detection with Glen enhancement
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Enhanced detections [N, 6]
        """
        # Get YOLOv9 detections with features
        detections, features = self.yolo.detect(image, return_features=True)
        
        # Enhance with Glen algorithm
        enhanced_detections = self.glen.enhance_detections(
            detections=detections,
            image=image,
            features=features
        )
        
        return enhanced_detections
    
    def batch_detect(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Run batch detection with Glen enhancement
        
        Args:
            images: List of input images
            
        Returns:
            List of enhanced detections
        """
        results = []
        for image in images:
            detections = self.detect(image)
            results.append(detections)
        return results
