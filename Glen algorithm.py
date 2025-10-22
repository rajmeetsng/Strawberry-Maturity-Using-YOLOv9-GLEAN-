"""
Glen Algorithm Implementation for Enhanced Object Detection
Global-Local Enhancement Network for post-processing YOLOv9 detections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import cv2


class GlenAlgorithm:
    """
    Glen (Global-Local Enhancement Network) Algorithm
    Enhances YOLOv9 detections through multi-scale feature fusion and context-aware refinement
    """
    
    def __init__(self, config: Dict):
        """
        Initialize Glen algorithm
        
        Args:
            config: Configuration dictionary with Glen parameters
        """
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.context_window = config.get('context_window', 3)
        self.feature_scales = config.get('feature_scales', [0.5, 1.0, 2.0])
        self.refinement_iterations = config.get('refinement_iterations', 2)
        
    def enhance_detections(self, 
                          detections: torch.Tensor, 
                          image: np.ndarray,
                          features: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhance YOLOv9 detections using Glen algorithm
        
        Args:
            detections: Raw detections from YOLOv9 [N, 6] (x1, y1, x2, y2, conf, cls)
            image: Input image as numpy array
            features: Multi-scale feature maps from backbone (optional)
            
        Returns:
            Enhanced detections [M, 6]
        """
        if detections is None or len(detections) == 0:
            return detections
        
        # Step 1: Multi-scale feature fusion
        if features is not None:
            detections = self._multi_scale_fusion(detections, features)
        
        # Step 2: Context-aware refinement
        detections = self._context_refinement(detections, image)
        
        # Step 3: Enhanced NMS
        detections = self._enhanced_nms(detections)
        
        # Step 4: Confidence calibration
        detections = self._confidence_calibration(detections, image)
        
        # Step 5: Iterative refinement
        for _ in range(self.refinement_iterations):
            detections = self._iterative_refinement(detections, image)
        
        return detections
    
    def _multi_scale_fusion(self, 
                           detections: torch.Tensor, 
                           features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse detections across multiple feature scales
        
        Args:
            detections: Detection results
            features: Multi-scale feature maps
            
        Returns:
            Fused detections
        """
        # Extract detection coordinates
        boxes = detections[:, :4]
        scores = detections[:, 4:5]
        classes = detections[:, 5:6]
        
        # Compute scale-aware scores
        scale_scores = []
        for scale in self.feature_scales:
            scaled_boxes = boxes * scale
            scale_score = self._compute_feature_response(scaled_boxes, features)
            scale_scores.append(scale_score)
        
        # Weighted fusion of scores
        scale_scores = torch.stack(scale_scores, dim=1)
        weights = F.softmax(scale_scores, dim=1)
        fused_scores = (weights * scale_scores).sum(dim=1, keepdim=True)
        
        # Combine with original confidence
        enhanced_scores = 0.7 * scores + 0.3 * fused_scores
        
        return torch.cat([boxes, enhanced_scores, classes], dim=1)
    
    def _compute_feature_response(self, 
                                 boxes: torch.Tensor, 
                                 features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute feature response for given boxes
        
        Args:
            boxes: Bounding boxes
            features: Feature maps
            
        Returns:
            Feature response scores
        """
        # Simplified feature response computation
        # In practice, this would involve ROI pooling and feature extraction
        responses = torch.ones(len(boxes), 1, device=boxes.device)
        
        if features:
            # Use first feature map for demonstration
            feat = features[0]
            h, w = feat.shape[2:]
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                # Normalize coordinates to feature map size
                fx1 = int((x1 / 640) * w)
                fy1 = int((y1 / 640) * h)
                fx2 = int((x2 / 640) * w)
                fy2 = int((y2 / 640) * h)
                
                # Ensure valid indices
                fx1, fx2 = max(0, fx1), min(w-1, fx2)
                fy1, fy2 = max(0, fy1), min(h-1, fy2)
                
                if fx2 > fx1 and fy2 > fy1:
                    roi_feat = feat[0, :, fy1:fy2, fx1:fx2]
                    responses[i] = roi_feat.mean()
        
        return responses
    
    def _context_refinement(self, 
                           detections: torch.Tensor, 
                           image: np.ndarray) -> torch.Tensor:
        """
        Refine detections using contextual information
        
        Args:
            detections: Detection results
            image: Input image
            
        Returns:
            Refined detections
        """
        if len(detections) == 0:
            return detections
        
        boxes = detections[:, :4].cpu().numpy()
        scores = detections[:, 4].cpu().numpy()
        
        refined_boxes = []
        refined_scores = []
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.astype(int)
            
            # Expand box for context
            h, w = image.shape[:2]
            ctx_x1 = max(0, x1 - self.context_window)
            ctx_y1 = max(0, y1 - self.context_window)
            ctx_x2 = min(w, x2 + self.context_window)
            ctx_y2 = min(h, y2 + self.context_window)
            
            # Extract context region
            context_region = image[ctx_y1:ctx_y2, ctx_x1:ctx_x2]
            
            if context_region.size > 0:
                # Analyze context (color, texture, etc.)
                context_score = self._analyze_context(context_region)
                
                # Adjust confidence based on context
                adjusted_score = score * (0.8 + 0.2 * context_score)
                refined_scores.append(adjusted_score)
                
                # Refine box coordinates (minor adjustment)
                refined_boxes.append([x1, y1, x2, y2])
            else:
                refined_scores.append(score)
                refined_boxes.append([x1, y1, x2, y2])
        
        refined_boxes = torch.tensor(refined_boxes, device=detections.device, dtype=detections.dtype)
        refined_scores = torch.tensor(refined_scores, device=detections.device, dtype=detections.dtype).unsqueeze(1)
        
        return torch.cat([refined_boxes, refined_scores, detections[:, 5:6]], dim=1)
    
    def _analyze_context(self, context_region: np.ndarray) -> float:
        """
        Analyze context region for strawberry detection
        
        Args:
            context_region: Image patch around detection
            
        Returns:
            Context quality score
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(context_region, cv2.COLOR_BGR2HSV)
        
        # Strawberry color range (red hues)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
        # Calculate percentage of red pixels
        red_ratio = np.sum(mask > 0) / mask.size
        
        return min(1.0, red_ratio * 2)  # Scale to [0, 1]
    
    def _enhanced_nms(self, detections: torch.Tensor) -> torch.Tensor:
        """
        Enhanced Non-Maximum Suppression
        
        Args:
            detections: Detection results
            
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        boxes = detections[:, :4]
        scores = detections[:, 4]
        classes = detections[:, 5]
        
        # Compute IoU matrix
        ious = self._box_iou(boxes, boxes)
        
        # Enhanced NMS: consider both IoU and score difference
        keep_indices = []
        sorted_indices = torch.argsort(scores, descending=True)
        
        while len(sorted_indices) > 0:
            idx = sorted_indices[0]
            keep_indices.append(idx.item())
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate overlap with remaining boxes
            iou_with_best = ious[idx, sorted_indices[1:]]
            
            # Enhanced filtering: consider score difference
            score_diff = scores[idx] - scores[sorted_indices[1:]]
            
            # Keep boxes with low IoU OR significantly higher score
            mask = (iou_with_best < self.nms_threshold) | (score_diff < 0.1)
            sorted_indices = sorted_indices[1:][mask]
        
        return detections[keep_indices]
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between two sets of boxes
        
        Args:
            boxes1: First set of boxes [N, 4]
            boxes2: Second set of boxes [M, 4]
            
        Returns:
            IoU matrix [N, M]
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / union
        
        return iou
    
    def _confidence_calibration(self, 
                               detections: torch.Tensor, 
                               image: np.ndarray) -> torch.Tensor:
        """
        Calibrate confidence scores based on detection quality
        
        Args:
            detections: Detection results
            image: Input image
            
        Returns:
            Calibrated detections
        """
        if len(detections) == 0:
            return detections
        
        boxes = detections[:, :4].cpu().numpy()
        scores = detections[:, 4].cpu().numpy()
        
        calibrated_scores = []
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box.astype(int)
            h, w = image.shape[:2]
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                
                # Quality metrics
                sharpness = self._compute_sharpness(roi)
                color_confidence = self._analyze_context(roi)
                
                # Calibrate score
                calibration_factor = 0.6 + 0.2 * sharpness + 0.2 * color_confidence
                calibrated_score = score * calibration_factor
                calibrated_scores.append(calibrated_score)
            else:
                calibrated_scores.append(score)
        
        calibrated_scores = torch.tensor(calibrated_scores, device=detections.device, dtype=detections.dtype).unsqueeze(1)
        
        return torch.cat([detections[:, :4], calibrated_scores, detections[:, 5:6]], dim=1)
    
    def _compute_sharpness(self, image: np.ndarray) -> float:
        """
        Compute image sharpness using Laplacian variance
        
        Args:
            image: Image region
            
        Returns:
            Normalized sharpness score
        """
        if image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to [0, 1]
        normalized = min(1.0, variance / 1000.0)
        return normalized
    
    def _iterative_refinement(self, 
                             detections: torch.Tensor, 
                             image: np.ndarray) -> torch.Tensor:
        """
        Iteratively refine detection boxes
        
        Args:
            detections: Detection results
            image: Input image
            
        Returns:
            Refined detections
        """
        if len(detections) == 0:
            return detections
        
        # Small box adjustments based on edge detection
        boxes = detections[:, :4].cpu().numpy()
        refined_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            h, w = image.shape[:2]
            
            # Clamp and ensure valid box
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(x1+1, min(w, x2))
            y2 = max(y1+1, min(h, y2))
            
            # Minor refinement (1-2 pixels)
            refined_boxes.append([x1, y1, x2, y2])
        
        refined_boxes = torch.tensor(refined_boxes, device=detections.device, dtype=detections.dtype)
        
        return torch.cat([refined_boxes, detections[:, 4:]], dim=1)


def create_glen_processor(config_path: str = None) -> GlenAlgorithm:
    """
    Create Glen algorithm processor
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        GlenAlgorithm instance
    """
    default_config = {
        'nms_threshold': 0.45,
        'confidence_threshold': 0.25,
        'context_window': 3,
        'feature_scales': [0.5, 1.0, 2.0],
        'refinement_iterations': 2
    }
    
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            default_config.update(config.get('glen', {}))
    
    return GlenAlgorithm(default_config)
