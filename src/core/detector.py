"""
YOLO object detection wrapper for P&ID symbol detection.

Provides a clean interface to the Ultralytics YOLO model for detecting
symbols in rendered PDF images.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO

from ..config import get_logger, Settings

logger = get_logger(__name__)


@dataclass
class Detection:
    """Represents a single object detection."""
    
    class_id: int
    class_name: str
    confidence: float
    bbox_pixels: Tuple[float, float, float, float]  # (x0, y0, x1, y1) in pixels
    bbox_pdf: Optional[Tuple[float, float, float, float]] = None  # Optional PDF coords
    associated_texts: List[str] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of detection bbox."""
        x0, y0, x1, y1 = self.bbox_pixels
        return ((x0 + x1) / 2, (y0 + y1) / 2)
    
    @property
    def width(self) -> float:
        """Get detection width in pixels."""
        return self.bbox_pixels[2] - self.bbox_pixels[0]
    
    @property
    def height(self) -> float:
        """Get detection height in pixels."""
        return self.bbox_pixels[3] - self.bbox_pixels[1]
    
    @property
    def area(self) -> float:
        """Get detection area in square pixels."""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary for serialization."""
        result = {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox_pixels": [round(v, 1) for v in self.bbox_pixels],
        }
        if self.bbox_pdf:
            result["bbox_pdf"] = [round(v, 3) for v in self.bbox_pdf]
        if self.associated_texts:
            result["texts_inside"] = self.associated_texts
        return result


@dataclass
class PageDetections:
    """Container for all detections on a single page."""
    
    page_number: int
    detections: List[Detection]
    image_size: Tuple[int, int]  # (width, height)
    dpi: int
    
    @property
    def count(self) -> int:
        """Get number of detections."""
        return len(self.detections)
    
    def get_by_class(self, class_name: str) -> List[Detection]:
        """Get detections filtered by class name."""
        return [d for d in self.detections if d.class_name == class_name]
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert all detections to list of dictionaries."""
        return [d.to_dict() for d in self.detections]


class YOLODetector:
    """
    YOLO object detection wrapper for P&ID symbol detection.
    
    Features:
    - Lazy model loading
    - Configurable confidence threshold
    - GPU/CPU device selection
    - Memory-efficient inference
    
    Usage:
        detector = YOLODetector("models/best.pt")
        detections = detector.detect(image_rgb, confidence=0.25)
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights (uses default if not provided)
            settings: Application settings
        """
        self.settings = settings or Settings()
        self._model_path = model_path or str(self.settings.paths.default_model_path)
        self._model: Optional[YOLO] = None
        self._class_names: Dict[int, str] = {}
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    @property
    def class_names(self) -> Dict[int, str]:
        """Get class names dictionary."""
        if not self._model:
            self.load()
        return self._class_names
    
    @property
    def model_path(self) -> str:
        """Get the model path."""
        return self._model_path
    
    def load(self) -> None:
        """Load the YOLO model."""
        if self._model is not None:
            return
        
        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"YOLO model not found: {self._model_path}")
        
        logger.info(f"Loading YOLO model: {self._model_path}")
        self._model = YOLO(self._model_path)
        self._class_names = self._model.names.copy()
        logger.info(f"Model loaded with {len(self._class_names)} classes: {list(self._class_names.values())}")
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._class_names = {}
    
    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        imgsz: int = 1280,
        verbose: bool = False
    ) -> List[Detection]:
        """
        Run object detection on an image.
        
        Args:
            image: Input image (RGB or BGR numpy array)
            confidence: Confidence threshold (uses default if not provided)
            imgsz: YOLO inference image size
            verbose: Whether to show YOLO verbose output
        
        Returns:
            List of Detection objects
        """
        if not self._model:
            self.load()
        
        confidence = confidence or self.settings.processing.confidence_threshold
        
        # Ensure RGB format if input is BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if image appears to be BGR (heuristic)
            # For YOLO, it handles both, but we'll convert for consistency
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Run inference
        results = self._model(
            source=image_rgb,
            conf=confidence,
            imgsz=imgsz,
            verbose=verbose
        )
        
        # Parse results
        detections = []
        result = results[0]  # Single image
        
        if hasattr(result, "boxes") and len(result.boxes) > 0:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]  # (x0, y0, x1, y1)
                cls_id = int(box.cls.cpu().numpy())
                conf_score = float(box.conf.cpu().numpy())
                
                detections.append(Detection(
                    class_id=cls_id,
                    class_name=self._class_names.get(cls_id, str(cls_id)),
                    confidence=conf_score,
                    bbox_pixels=(
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2]),
                        float(xyxy[3])
                    )
                ))
        
        logger.debug(f"Detected {len(detections)} objects with confidence >= {confidence}")
        return detections
    
    def detect_page(
        self,
        rendered_page,
        confidence: Optional[float] = None
    ) -> PageDetections:
        """
        Run detection on a RenderedPage object.
        
        Args:
            rendered_page: RenderedPage from PDFRenderer
            confidence: Confidence threshold
        
        Returns:
            PageDetections container
        """
        detections = self.detect(rendered_page.image, confidence)
        
        return PageDetections(
            page_number=rendered_page.page_number,
            detections=detections,
            image_size=rendered_page.size,
            dpi=rendered_page.dpi
        )
    
    def get_class_counts(self, detections: List[Detection]) -> Dict[str, int]:
        """
        Count detections by class.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            Dictionary mapping class names to counts
        """
        counts = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        return counts
