"""
Main processing pipeline for P&ID PDF detection.

Orchestrates the complete workflow from PDF input to detection results,
integrating PDF rendering, YOLO detection, and text extraction.
"""

import json
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm

from .pdf_renderer import PDFRenderer, RenderedPage
from .text_extractor import TextExtractor, TextWord
from .coordinate_transformer import CoordinateTransformer
from .detector import YOLODetector, Detection, PageDetections

from ..config import get_logger, Settings

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Complete processing result for a PDF document."""
    
    pdf_path: str
    total_pages: int
    total_detections: int
    detections: List[Dict[str, Any]]
    class_counts: Dict[str, int]
    dpi_used: int
    confidence_threshold: float
    
    def to_json(self, path: str, indent: int = 2) -> None:
        """Save results to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "total_detections": self.total_detections,
            "class_counts": self.class_counts,
            "dpi_used": self.dpi_used,
            "confidence_threshold": self.confidence_threshold,
            "detections": self.detections
        }


class TextAssociator:
    """Associates text words with detection bounding boxes."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the text associator.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
    
    def find_words_in_bbox(
        self,
        bbox: tuple,
        words: List[TextWord],
        margin: Optional[float] = None
    ) -> List[str]:
        """
        Find text words that fall within or near a bounding box.
        
        Args:
            bbox: (x0, y0, x1, y1) in PDF points
            words: List of TextWord objects
            margin: Margin to extend bbox (uses setting default if not provided)
        
        Returns:
            List of text strings found inside the bbox
        """
        margin = margin if margin is not None else self.settings.processing.text_margin_pts
        excluded = self.settings.processing.excluded_text_patterns
        
        # Expand bbox by margin
        x0, y0, x1, y1 = bbox
        x0m = x0 - margin
        y0m = y0 - margin
        x1m = x1 + margin
        y1m = y1 + margin
        
        found = []
        for word in words:
            # Use word center for containment check
            cx, cy = word.center
            
            # Check if center is inside expanded bbox
            if x0m <= cx <= x1m and y0m <= cy <= y1m:
                # Filter out excluded patterns
                text_upper = word.text.upper()
                if not any(pattern in text_upper for pattern in excluded):
                    found.append((word.x0, word.y0, word.text))
        
        # Sort by vertical then horizontal position (top-left sweep)
        found_sorted = sorted(found, key=lambda w: (w[1], w[0]))
        return [w[2] for w in found_sorted]


class PIDProcessor:
    """
    Main processing pipeline for P&ID PDF analysis.
    
    Orchestrates:
    1. PDF rendering to images
    2. YOLO object detection
    3. Text extraction from PDF
    4. Text-symbol association
    5. Result aggregation
    
    Usage:
        processor = PIDProcessor()
        results = processor.process("document.pdf")
        results.to_json("output.json")
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize the processor.
        
        Args:
            model_path: Path to YOLO model (uses default if not provided)
            settings: Application settings
        """
        self.settings = settings or Settings()
        self._model_path = model_path
        
        # Initialize components (lazy loading)
        self._renderer: Optional[PDFRenderer] = None
        self._detector: Optional[YOLODetector] = None
        self._extractor: Optional[TextExtractor] = None
        self._associator: Optional[TextAssociator] = None
    
    @property
    def renderer(self) -> PDFRenderer:
        """Get PDF renderer instance."""
        if self._renderer is None:
            self._renderer = PDFRenderer(self.settings)
        return self._renderer
    
    @property
    def detector(self) -> YOLODetector:
        """Get YOLO detector instance."""
        if self._detector is None:
            self._detector = YOLODetector(self._model_path, self.settings)
        return self._detector
    
    @property
    def extractor(self) -> TextExtractor:
        """Get text extractor instance."""
        if self._extractor is None:
            self._extractor = TextExtractor(self.settings)
        return self._extractor
    
    @property
    def associator(self) -> TextAssociator:
        """Get text associator instance."""
        if self._associator is None:
            self._associator = TextAssociator(self.settings)
        return self._associator
    
    def process(
        self,
        pdf_path: str,
        dpi: Optional[int] = None,
        confidence: Optional[float] = None,
        output_json: Optional[str] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
        show_progress: bool = True
    ) -> ProcessingResult:
        """
        Process a PDF document for P&ID symbol detection.
        
        Args:
            pdf_path: Path to input PDF
            dpi: Rendering DPI (auto-calculated if not provided)
            confidence: Detection confidence threshold
            output_json: Optional path to save results
            progress_callback: Callback function receiving progress percentage
            show_progress: Whether to show tqdm progress bar
        
        Returns:
            ProcessingResult with all detections
        """
        pdf_path = str(Path(pdf_path).resolve())
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        confidence = confidence or self.settings.processing.confidence_threshold
        
        # Log configuration
        self._log_config(pdf_path, dpi, confidence)
        
        all_detections = []
        class_counts: Dict[str, int] = {}
        actual_dpi = dpi
        
        try:
            # Open documents
            self.renderer.open(pdf_path)
            self.extractor.open(pdf_path)
            
            total_pages = self.renderer.page_count
            
            # Auto-calculate DPI if not specified
            if actual_dpi is None:
                actual_dpi = self.renderer.calculate_optimal_dpi()
                logger.info(f"Auto-calculated optimal DPI: {actual_dpi}")
            
            # Process each page
            page_iterator = range(1, total_pages + 1)
            if show_progress:
                page_iterator = tqdm(page_iterator, desc="Processing pages")
            
            for page_num in page_iterator:
                # Update progress
                if progress_callback:
                    progress = int((page_num - 1) / total_pages * 100)
                    progress_callback(progress)
                
                # Process single page
                page_detections = self._process_page(
                    page_num, actual_dpi, confidence
                )
                
                # Aggregate results
                for det_dict in page_detections:
                    det_dict["page"] = page_num
                    all_detections.append(det_dict)
                    
                    # Update class counts
                    cls_name = det_dict.get("class_name", "Unknown")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                
                # Final progress update
                if progress_callback:
                    progress = int(page_num / total_pages * 100)
                    progress_callback(progress)
            
            # Create result
            result = ProcessingResult(
                pdf_path=pdf_path,
                total_pages=total_pages,
                total_detections=len(all_detections),
                detections=all_detections,
                class_counts=class_counts,
                dpi_used=actual_dpi,
                confidence_threshold=confidence
            )
            
            # Save to JSON if requested
            if output_json:
                result.to_json(output_json)
                logger.info(f"Results saved to: {output_json}")
            
            # Log summary
            self._log_summary(result)
            
            return result
            
        finally:
            # Clean up
            self.renderer.close()
            self.extractor.close()
    
    def _process_page(
        self,
        page_number: int,
        dpi: int,
        confidence: float
    ) -> List[Dict[str, Any]]:
        """
        Process a single PDF page.
        
        Args:
            page_number: 1-indexed page number
            dpi: Rendering DPI
            confidence: Detection confidence threshold
        
        Returns:
            List of detection dictionaries for this page
        """
        # Render page
        rendered = self.renderer.render_page(page_number, dpi)
        
        if page_number == 1:
            logger.info(f"Rendered page 1: {rendered.width}×{rendered.height}px at {dpi} DPI")
        
        # Run detection
        detections = self.detector.detect(rendered.image, confidence)
        
        if not detections:
            if page_number == 1:
                logger.info(f"No detections on page 1 (try lowering confidence: {confidence})")
            return []
        
        # Extract text
        words = self.extractor.extract_page_words(page_number)
        
        if page_number == 1:
            logger.info(f"Page 1: {len(detections)} detections, {len(words)} text elements")
        
        # Create coordinate transformer
        transformer = CoordinateTransformer.from_rendered_page(
            rendered, 
            flip_y=self.settings.processing.flip_y
        )
        
        # Process each detection
        result_list = []
        for det in detections:
            # Convert bbox to PDF coordinates
            pdf_bbox = transformer.image_to_pdf_bbox(det.bbox_pixels)
            
            # Find associated text
            texts = self.associator.find_words_in_bbox(pdf_bbox, words)
            
            # Build result dictionary
            result_list.append({
                "class_name": det.class_name,
                "confidence": round(det.confidence, 4),
                "bbox_pdf": [round(v, 3) for v in pdf_bbox],
                "bbox_pixels": [round(v, 1) for v in det.bbox_pixels],
                "texts_inside": texts
            })
        
        return result_list
    
    def _log_config(
        self, 
        pdf_path: str, 
        dpi: Optional[int], 
        confidence: float
    ) -> None:
        """Log processing configuration."""
        logger.info("=" * 60)
        logger.info("PDF PROCESSING CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"PDF: {Path(pdf_path).name}")
        logger.info(f"Rendering DPI: {dpi or 'auto'}")
        logger.info(f"Confidence threshold: {confidence}")
        logger.info(f"YOLO model: {self.detector.model_path}")
        logger.info(f"Text extraction: {self.extractor.extraction_method}")
    
    def _log_summary(self, result: ProcessingResult) -> None:
        """Log processing summary."""
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total pages: {result.total_pages}")
        logger.info(f"Total detections: {result.total_detections}")
        if result.total_pages > 0:
            logger.info(f"Average per page: {result.total_detections / result.total_pages:.1f}")
        
        if result.class_counts:
            logger.info("\nDetections by class:")
            for cls, count in sorted(
                result.class_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                logger.info(f"  {cls}: {count}")
        
        logger.info("=" * 60)


def process_pdf_with_yolo(
    pdf_path: str,
    yolo_model_path: str,
    out_json_path: Optional[str] = None,
    dpi: Optional[int] = None,
    conf: float = 0.25,
    progress_callback: Optional[Callable[[int], None]] = None
) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.
    
    Args:
        pdf_path: Path to input PDF
        yolo_model_path: Path to YOLO model weights
        out_json_path: Path to save JSON results
        dpi: Rendering DPI
        conf: Confidence threshold
        progress_callback: Progress callback function
    
    Returns:
        List of detection dictionaries
    """
    processor = PIDProcessor(model_path=yolo_model_path)
    result = processor.process(
        pdf_path=pdf_path,
        dpi=dpi,
        confidence=conf,
        output_json=out_json_path,
        progress_callback=progress_callback,
        show_progress=False
    )
    return result.detections
