"""
PDF rendering module for converting PDF pages to images.

This module handles the rendering of PDF pages to images suitable for 
YOLO object detection, with optimal DPI calculation for best results.
"""

import numpy as np
import cv2
import fitz  # PyMuPDF
from typing import Tuple, Optional
from dataclasses import dataclass

from ..config import get_logger, Settings

logger = get_logger(__name__)


@dataclass
class RenderedPage:
    """Container for a rendered PDF page and its metadata."""
    
    image: np.ndarray  # BGR image
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    dpi: int  # DPI used for rendering
    page_number: int  # 1-indexed page number
    page_width_pts: float  # Original page width in points
    page_height_pts: float  # Original page height in points
    
    @property
    def aspect_ratio(self) -> float:
        """Get the image aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get image dimensions as (width, height)."""
        return (self.width, self.height)
    
    @property
    def page_size_pts(self) -> Tuple[float, float]:
        """Get original page dimensions in points as (width, height)."""
        return (self.page_width_pts, self.page_height_pts)


class PDFRenderer:
    """
    Handles PDF page rendering to images for object detection.
    
    Features:
    - Automatic optimal DPI calculation for YOLO inference
    - Configurable target image size
    - Memory-efficient page-by-page rendering
    
    Usage:
        renderer = PDFRenderer()
        with renderer.open("document.pdf") as doc:
            for page in renderer.render_pages():
                process(page.image)
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the PDF renderer.
        
        Args:
            settings: Application settings (uses defaults if not provided)
        """
        self.settings = settings or Settings()
        self._document: Optional[fitz.Document] = None
        self._pdf_path: Optional[str] = None
    
    def open(self, pdf_path: str) -> 'PDFRenderer':
        """
        Open a PDF document for rendering.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            self for context manager usage
        """
        self.close()  # Close any existing document
        self._pdf_path = pdf_path
        self._document = fitz.open(pdf_path)
        logger.info(f"Opened PDF: {pdf_path} ({self._document.page_count} pages)")
        return self
    
    def close(self) -> None:
        """Close the current PDF document."""
        if self._document:
            self._document.close()
            self._document = None
            self._pdf_path = None
    
    def __enter__(self) -> 'PDFRenderer':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        return self._document.page_count if self._document else 0
    
    @property
    def is_open(self) -> bool:
        """Check if a document is currently open."""
        return self._document is not None
    
    def calculate_optimal_dpi(
        self, 
        page_index: int = 0, 
        target_size: Optional[int] = None
    ) -> int:
        """
        Calculate optimal DPI to render PDF page close to YOLO's training size.
        
        Args:
            page_index: Page index (0-based) to calculate DPI for
            target_size: Target pixel size for the longer edge (default: 1280)
        
        Returns:
            Optimal DPI value
        
        Note:
            YOLO will resize to exactly 1280px during inference, but rendering at 
            a similar size improves coordinate mapping accuracy.
        """
        if not self._document:
            raise RuntimeError("No PDF document is open")
        
        target_size = target_size or self.settings.processing.target_size
        page = self._document.load_page(page_index)
        
        page_w = page.rect.width  # in points (1/72 inch)
        page_h = page.rect.height
        
        # Determine longer edge
        longer_edge_pts = max(page_w, page_h)
        
        # Calculate DPI to make longer edge approximately target_size pixels
        # DPI = (target_pixels * 72) / points
        optimal_dpi = (target_size * 72.0) / longer_edge_pts
        
        # Round to nearest 10 for cleaner values
        optimal_dpi = round(optimal_dpi / 10) * 10
        
        # Clamp between reasonable bounds
        min_dpi = self.settings.processing.min_dpi
        max_dpi = self.settings.processing.max_dpi
        optimal_dpi = max(min_dpi, min(max_dpi, optimal_dpi))
        
        logger.debug(f"Calculated optimal DPI: {optimal_dpi} for target size {target_size}px")
        return int(optimal_dpi)
    
    def render_page(
        self, 
        page_number: int, 
        dpi: Optional[int] = None
    ) -> RenderedPage:
        """
        Render a single PDF page to an image.
        
        Args:
            page_number: 1-indexed page number
            dpi: Rendering DPI (auto-calculated if not provided)
        
        Returns:
            RenderedPage containing the image and metadata
        """
        if not self._document:
            raise RuntimeError("No PDF document is open")
        
        page_index = page_number - 1
        if page_index < 0 or page_index >= self._document.page_count:
            raise ValueError(f"Invalid page number: {page_number}")
        
        page = self._document.load_page(page_index)
        
        # Calculate DPI if not specified
        if dpi is None:
            dpi = self.calculate_optimal_dpi(page_index)
        
        # Render the page
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # Convert to BGR for OpenCV compatibility
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return RenderedPage(
            image=img,
            width=pix.width,
            height=pix.height,
            dpi=dpi,
            page_number=page_number,
            page_width_pts=page.rect.width,
            page_height_pts=page.rect.height
        )
    
    def render_all_pages(
        self, 
        dpi: Optional[int] = None
    ):
        """
        Generator that yields all pages as RenderedPage objects.
        
        Args:
            dpi: Rendering DPI (auto-calculated per page if not provided)
        
        Yields:
            RenderedPage for each page in the document
        """
        if not self._document:
            raise RuntimeError("No PDF document is open")
        
        for page_num in range(1, self._document.page_count + 1):
            yield self.render_page(page_num, dpi)
    
    def get_page_rect(self, page_number: int) -> fitz.Rect:
        """
        Get the rectangle (dimensions) of a specific page.
        
        Args:
            page_number: 1-indexed page number
        
        Returns:
            fitz.Rect with page dimensions in points
        """
        if not self._document:
            raise RuntimeError("No PDF document is open")
        
        page = self._document.load_page(page_number - 1)
        return page.rect
