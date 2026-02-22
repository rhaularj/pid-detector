"""
Coordinate transformation utilities for PDF to image mapping.

Handles the conversion between image pixel coordinates (from YOLO detection)
and PDF point coordinates (for text association).
"""

from typing import Tuple, List, Optional
import warnings

from ..config import get_logger, Settings

logger = get_logger(__name__)


class CoordinateTransformer:
    """
    Handles coordinate transformations between image and PDF spaces.
    
    Coordinate systems:
    - PDF points: 1 point = 1/72 inch, origin at top-left
    - Image pixels: Origin at top-left, size depends on rendering DPI
    
    Relationship: 
        pixels = points * (dpi / 72)
        points = pixels * (72 / dpi)
    
    Usage:
        transformer = CoordinateTransformer(
            page_width_pts=612, page_height_pts=792,
            image_width_px=2550, image_height_px=3300,
            dpi=300
        )
        pdf_bbox = transformer.image_to_pdf_bbox((100, 100, 200, 200))
    """
    
    def __init__(
        self,
        page_width_pts: float,
        page_height_pts: float,
        image_width_px: int,
        image_height_px: int,
        dpi: int,
        flip_y: bool = False
    ):
        """
        Initialize the coordinate transformer.
        
        Args:
            page_width_pts: PDF page width in points
            page_height_pts: PDF page height in points  
            image_width_px: Rendered image width in pixels
            image_height_px: Rendered image height in pixels
            dpi: DPI used for rendering
            flip_y: Whether to flip Y coordinates (rarely needed)
        """
        self.page_width_pts = page_width_pts
        self.page_height_pts = page_height_pts
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        self.dpi = dpi
        self.flip_y = flip_y
        
        # Calculate scale factors
        self._scale_x = page_width_pts / image_width_px
        self._scale_y = page_height_pts / image_height_px
        
        # Validate scale factors
        self._validate_scale()
    
    def _validate_scale(self) -> None:
        """Validate that scale factors match expected DPI."""
        expected_scale = 72.0 / self.dpi
        tolerance = 0.01
        
        if abs(self._scale_x - expected_scale) > tolerance:
            warnings.warn(
                f"X scale factor mismatch! Expected ~{expected_scale:.3f} for {self.dpi} DPI, "
                f"but got {self._scale_x:.3f}. Image may not have been rendered at specified DPI."
            )
        
        if abs(self._scale_y - expected_scale) > tolerance:
            warnings.warn(
                f"Y scale factor mismatch! Expected ~{expected_scale:.3f} for {self.dpi} DPI, "
                f"but got {self._scale_y:.3f}. Image may not have been rendered at specified DPI."
            )
    
    @property
    def scale_x(self) -> float:
        """Get X scale factor (points per pixel)."""
        return self._scale_x
    
    @property
    def scale_y(self) -> float:
        """Get Y scale factor (points per pixel)."""
        return self._scale_y
    
    @classmethod
    def from_rendered_page(
        cls, 
        rendered_page, 
        flip_y: bool = False
    ) -> 'CoordinateTransformer':
        """
        Create transformer from a RenderedPage object.
        
        Args:
            rendered_page: RenderedPage from PDFRenderer
            flip_y: Whether to flip Y coordinates
        
        Returns:
            CoordinateTransformer instance
        """
        return cls(
            page_width_pts=rendered_page.page_width_pts,
            page_height_pts=rendered_page.page_height_pts,
            image_width_px=rendered_page.width,
            image_height_px=rendered_page.height,
            dpi=rendered_page.dpi,
            flip_y=flip_y
        )
    
    def image_to_pdf_point(
        self, 
        x_px: float, 
        y_px: float
    ) -> Tuple[float, float]:
        """
        Convert a single point from image to PDF coordinates.
        
        Args:
            x_px: X coordinate in pixels
            y_px: Y coordinate in pixels
        
        Returns:
            (x_pt, y_pt) in PDF points
        """
        x_pt = x_px * self._scale_x
        y_pt = y_px * self._scale_y
        
        if self.flip_y:
            y_pt = self.page_height_pts - y_pt
        
        return (x_pt, y_pt)
    
    def pdf_to_image_point(
        self, 
        x_pt: float, 
        y_pt: float
    ) -> Tuple[float, float]:
        """
        Convert a single point from PDF to image coordinates.
        
        Args:
            x_pt: X coordinate in points
            y_pt: Y coordinate in points
        
        Returns:
            (x_px, y_px) in pixels
        """
        if self.flip_y:
            y_pt = self.page_height_pts - y_pt
        
        x_px = x_pt / self._scale_x
        y_px = y_pt / self._scale_y
        
        return (x_px, y_px)
    
    def image_to_pdf_bbox(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from image to PDF coordinates.
        
        Args:
            bbox: (x0_px, y0_px, x1_px, y1_px) in image pixels
        
        Returns:
            (x0_pt, y0_pt, x1_pt, y1_pt) in PDF points
        """
        x0_px, y0_px, x1_px, y1_px = bbox
        
        x0_pt = x0_px * self._scale_x
        x1_pt = x1_px * self._scale_x
        y0_pt = y0_px * self._scale_y
        y1_pt = y1_px * self._scale_y
        
        if self.flip_y:
            y0_pt, y1_pt = self.page_height_pts - y1_pt, self.page_height_pts - y0_pt
        
        # Normalize to ensure x0 < x1 and y0 < y1
        return (
            min(x0_pt, x1_pt),
            min(y0_pt, y1_pt),
            max(x0_pt, x1_pt),
            max(y0_pt, y1_pt)
        )
    
    def pdf_to_image_bbox(
        self, 
        bbox: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Convert bounding box from PDF to image coordinates.
        
        Args:
            bbox: (x0_pt, y0_pt, x1_pt, y1_pt) in PDF points
        
        Returns:
            (x0_px, y0_px, x1_px, y1_px) in image pixels
        """
        x0_pt, y0_pt, x1_pt, y1_pt = bbox
        
        if self.flip_y:
            y0_pt, y1_pt = self.page_height_pts - y1_pt, self.page_height_pts - y0_pt
        
        x0_px = x0_pt / self._scale_x
        x1_px = x1_pt / self._scale_x
        y0_px = y0_pt / self._scale_y
        y1_px = y1_pt / self._scale_y
        
        return (
            min(x0_px, x1_px),
            min(y0_px, y1_px),
            max(x0_px, x1_px),
            max(y0_px, y1_px)
        )
    
    @staticmethod
    def point_in_bbox(
        bbox: Tuple[float, float, float, float],
        x: float,
        y: float
    ) -> bool:
        """
        Check if a point is inside a bounding box.
        
        Args:
            bbox: (x0, y0, x1, y1) bounding box
            x: X coordinate of point
            y: Y coordinate of point
        
        Returns:
            True if point is inside bbox
        """
        x0, y0, x1, y1 = bbox
        return (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1)
    
    @staticmethod
    def expand_bbox(
        bbox: Tuple[float, float, float, float],
        margin: float
    ) -> Tuple[float, float, float, float]:
        """
        Expand a bounding box by a margin on all sides.
        
        Args:
            bbox: (x0, y0, x1, y1) bounding box
            margin: Distance to expand on each side
        
        Returns:
            Expanded bounding box
        """
        x0, y0, x1, y1 = bbox
        return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
