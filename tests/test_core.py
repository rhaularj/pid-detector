"""
Tests for the core processor module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.config import Settings, ProcessingConfig
from src.core.pdf_renderer import PDFRenderer, RenderedPage
from src.core.text_extractor import TextExtractor, TextWord
from src.core.coordinate_transformer import CoordinateTransformer
from src.core.detector import YOLODetector, Detection, PageDetections


class TestCoordinateTransformer:
    """Tests for coordinate transformation."""
    
    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        transformer = CoordinateTransformer(
            page_width_pts=612,
            page_height_pts=792,
            image_width_px=2550,
            image_height_px=3300,
            dpi=300
        )
        
        assert transformer.page_width_pts == 612
        assert transformer.page_height_pts == 792
        assert transformer.image_width_px == 2550
        assert transformer.image_height_px == 3300
        assert transformer.dpi == 300
    
    def test_scale_factors(self):
        """Test scale factor calculation."""
        transformer = CoordinateTransformer(
            page_width_pts=612,
            page_height_pts=792,
            image_width_px=2550,
            image_height_px=3300,
            dpi=300
        )
        
        # At 300 DPI, scale should be approximately 72/300 = 0.24
        expected_scale = 72.0 / 300
        assert abs(transformer.scale_x - expected_scale) < 0.01
        assert abs(transformer.scale_y - expected_scale) < 0.01
    
    def test_image_to_pdf_point(self):
        """Test single point conversion from image to PDF."""
        transformer = CoordinateTransformer(
            page_width_pts=612,
            page_height_pts=792,
            image_width_px=2550,
            image_height_px=3300,
            dpi=300
        )
        
        # Image center
        x_px, y_px = 1275, 1650  # Center of 2550x3300 image
        x_pt, y_pt = transformer.image_to_pdf_point(x_px, y_px)
        
        # Should be near center of page
        assert abs(x_pt - 306) < 1  # 612/2
        assert abs(y_pt - 396) < 1  # 792/2
    
    def test_pdf_to_image_point(self):
        """Test single point conversion from PDF to image."""
        transformer = CoordinateTransformer(
            page_width_pts=612,
            page_height_pts=792,
            image_width_px=2550,
            image_height_px=3300,
            dpi=300
        )
        
        # PDF center
        x_pt, y_pt = 306, 396
        x_px, y_px = transformer.pdf_to_image_point(x_pt, y_pt)
        
        # Should be near center of image
        assert abs(x_px - 1275) < 5
        assert abs(y_px - 1650) < 5
    
    def test_image_to_pdf_bbox(self):
        """Test bounding box conversion."""
        transformer = CoordinateTransformer(
            page_width_pts=612,
            page_height_pts=792,
            image_width_px=2550,
            image_height_px=3300,
            dpi=300
        )
        
        # Test bbox at top-left corner
        img_bbox = (0, 0, 100, 100)
        pdf_bbox = transformer.image_to_pdf_bbox(img_bbox)
        
        # Should be small values near origin
        assert all(v >= 0 for v in pdf_bbox)
        assert pdf_bbox[0] < pdf_bbox[2]  # x0 < x1
        assert pdf_bbox[1] < pdf_bbox[3]  # y0 < y1
    
    def test_point_in_bbox(self):
        """Test point-in-bbox check."""
        bbox = (10, 10, 100, 100)
        
        # Inside
        assert CoordinateTransformer.point_in_bbox(bbox, 50, 50)
        
        # Outside
        assert not CoordinateTransformer.point_in_bbox(bbox, 0, 0)
        assert not CoordinateTransformer.point_in_bbox(bbox, 150, 150)
        
        # Edge
        assert CoordinateTransformer.point_in_bbox(bbox, 10, 10)
        assert CoordinateTransformer.point_in_bbox(bbox, 100, 100)
    
    def test_expand_bbox(self):
        """Test bbox expansion."""
        bbox = (10, 10, 100, 100)
        expanded = CoordinateTransformer.expand_bbox(bbox, 5)
        
        assert expanded == (5, 5, 105, 105)


class TestTextWord:
    """Tests for TextWord named tuple."""
    
    def test_center_property(self):
        """Test center calculation."""
        word = TextWord(x0=10, y0=10, x1=90, y1=30, text="test")
        cx, cy = word.center
        
        assert cx == 50  # (10+90)/2
        assert cy == 20  # (10+30)/2
    
    def test_bbox_property(self):
        """Test bbox property."""
        word = TextWord(x0=10, y0=20, x1=30, y1=40, text="test")
        
        assert word.bbox == (10, 20, 30, 40)


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_center_property(self):
        """Test detection center calculation."""
        det = Detection(
            class_id=0,
            class_name="Valve",
            confidence=0.95,
            bbox_pixels=(100, 100, 200, 200)
        )
        
        cx, cy = det.center
        assert cx == 150
        assert cy == 150
    
    def test_dimensions(self):
        """Test width/height/area calculations."""
        det = Detection(
            class_id=0,
            class_name="Valve",
            confidence=0.95,
            bbox_pixels=(100, 100, 150, 200)
        )
        
        assert det.width == 50
        assert det.height == 100
        assert det.area == 5000
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        det = Detection(
            class_id=0,
            class_name="Valve",
            confidence=0.9567,
            bbox_pixels=(100.12, 100.34, 200.56, 200.78)
        )
        
        d = det.to_dict()
        
        assert d["class_name"] == "Valve"
        assert d["confidence"] == 0.9567
        assert len(d["bbox_pixels"]) == 4


class TestSettings:
    """Tests for Settings configuration."""
    
    def test_singleton(self):
        """Test that Settings is a singleton."""
        Settings.reset()
        s1 = Settings()
        s2 = Settings()
        
        assert s1 is s2
    
    def test_default_values(self):
        """Test default configuration values."""
        Settings.reset()
        settings = Settings()
        
        assert settings.processing.default_dpi == 300
        assert settings.processing.confidence_threshold == 0.25
        assert settings.processing.text_margin_pts == 6.0
    
    def test_update_processing(self):
        """Test updating processing settings."""
        Settings.reset()
        settings = Settings()
        
        original_dpi = settings.processing.default_dpi
        settings.update_processing(default_dpi=600)
        
        assert settings.processing.default_dpi == 600
        
        # Reset for other tests
        settings.update_processing(default_dpi=original_dpi)


class TestProcessingConfig:
    """Tests for ProcessingConfig."""
    
    def test_default_values(self):
        """Test default configuration."""
        config = ProcessingConfig()
        
        assert config.default_dpi == 300
        assert config.min_dpi == 150
        assert config.max_dpi == 600
        assert config.target_size == 1280
        assert config.confidence_threshold == 0.25


# Integration tests (require mocking)
class TestPDFRendererMocked:
    """Mocked tests for PDFRenderer."""
    
    @patch('src.core.pdf_renderer.fitz')
    def test_calculate_optimal_dpi(self, mock_fitz):
        """Test optimal DPI calculation."""
        # Setup mock page
        mock_page = MagicMock()
        mock_page.rect.width = 612  # 8.5 inches at 72 DPI
        mock_page.rect.height = 792  # 11 inches at 72 DPI
        
        mock_doc = MagicMock()
        mock_doc.load_page.return_value = mock_page
        mock_doc.page_count = 1
        
        mock_fitz.open.return_value = mock_doc
        
        renderer = PDFRenderer()
        renderer._document = mock_doc
        
        dpi = renderer.calculate_optimal_dpi(0, target_size=1280)
        
        # For a letter-size page, optimal DPI should be calculated
        assert isinstance(dpi, int)
        assert 150 <= dpi <= 600


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
