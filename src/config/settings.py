"""
Application settings and configuration management.

This module provides centralized configuration for the PidDetector application,
including processing parameters, paths, and default values.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class ProcessingConfig:
    """Configuration for PDF processing parameters."""
    
    # PDF rendering
    default_dpi: int = 300
    min_dpi: int = 150
    max_dpi: int = 600
    target_size: int = 1280  # YOLO inference size
    
    # YOLO detection
    confidence_threshold: float = 0.25
    min_confidence: float = 0.1
    max_confidence: float = 0.9
    
    # Text extraction
    text_margin_pts: float = 6.0  # Points margin for text association
    flip_y: bool = False  # Y-axis flip for coordinate mapping
    
    # OCR settings
    use_ocr: bool = False
    ocr_method: str = "easyocr"  # Options: "tesseract", "easyocr"
    
    # Debug settings
    debug_bbox: bool = False
    save_visualizations: bool = False
    
    # Excluded text patterns (uppercase)
    excluded_text_patterns: List[str] = field(
        default_factory=lambda: ['H', 'L', 'O', 'C', 'NOTE', 'PUSH', 'BUTTON']
    )


@dataclass
class PathConfig:
    """Configuration for application paths."""
    
    # Base directories (computed from project root)
    _project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    @property
    def project_root(self) -> Path:
        return self._project_root
    
    @property
    def models_dir(self) -> Path:
        return self._project_root / "models"
    
    @property
    def data_dir(self) -> Path:
        return self._project_root / "data"
    
    @property
    def runs_dir(self) -> Path:
        return self._project_root / "runs"
    
    @property
    def output_dir(self) -> Path:
        return self._project_root / "output"
    
    @property
    def default_model_path(self) -> Path:
        """Returns the best available trained model."""
        # Try to find the latest trained model
        train_weights = self.runs_dir / "detect"
        if train_weights.exists():
            # Find latest training run
            train_dirs = sorted(
                [d for d in train_weights.iterdir() if d.is_dir() and d.name.startswith("train")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            for train_dir in train_dirs:
                best_pt = train_dir / "weights" / "best.pt"
                if best_pt.exists():
                    return best_pt
        
        # Fallback to models directory
        models_best = self.models_dir / "best.pt"
        if models_best.exists():
            return models_best
        
        return self.runs_dir / "detect" / "train27" / "weights" / "best.pt"
    
    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [self.models_dir, self.data_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass  
class GUIConfig:
    """Configuration for GUI appearance and behavior."""
    
    # Window settings
    window_title: str = "P&ID Detector"
    window_geometry: str = "1000x700"
    min_width: int = 800
    min_height: int = 600
    
    # Colors (using modern UI palette)
    primary_color: str = "#009DF0"
    secondary_color: str = "#6c757d"
    background_color: str = "#f8f9fa"
    card_background: str = "#ffffff"
    text_primary: str = "#212529"
    text_secondary: str = "#495057"
    text_muted: str = "#6c757d"
    
    # Fonts
    font_family: str = "Segoe UI"
    title_font_size: int = 24
    subtitle_font_size: int = 11
    body_font_size: int = 10
    small_font_size: int = 9
    
    # Supported file types
    supported_extensions: List[str] = field(
        default_factory=lambda: [".pdf"]
    )


class Settings:
    """
    Central settings manager for the application.
    
    Usage:
        settings = Settings()
        print(settings.processing.default_dpi)
        print(settings.paths.default_model_path)
    """
    
    _instance: Optional['Settings'] = None
    
    def __new__(cls) -> 'Settings':
        """Singleton pattern to ensure consistent settings across the app."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize all configuration objects."""
        self._processing = ProcessingConfig()
        self._paths = PathConfig()
        self._gui = GUIConfig()
    
    @property
    def processing(self) -> ProcessingConfig:
        """Get processing configuration."""
        return self._processing
    
    @property
    def paths(self) -> PathConfig:
        """Get path configuration."""
        return self._paths
    
    @property
    def gui(self) -> GUIConfig:
        """Get GUI configuration."""
        return self._gui
    
    def update_processing(self, **kwargs) -> None:
        """Update processing settings with new values."""
        for key, value in kwargs.items():
            if hasattr(self._processing, key):
                setattr(self._processing, key, value)
    
    @classmethod
    def reset(cls) -> None:
        """Reset settings to defaults (useful for testing)."""
        cls._instance = None


# Convenience function for quick access
def get_settings() -> Settings:
    """Get the global settings instance."""
    return Settings()
