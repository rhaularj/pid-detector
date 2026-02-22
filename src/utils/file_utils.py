"""
File handling utilities for PidDetector.

Provides common file operations, path validation, and export functionality.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from ..config import get_logger, Settings

logger = get_logger(__name__)


class FileHandler:
    """
    Utility class for file operations.
    
    Features:
    - Path validation
    - JSON/Excel export
    - Directory management
    - Temp file handling
    
    Usage:
        handler = FileHandler()
        if handler.validate_pdf("document.pdf"):
            output = handler.get_output_path("document.pdf", "xlsx")
    """
    
    SUPPORTED_INPUT_EXTENSIONS = {'.pdf'}
    SUPPORTED_OUTPUT_EXTENSIONS = {'.json', '.xlsx', '.csv'}
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the file handler.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
    
    def validate_pdf(self, path: Union[str, Path]) -> bool:
        """
        Validate that a file is a valid PDF.
        
        Args:
            path: Path to file
        
        Returns:
            True if file is valid PDF
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"File does not exist: {path}")
            return False
        
        if not path.is_file():
            logger.warning(f"Path is not a file: {path}")
            return False
        
        if path.suffix.lower() not in self.SUPPORTED_INPUT_EXTENSIONS:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return False
        
        # Basic PDF header check
        try:
            with open(path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF'):
                    logger.warning(f"File does not appear to be a valid PDF: {path}")
                    return False
        except IOError as e:
            logger.error(f"Could not read file: {e}")
            return False
        
        return True
    
    def validate_model(self, path: Union[str, Path]) -> bool:
        """
        Validate that a YOLO model file exists.
        
        Args:
            path: Path to model file
        
        Returns:
            True if model file exists
        """
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Model file does not exist: {path}")
            return False
        
        if path.suffix.lower() != '.pt':
            logger.warning(f"Expected .pt file, got: {path.suffix}")
            return False
        
        return True
    
    def get_output_path(
        self,
        input_path: Union[str, Path],
        extension: str,
        suffix: str = "",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate an output path based on input file.
        
        Args:
            input_path: Input file path
            extension: Output file extension (e.g., 'xlsx', 'json')
            suffix: Optional suffix to add to filename
            output_dir: Output directory (uses settings default if not provided)
        
        Returns:
            Output file path
        """
        input_path = Path(input_path)
        output_dir = output_dir or self.settings.paths.output_dir
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build output filename
        stem = input_path.stem
        if suffix:
            stem = f"{stem}_{suffix}"
        
        # Ensure extension has dot
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        return output_dir / f"{stem}{extension}"
    
    def get_timestamped_path(
        self,
        base_name: str,
        extension: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate a timestamped output path.
        
        Args:
            base_name: Base filename without extension
            extension: File extension
            output_dir: Output directory
        
        Returns:
            Timestamped file path
        """
        output_dir = output_dir or self.settings.paths.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        return output_dir / f"{base_name}_{timestamp}{extension}"
    
    @staticmethod
    def save_json(
        data: Union[List, Dict],
        path: Union[str, Path],
        indent: int = 2
    ) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            path: Output file path
            indent: JSON indentation
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        logger.info(f"Saved JSON to: {path}")
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Union[List, Dict]:
        """
        Load data from JSON file.
        
        Args:
            path: Input file path
        
        Returns:
            Loaded data
        """
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_excel(
        data: List[Dict[str, Any]],
        path: Union[str, Path],
        sheet_name: str = "Detections"
    ) -> None:
        """
        Save detection data to Excel file.
        
        Args:
            data: List of detection dictionaries
            path: Output file path
            sheet_name: Excel sheet name
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for Excel export. Install with: pip install pandas")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(data)
        
        # Flatten nested fields for Excel
        if 'texts_inside' in df.columns:
            df['texts_inside'] = df['texts_inside'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        if 'bbox_pdf' in df.columns:
            df['bbox_pdf'] = df['bbox_pdf'].apply(
                lambda x: str(x) if isinstance(x, list) else str(x)
            )
        
        if 'bbox_pixels' in df.columns:
            df['bbox_pixels'] = df['bbox_pixels'].apply(
                lambda x: str(x) if isinstance(x, list) else str(x)
            )
        
        df.to_excel(path, sheet_name=sheet_name, index=False)
        logger.info(f"Saved Excel to: {path}")
    
    def clean_path(self, path: str) -> str:
        """
        Clean a file path (remove quotes, braces, etc.).
        
        Args:
            path: Input path string
        
        Returns:
            Cleaned path string
        """
        # Remove common wrappers from drag-drop
        path = path.strip()
        path = path.strip('{}')
        path = path.strip('"')
        path = path.strip("'")
        return path
    
    def ensure_directory(self, path: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            path: Directory path
        
        Returns:
            Path object for the directory
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
