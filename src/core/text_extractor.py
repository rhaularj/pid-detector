"""
Text extraction module for PDF documents.

Supports multiple extraction methods with fallback strategies for robust
text extraction from P&ID diagrams.
"""

from typing import List, Tuple, Optional, NamedTuple
from pathlib import Path
import fitz  # PyMuPDF

from ..config import get_logger, Settings

logger = get_logger(__name__)

# Try to import pdfplumber for enhanced extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Install with: pip install pdfplumber")


class TextWord(NamedTuple):
    """Represents an extracted word with its bounding box."""
    
    x0: float  # Left x coordinate (points)
    y0: float  # Top y coordinate (points)
    x1: float  # Right x coordinate (points)
    y1: float  # Bottom y coordinate (points)
    text: str  # Word text
    block_no: int = 0
    line_no: int = 0
    word_no: int = 0
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the word bbox."""
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get the bounding box as (x0, y0, x1, y1)."""
        return (self.x0, self.y0, self.x1, self.y1)


class TextExtractor:
    """
    Extracts text from PDF documents with multiple fallback methods.
    
    Extraction methods (in order of preference):
    1. pdfplumber - Enhanced positioning for engineering drawings
    2. PyMuPDF words - Standard word extraction
    3. PyMuPDF dict - Detailed block/span extraction
    
    Usage:
        extractor = TextExtractor()
        with extractor.open("document.pdf") as doc:
            words = extractor.extract_page_words(1)
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the text extractor.
        
        Args:
            settings: Application settings (uses defaults if not provided)
        """
        self.settings = settings or Settings()
        self._fitz_doc: Optional[fitz.Document] = None
        self._plumber_doc = None
        self._pdf_path: Optional[str] = None
    
    def open(self, pdf_path: str) -> 'TextExtractor':
        """
        Open a PDF document for text extraction.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            self for context manager usage
        """
        self.close()
        self._pdf_path = pdf_path
        self._fitz_doc = fitz.open(pdf_path)
        
        # Try to open with pdfplumber for enhanced extraction
        if PDFPLUMBER_AVAILABLE:
            try:
                self._plumber_doc = pdfplumber.open(pdf_path)
                logger.debug("Using pdfplumber for text extraction")
            except Exception as e:
                logger.warning(f"pdfplumber failed to open PDF: {e}")
                self._plumber_doc = None
        
        return self
    
    def close(self) -> None:
        """Close all open documents."""
        if self._fitz_doc:
            self._fitz_doc.close()
            self._fitz_doc = None
        
        if self._plumber_doc:
            self._plumber_doc.close()
            self._plumber_doc = None
        
        self._pdf_path = None
    
    def __enter__(self) -> 'TextExtractor':
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    @property
    def page_count(self) -> int:
        """Get the number of pages in the document."""
        return self._fitz_doc.page_count if self._fitz_doc else 0
    
    @property
    def is_open(self) -> bool:
        """Check if a document is currently open."""
        return self._fitz_doc is not None
    
    @property
    def extraction_method(self) -> str:
        """Get the current extraction method being used."""
        if self._plumber_doc:
            return "pdfplumber"
        return "pymupdf"
    
    def extract_page_words(self, page_number: int) -> List[TextWord]:
        """
        Extract all words from a PDF page.
        
        Args:
            page_number: 1-indexed page number
        
        Returns:
            List of TextWord objects with bounding boxes
        """
        if not self._fitz_doc:
            raise RuntimeError("No PDF document is open")
        
        # Try pdfplumber first (better for engineering drawings)
        if self._plumber_doc:
            words = self._extract_with_pdfplumber(page_number)
            if words:
                return words
        
        # Fallback to PyMuPDF
        return self._extract_with_pymupdf(page_number)
    
    def _extract_with_pdfplumber(self, page_number: int) -> List[TextWord]:
        """
        Extract text using pdfplumber (better for complex layouts).
        
        Args:
            page_number: 1-indexed page number
        
        Returns:
            List of TextWord objects
        """
        words = []
        
        try:
            # pdfplumber uses 0-indexed pages
            page = self._plumber_doc.pages[page_number - 1]
            
            # Extract words with their bounding boxes
            extracted_words = page.extract_words()
            
            for word_dict in extracted_words:
                text = word_dict.get('text', '').strip()
                if not text:
                    continue
                
                words.append(TextWord(
                    x0=word_dict['x0'],
                    y0=word_dict['top'],
                    x1=word_dict['x1'],
                    y1=word_dict['bottom'],
                    text=text
                ))
            
            logger.debug(f"pdfplumber extracted {len(words)} words from page {page_number}")
            return words
            
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for page {page_number}: {e}")
            return []
    
    def _extract_with_pymupdf(self, page_number: int) -> List[TextWord]:
        """
        Extract text using PyMuPDF with multiple fallback methods.
        
        Args:
            page_number: 1-indexed page number
        
        Returns:
            List of TextWord objects
        """
        page = self._fitz_doc.load_page(page_number - 1)
        
        # Method 1: Try standard word extraction (fastest)
        words = self._try_pymupdf_words(page)
        if words:
            logger.debug(f"PyMuPDF 'words' extracted {len(words)} words from page {page_number}")
            return words
        
        # Method 2: Try dict extraction
        words = self._try_pymupdf_dict(page)
        if words:
            logger.debug(f"PyMuPDF 'dict' extracted {len(words)} words from page {page_number}")
            return words
        
        # Method 3: Try rawdict extraction
        words = self._try_pymupdf_rawdict(page)
        logger.debug(f"PyMuPDF 'rawdict' extracted {len(words)} words from page {page_number}")
        return words
    
    def _try_pymupdf_words(self, page: fitz.Page) -> List[TextWord]:
        """Extract using PyMuPDF's get_text('words')."""
        try:
            raw_words = page.get_text("words")
            if raw_words:
                return [
                    TextWord(
                        x0=w[0], y0=w[1], x1=w[2], y1=w[3],
                        text=w[4],
                        block_no=w[5] if len(w) > 5 else 0,
                        line_no=w[6] if len(w) > 6 else 0,
                        word_no=w[7] if len(w) > 7 else 0
                    )
                    for w in raw_words if w[4].strip()
                ]
        except Exception as e:
            logger.debug(f"PyMuPDF 'words' extraction failed: {e}")
        return []
    
    def _try_pymupdf_dict(self, page: fitz.Page) -> List[TextWord]:
        """Extract using PyMuPDF's get_text('dict')."""
        words = []
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_no = block.get("number", 0)
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            text = span.get("text", "").strip()
                            if text:
                                words.append(TextWord(
                                    x0=bbox[0], y0=bbox[1],
                                    x1=bbox[2], y1=bbox[3],
                                    text=text,
                                    block_no=block_no
                                ))
        except Exception as e:
            logger.debug(f"PyMuPDF 'dict' extraction failed: {e}")
        return words
    
    def _try_pymupdf_rawdict(self, page: fitz.Page) -> List[TextWord]:
        """Extract using PyMuPDF's get_text('rawdict')."""
        words = []
        try:
            text_dict = page.get_text("rawdict")
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_no = block.get("number", 0)
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            bbox = span.get("bbox", [0, 0, 0, 0])
                            text = span.get("text", "").strip()
                            if text:
                                words.append(TextWord(
                                    x0=bbox[0], y0=bbox[1],
                                    x1=bbox[2], y1=bbox[3],
                                    text=text,
                                    block_no=block_no
                                ))
        except Exception as e:
            logger.debug(f"PyMuPDF 'rawdict' extraction failed: {e}")
        return words
    
    def get_fitz_page(self, page_number: int) -> fitz.Page:
        """
        Get the PyMuPDF page object for additional operations.
        
        Args:
            page_number: 1-indexed page number
        
        Returns:
            fitz.Page object
        """
        if not self._fitz_doc:
            raise RuntimeError("No PDF document is open")
        
        return self._fitz_doc.load_page(page_number - 1)
