"""
Logging configuration for the PidDetector application.

Provides consistent logging setup across all modules with support for
both console and file logging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Module-level logger cache
_loggers: dict = {}

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: Optional[Path] = None,
    log_format: str = DEFAULT_FORMAT
) -> logging.Logger:
    """
    Configure the root logger for the application.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_to_file: Whether to also log to a file
        log_dir: Directory for log files (default: project_root/logs)
        log_format: Format string for log messages
    
    Returns:
        Configured root logger
    """
    # Get root logger for the application
    root_logger = logging.getLogger("piddetector")
    
    # Avoid duplicate handlers
    if root_logger.handlers:
        return root_logger
    
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(SIMPLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"piddetector_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__ from the calling module)
    
    Returns:
        Logger instance
    
    Usage:
        from src.config import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create child logger under piddetector namespace
    if not name.startswith("piddetector"):
        # Convert module path to logger name
        # e.g., "src.core.processor" -> "piddetector.core.processor"
        parts = name.split(".")
        if "src" in parts:
            parts = parts[parts.index("src") + 1:]
        name = f"piddetector.{'.'.join(parts)}"
    
    logger = logging.getLogger(name)
    _loggers[name] = logger
    
    return logger


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    
    Usage:
        class MyProcessor(LoggerMixin):
            def process(self):
                self.logger.info("Processing...")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger
