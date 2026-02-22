"""
P&ID Detector - Main Entry Point

This module provides the main entry points for the P&ID Detector application.

Usage:
    # Run GUI application
    python -m src.app
    
    # Or directly
    python src/app.py
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gui import PIDDetectorApp
from src.core import PIDProcessor, process_pdf_with_yolo
from src.config import setup_logging, Settings


def run_gui() -> None:
    """Run the GUI application."""
    PIDDetectorApp.create_and_run()


def run_cli(args: argparse.Namespace) -> None:
    """
    Run CLI processing.
    
    Args:
        args: Parsed command line arguments
    """
    setup_logging()
    
    processor = PIDProcessor(model_path=args.model)
    
    result = processor.process(
        pdf_path=args.pdf,
        dpi=args.dpi,
        confidence=args.conf,
        output_json=args.output
    )
    
    print(f"\nProcessed {result.total_pages} pages")
    print(f"Total detections: {result.total_detections}")
    
    if args.output:
        print(f"Results saved to: {args.output}")


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="P&ID Detector - Detect symbols in P&ID PDF documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GUI
  python -m src.app
  
  # Process PDF from command line
  python -m src.app --cli input.pdf --model models/best.pt
  
  # With custom settings
  python -m src.app --cli input.pdf --model models/best.pt --dpi 300 --conf 0.25
        """
    )
    
    parser.add_argument(
        "--cli", 
        type=str, 
        metavar="PDF",
        help="Run in CLI mode with specified PDF file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Path to YOLO model (default: auto-detect)"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=None,
        help="PDF rendering DPI (default: auto-calculate)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str, 
        default="pid_detections.json",
        help="Output JSON file path (default: pid_detections.json)"
    )
    
    args = parser.parse_args()
    
    if args.cli:
        # CLI mode
        run_cli(args)
    else:
        # GUI mode
        run_gui()


if __name__ == "__main__":
    main()
