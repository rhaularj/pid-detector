# P&ID Detector - System Architecture

## Overview

The P&ID Detector is a modular application for detecting and classifying components in P&ID diagrams from PDFs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTRY POINT (src/app.py)                      │
│                    CLI / GUI Mode Selection                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                CONFIGURATION (src/config/)                       │
│  settings.py (ProcessingConfig, PathConfig, GUIConfig)          │
│  logging_config.py (setup_logging, get_logger)                  │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                    GUI LAYER (src/gui/)                          │
│  app.py (PIDDetectorApp)                                         │
│  views/ (UploadView, ProgressView, DataView, SettingsDialog)    │
│  components/ (Toolbar)                                           │
│  styles.py (StyleManager)                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                   CORE LAYER (src/core/)                         │
│  processor.py (PIDProcessor - orchestrates pipeline)            │
│  ├── pdf_renderer.py (PDFRenderer, RenderedPage)                │
│  ├── detector.py (YOLODetector, Detection, PageDetections)      │
│  ├── text_extractor.py (TextExtractor, TextWord)                │
│  └── coordinate_transformer.py (CoordinateTransformer)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┼────────────────────────────────────┐
│                 UTILITIES (src/utils/)                           │
│  visualization.py (DetectionVisualizer)                          │
│  file_utils.py (FileHandler)                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── __init__.py              # Package root
├── app.py                   # Main entry point
├── config/                  # Configuration
│   ├── settings.py          # Settings singleton
│   └── logging_config.py    # Logging setup
├── core/                    # Processing pipeline
│   ├── pdf_renderer.py      # PDF to image
│   ├── detector.py          # YOLO detection
│   ├── text_extractor.py    # Text extraction
│   ├── coordinate_transformer.py  # Coord mapping
│   └── processor.py         # Pipeline orchestration
├── gui/                     # GUI components
│   ├── app.py               # Main window
│   ├── styles.py            # ttk styling
│   ├── views/               # View components
│   └── components/          # Reusable widgets
├── training/                # Training utilities
│   └── train_yolo.py        # YOLO training
└── utils/                   # Support utilities
    ├── visualization.py     # Detection plotting
    └── file_utils.py        # File handling
```

## Data Flow

```
PDF File → PDFRenderer (render pages)
              ↓
         RenderedPage (image + dimensions)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
YOLODetector      TextExtractor
    ↓                   ↓
Detection[]        TextWord[]
    ↓                   ↓
    └─────────┬─────────┘
              ↓
    CoordinateTransformer (pixel ↔ PDF)
              ↓
    TextAssociator (match text to detections)
              ↓
    ProcessingResult → DataFrame → Excel/JSON
```

## Key Classes

### Configuration
- **Settings**: Singleton holding all configuration
- **ProcessingConfig**: DPI, confidence, text margin settings
- **PathConfig**: Model and output paths
- **GUIConfig**: Colors, fonts, window settings

### Core Processing
- **PDFRenderer**: Renders PDF pages to images using PyMuPDF
- **YOLODetector**: Runs YOLO inference on images
- **TextExtractor**: Extracts text from PDFs (pdfplumber + PyMuPDF fallback)
- **CoordinateTransformer**: Converts between pixel and PDF coordinates
- **PIDProcessor**: Orchestrates the complete pipeline

### Data Classes
- **RenderedPage**: Image array with dimension metadata
- **Detection**: Single detected object with bbox and confidence
- **PageDetections**: All detections for one page
- **TextWord**: Word with bounding box coordinates
- **ProcessingResult**: Complete results with DataFrame conversion

## Technology Stack

| Component | Technology |
|-----------|------------|
| Object Detection | Ultralytics YOLOv8 |
| PDF Rendering | PyMuPDF (fitz) |
| Text Extraction | pdfplumber, PyMuPDF |
| Image Processing | OpenCV, NumPy |
| Data Handling | Pandas |
| GUI Framework | Tkinter + ttk |
| Drag-Drop | tkinterdnd2 |
