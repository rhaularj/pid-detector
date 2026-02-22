# P&ID Detector

An intelligent PDF processing application that uses computer vision and OCR to automatically detect, classify, and extract text from Piping & Instrumentation Diagrams (P&ID). Built with a custom-trained YOLO model for shape detection and multiple OCR engines for text extraction.

## Features

- **Custom YOLO Model**: Trained specifically for P&ID component detection
- **Multi-OCR Support**: Choose between PaddleOCR, EasyOCR, or native PDF text extraction
- **Interactive GUI**: Drag-and-drop interface built with Tkinter
- **CLI Support**: Process PDFs from the command line
- **Modular Architecture**: Clean separation of concerns for maintainability
- **Export Functionality**: Export results to Excel (XLSX) or JSON format
- **Visual Detection**: View detected components with bounding boxes and confidence scores
- **Robust Error Handling**: Multiple fallback strategies for reliable processing

## Project Structure

```
PidDetector/
├── src/                        # Source code package
│   ├── __init__.py
│   ├── app.py                  # Main entry point
│   ├── config/                 # Configuration management
│   │   ├── settings.py         # Settings and configuration
│   │   └── logging_config.py   # Logging setup
│   ├── core/                   # Core processing modules
│   │   ├── pdf_renderer.py     # PDF to image conversion
│   │   ├── text_extractor.py   # Text extraction from PDFs
│   │   ├── coordinate_transformer.py  # Coordinate mapping
│   │   ├── detector.py         # YOLO detection wrapper
│   │   └── processor.py        # Main processing pipeline
│   ├── gui/                    # GUI components
│   │   ├── app.py              # Main application window
│   │   ├── styles.py           # Tkinter styling
│   │   ├── views/              # View components
│   │   │   ├── upload_view.py
│   │   │   ├── progress_view.py
│   │   │   ├── data_view.py
│   │   │   └── settings_dialog.py
│   │   └── components/         # Reusable widgets
│   │       └── toolbar.py
│   ├── training/               # Model training utilities
│   │   └── train_yolo.py
│   └── utils/                  # Utilities
│       ├── visualization.py    # Detection plotting
│       └── file_utils.py       # File handling
├── tests/                      # Test suite
├── runs/                       # YOLO training outputs
├── docs/                       # Documentation
├── data.yaml                   # Dataset configuration
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
└── README.md
```

## Requirements

### System Requirements
- Python 3.8 or higher
- Windows/Linux/macOS
- Minimum 8GB RAM (16GB recommended for large PDFs)
- CUDA-compatible GPU (optional, for faster processing)

### Core Dependencies
- `ultralytics>=8.0.0` - YOLO object detection
- `opencv-python>=4.5.0` - Image processing
- `PyMuPDF>=1.20.0` - PDF rendering
- `pdfplumber>=0.7.0` - PDF text extraction
- `pandas>=1.3.0` - Data handling
- `matplotlib>=3.3.0` - Visualization

## Installation

### From Source

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/piddetector.git
cd piddetector
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install as package** (optional)
```bash
pip install -e .
```

### Using pip

```bash
pip install piddetector
```

## Usage

### GUI Application

```bash
# Run GUI
python -m src.app

# Or if installed as package
piddetector
```

1. **Load PDF**: Drag and drop a PDF file or use the browse button
2. **Processing**: The application will automatically detect P&ID components
3. **Review Results**: View detected shapes and extracted text in the data grid
4. **Export**: Save results to Excel or JSON format

### Command Line Interface

```bash
# Process a PDF file
python -m src.app --cli input.pdf -o results.xlsx

# Adjust confidence threshold
python -m src.app --cli input.pdf --confidence 0.5

# Process with custom model
python -m src.app --cli input.pdf --model path/to/model.pt
```

### Programmatic Usage

```python
from src.core.processor import PIDProcessor
from src.config import Settings

# Configure settings
settings = Settings()
settings.update_processing(confidence_threshold=0.3)

# Process a PDF
processor = PIDProcessor()
result = processor.process("your_pid_diagram.pdf")

# Access results
print(f"Found {result.total_detections} detections across {result.page_count} pages")
df = result.to_dataframe()
df.to_excel("results.xlsx", index=False)
```

## YOLO Model Training

### Using the Training Module

```bash
# Train a new model
python -m src.training.train_yolo --data data.yaml --epochs 100

# Or use the module
piddetector-train --data data.yaml --epochs 100 --model yolov8n.pt
```

### Dataset Preparation

1. **Collect P&ID Images**: Gather diverse P&ID diagrams
2. **Annotation**: Use tools like [LabelImg](https://github.com/heartexlabs/labelImg) or [Roboflow](https://roboflow.com)
3. **Classes**: Define your P&ID component classes

### Dataset Structure
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### Sample `data.yaml`
```yaml
path: ./dataset
train: images/train
val: images/val
test: images/test

nc: 8  # number of classes
names: ['valve', 'pump', 'instrument', 'pipe', 'tank', 'heat_exchanger', 'compressor', 'control_valve']
```

## Configuration

### Settings

Configure processing via `Settings` singleton:

```python
from src.config import Settings

settings = Settings()

# Update processing settings
settings.update_processing(
    default_dpi=300,
    confidence_threshold=0.25,
    text_margin_pts=6.0
)

# Update paths
settings.paths.model_path = "path/to/model.pt"
```

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
DEFAULT_DPI=300
CONFIDENCE_THRESHOLD=0.25
MODEL_PATH=
LOG_LEVEL=INFO
```

## Output Format

The application generates a structured DataFrame with the following columns:

| Column | Description |
|--------|-------------|
| shape | Detected P&ID component type |
| text | Extracted text from PDF |
| page | Page number |
| x, y | Coordinates in PDF points |
| width, height | Dimensions in PDF points |
| confidence | Detection confidence score |
| pdf_name | Source PDF filename |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_core.py -v
```

### Code Formatting

```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Project Commands

```bash
# Install development dependencies
pip install -e ".[dev]"

# Build package
python -m build

# Install with all optional dependencies
pip install -e ".[all]"
```

## Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure YOLO model is in `runs/detect/train*/weights/` or specify path
   - Check file permissions

2. **OCR failures**
   - PDF must contain extractable text (not scanned images)
   - For scanned PDFs, use separate OCR preprocessing

3. **Memory issues**
   - Reduce DPI setting for large PDFs
   - Process pages sequentially
   - Close other applications

### Performance Optimization

- **GPU Acceleration**: Ensure CUDA is properly installed
- **DPI Setting**: Lower DPI (150-200) for faster processing, higher (300-600) for accuracy
- **Model Selection**: Use YOLOv8n for speed, YOLOv8l for accuracy

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF rendering
- [pdfplumber](https://github.com/jsvine/pdfplumber) for text extraction

---

**Made for the Process Engineering Community**
