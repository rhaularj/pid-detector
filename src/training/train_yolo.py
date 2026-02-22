"""
YOLO model training utilities for P&ID detection.

Provides a clean interface for training and fine-tuning YOLO models
on custom P&ID symbol datasets.
"""

import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from ultralytics import YOLO

from ..config import get_logger, Settings

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for YOLO model training."""
    
    # Model configuration
    base_model: str = "yolo11s.pt"  # Base model to start from
    
    # Dataset configuration
    data_yaml: str = "data.yaml"
    
    # Training parameters
    epochs: int = 200
    imgsz: int = 1280  # Match inference size for best results
    batch: int = 8
    workers: int = 4
    patience: int = 100  # Early stopping patience
    
    # Device configuration
    device: str = "0"  # GPU device ID or "cpu"
    
    # Augmentation
    augment: bool = True
    close_mosaic: int = 10  # Disable mosaic for last N epochs
    
    # Validation
    val: bool = True
    
    # Output
    project: str = "runs/detect"
    name: str = "train"
    exist_ok: bool = False  # Overwrite existing experiment
    
    # Display
    show: bool = False  # Show training images
    verbose: bool = True


class YOLOTrainer:
    """
    YOLO model trainer for P&ID symbol detection.
    
    Features:
    - Configurable training parameters
    - Progress tracking
    - Model validation
    - Export to different formats
    
    Usage:
        trainer = YOLOTrainer()
        results = trainer.train()
        trainer.export("best.pt", format="onnx")
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        settings: Optional[Settings] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            settings: Application settings
        """
        self.config = config or TrainingConfig()
        self.settings = settings or Settings()
        self._model: Optional[YOLO] = None
        self._results = None
    
    @property
    def model(self) -> Optional[YOLO]:
        """Get the current model."""
        return self._model
    
    @property
    def results(self):
        """Get training results."""
        return self._results
    
    def train(self) -> Any:
        """
        Train the YOLO model.
        
        Returns:
            Training results object
        """
        logger.info("=" * 60)
        logger.info("STARTING YOLO TRAINING")
        logger.info("=" * 60)
        logger.info(f"Base model: {self.config.base_model}")
        logger.info(f"Dataset: {self.config.data_yaml}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Image size: {self.config.imgsz}")
        logger.info(f"Batch size: {self.config.batch}")
        logger.info(f"Device: {self.config.device}")
        
        # Initialize model
        self._model = YOLO(self.config.base_model)
        
        # Start training
        self._results = self._model.train(
            data=self.config.data_yaml,
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch,
            workers=self.config.workers,
            device=self.config.device,
            show=self.config.show,
            patience=self.config.patience,
            val=self.config.val,
            augment=self.config.augment,
            pretrained=True,
            close_mosaic=self.config.close_mosaic,
            project=self.config.project,
            name=self.config.name,
            exist_ok=self.config.exist_ok,
            verbose=self.config.verbose
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        
        return self._results
    
    def validate(
        self,
        model_path: Optional[str] = None,
        data: Optional[str] = None
    ) -> Any:
        """
        Validate a trained model.
        
        Args:
            model_path: Path to model weights (uses last trained if not provided)
            data: Path to data.yaml (uses config if not provided)
        
        Returns:
            Validation results
        """
        if model_path:
            model = YOLO(model_path)
        elif self._model:
            model = self._model
        else:
            raise ValueError("No model available for validation")
        
        data = data or self.config.data_yaml
        
        logger.info(f"Validating model on {data}")
        results = model.val(data=data)
        
        return results
    
    def export(
        self,
        model_path: str,
        format: str = "onnx",
        **kwargs
    ) -> str:
        """
        Export model to different formats.
        
        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, tensorrt, etc.)
            **kwargs: Additional export arguments
        
        Returns:
            Path to exported model
        """
        model = YOLO(model_path)
        export_path = model.export(format=format, **kwargs)
        logger.info(f"Exported model to: {export_path}")
        return export_path
    
    @staticmethod
    def get_best_model(runs_dir: str = "runs/detect") -> Optional[str]:
        """
        Find the best model from training runs.
        
        Args:
            runs_dir: Training runs directory
        
        Returns:
            Path to best.pt or None if not found
        """
        runs_path = Path(runs_dir)
        if not runs_path.exists():
            return None
        
        # Find most recent training run
        train_dirs = sorted(
            [d for d in runs_path.iterdir() if d.is_dir() and d.name.startswith("train")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for train_dir in train_dirs:
            best_pt = train_dir / "weights" / "best.pt"
            if best_pt.exists():
                return str(best_pt)
        
        return None


def train_yolo(config: Optional[TrainingConfig] = None) -> None:
    """
    Main training function with multiprocessing support.
    
    Args:
        config: Training configuration
    """
    trainer = YOLOTrainer(config)
    trainer.train()


def main():
    """CLI entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train YOLO model for P&ID symbol detection"
    )
    parser.add_argument("--model", type=str, default="yolo11s.pt",
                       help="Base model to train from")
    parser.add_argument("--data", type=str, default="data.yaml",
                       help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280,
                       help="Training image size")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="0",
                       help="Device to train on (GPU ID or 'cpu')")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--project", type=str, default="runs/detect",
                       help="Project directory")
    parser.add_argument("--name", type=str, default="train",
                       help="Experiment name")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        base_model=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name
    )
    
    # Enable multiprocessing support for Windows
    multiprocessing.freeze_support()
    
    train_yolo(config)


if __name__ == "__main__":
    main()
