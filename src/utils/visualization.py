"""
Visualization utilities for detection results.

Provides matplotlib-based visualization for debugging and result display.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from ..config import get_logger, Settings

logger = get_logger(__name__)


class DetectionVisualizer:
    """
    Visualizes object detections using matplotlib.
    
    Features:
    - Bounding box overlay on images
    - Class labels with confidence scores  
    - Debug visualizations for coordinate mapping
    - Save to file or display interactively
    
    Usage:
        visualizer = DetectionVisualizer()
        fig = visualizer.plot_detections(image, detections, class_names)
        visualizer.save("output.png")
    """
    
    # Default color palette for different classes
    DEFAULT_COLORS = [
        '#2563eb',  # blue
        '#16a34a',  # green
        '#dc2626',  # red
        '#0891b2',  # cyan
        '#c026d3',  # magenta
        '#ca8a04',  # yellow
        '#7c3aed',  # purple
        '#ea580c',  # orange
        '#0d9488',  # teal
        '#be123c',  # rose
    ]
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the visualizer.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self._current_fig: Optional[plt.Figure] = None
    
    def plot_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        figsize: Tuple[int, int] = (15, 10),
        font_size: int = 10,
        linewidth: int = 2,
        show_confidence: bool = True,
        show_text: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot detections on an image.
        
        Args:
            image: Input image (BGR or RGB)
            detections: List of detection dictionaries with bbox_pixels, class_name, etc.
            figsize: Figure size (width, height)
            font_size: Font size for labels
            linewidth: Line width for bounding boxes
            show_confidence: Whether to show confidence scores
            show_text: Whether to show associated text
            title: Optional figure title
        
        Returns:
            Matplotlib figure
        """
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        ax.imshow(image_rgb)
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=font_size + 2, fontweight='bold')
        
        # Draw each detection
        for i, det in enumerate(detections):
            self._draw_detection(
                ax, det, i, 
                font_size=font_size,
                linewidth=linewidth,
                show_confidence=show_confidence,
                show_text=show_text
            )
        
        plt.tight_layout()
        self._current_fig = fig
        return fig
    
    def _draw_detection(
        self,
        ax: plt.Axes,
        detection: Dict[str, Any],
        index: int,
        font_size: int,
        linewidth: int,
        show_confidence: bool,
        show_text: bool
    ) -> None:
        """Draw a single detection on the axes."""
        # Get bbox coordinates (prefer pixels for display)
        bbox = detection.get('bbox_pixels', detection.get('bbox_pdf', [0, 0, 0, 0]))
        x0, y0, x1, y1 = bbox
        
        # Get class info
        class_name = detection.get('class_name', 'Unknown')
        confidence = detection.get('confidence', 0.0)
        
        # Select color
        class_id = detection.get('class_id', index)
        color = self.DEFAULT_COLORS[class_id % len(self.DEFAULT_COLORS)]
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Build label
        label_parts = [class_name]
        if show_confidence:
            label_parts.append(f'{confidence:.2f}')
        label = ' '.join(label_parts)
        
        # Draw label
        ax.text(
            x0, y0 - 5, label,
            fontsize=font_size,
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor=color,
                edgecolor='none',
                alpha=0.8
            )
        )
        
        # Draw associated text if available
        if show_text:
            texts = detection.get('texts_inside', [])
            if texts:
                text_label = ', '.join(texts[:3])
                if len(texts) > 3:
                    text_label += f' (+{len(texts) - 3})'
                
                ax.text(
                    x0, y1 + 5, text_label,
                    fontsize=font_size - 2,
                    color='white',
                    verticalalignment='top',
                    bbox=dict(
                        boxstyle='round,pad=0.2',
                        facecolor='#333333',
                        edgecolor='none',
                        alpha=0.7
                    )
                )
    
    def plot_coordinate_debug(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        page_size_pts: Tuple[float, float],
        image_size_px: Tuple[int, int],
        dpi: int,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create debug visualization showing coordinate mapping.
        
        Args:
            image: Rendered page image
            detections: Detection dictionaries with both bbox_pixels and bbox_pdf
            page_size_pts: (width, height) of PDF page in points
            image_size_px: (width, height) of image in pixels
            dpi: DPI used for rendering
            save_path: Optional path to save the figure
        
        Returns:
            Matplotlib figure with side-by-side comparison
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Left: Image pixel coordinates
        ax1.imshow(image_rgb)
        ax1.set_title(
            f'Image Space (Pixels) - {image_size_px[0]}×{image_size_px[1]}',
            fontsize=12, fontweight='bold'
        )
        ax1.axis('off')
        
        # Right: PDF coordinates mapped back to image
        ax2.imshow(image_rgb)
        ax2.set_title(
            f'PDF Space (Points) Mapped to Image - DPI: {dpi}',
            fontsize=12, fontweight='bold'
        )
        ax2.axis('off')
        
        page_w, page_h = page_size_pts
        img_w, img_h = image_size_px
        scale_x = img_w / page_w
        scale_y = img_h / page_h
        
        for i, det in enumerate(detections):
            color = self.DEFAULT_COLORS[i % len(self.DEFAULT_COLORS)]
            
            # Left: Draw original pixel bbox
            bbox_px = det.get('bbox_pixels', [0, 0, 0, 0])
            x0, y0, x1, y1 = bbox_px
            rect_img = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor=color,
                facecolor='none', linestyle='-'
            )
            ax1.add_patch(rect_img)
            ax1.text(x0, y0 - 5, f"#{i+1}", fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
            
            # Right: Convert PDF bbox back to pixels for display
            bbox_pdf = det.get('bbox_pdf')
            if bbox_pdf:
                x0_p = bbox_pdf[0] * scale_x
                y0_p = bbox_pdf[1] * scale_y
                x1_p = bbox_pdf[2] * scale_x
                y1_p = bbox_pdf[3] * scale_y
                
                rect_pdf = patches.Rectangle(
                    (x0_p, y0_p), x1_p - x0_p, y1_p - y0_p,
                    linewidth=2, edgecolor=color,
                    facecolor='none', linestyle='--'
                )
                ax2.add_patch(rect_pdf)
                
                # Show class and text
                class_name = det.get('class_name', 'Unknown')
                texts = det.get('texts_inside', [])
                text_str = ', '.join(texts[:2]) if texts else 'No text'
                label = f"#{i+1} {class_name}\n{text_str}"
                
                ax2.text(x0_p, y0_p - 5, label, fontsize=8, color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
        
        # Add legend
        legend_elements = [
            patches.Patch(facecolor='none', edgecolor='gray', linewidth=2,
                         label='Image pixels (solid)', linestyle='-'),
            patches.Patch(facecolor='none', edgecolor='gray', linewidth=2,
                         label='PDF points (dashed)', linestyle='--')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Debug visualization saved to: {save_path}")
        
        self._current_fig = fig
        return fig
    
    def save(
        self,
        path: str,
        dpi: int = 150,
        bbox_inches: str = 'tight'
    ) -> None:
        """
        Save the current figure to file.
        
        Args:
            path: Output file path
            dpi: Output DPI
            bbox_inches: Bbox setting for savefig
        """
        if self._current_fig is None:
            raise RuntimeError("No figure to save. Call plot_detections first.")
        
        self._current_fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Figure saved to: {path}")
    
    def show(self) -> None:
        """Display the current figure."""
        if self._current_fig is None:
            raise RuntimeError("No figure to show. Call plot_detections first.")
        plt.show()
    
    def close(self) -> None:
        """Close the current figure."""
        if self._current_fig is not None:
            plt.close(self._current_fig)
            self._current_fig = None
