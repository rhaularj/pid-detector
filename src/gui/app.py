"""
Main GUI application for P&ID Detector.

Integrates all GUI components and processing logic into a cohesive application.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from pathlib import Path
from typing import Optional
import pandas as pd

from .styles import StyleManager
from .views import UploadView, ProgressView, DataView, SettingsDialog
from ..config import Settings, get_logger, setup_logging
from ..core import PIDProcessor
from ..utils import FileHandler

# Try to import drag-drop support
try:
    from tkinterdnd2 import TkinterDnD
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False

logger = get_logger(__name__)


class PIDDetectorApp:
    """
    Main application class for the P&ID Detector GUI.
    
    Orchestrates:
    - GUI layout and navigation
    - User interactions
    - Background processing
    - Data export
    
    Usage:
        app = PIDDetectorApp()
        app.run()
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the application.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        
        # Initialize logging
        setup_logging()
        
        # Create root window
        self._create_root()
        
        # Application state
        self._current_df: Optional[pd.DataFrame] = None
        self._current_pdf_path: Optional[str] = None
        
        # Tkinter variables
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Ready")
        self.dpi_var = tk.IntVar(value=self.settings.processing.default_dpi)
        self.conf_var = tk.DoubleVar(value=self.settings.processing.confidence_threshold)
        
        # Processing components
        self._processor: Optional[PIDProcessor] = None
        self._file_handler = FileHandler(self.settings)
        
        # Apply styles
        self.style_manager = StyleManager(self.root, self.settings)
        self.style_manager.apply()
        
        # Build UI
        self._create_layout()
        self._show_upload_view()
    
    def _create_root(self) -> None:
        """Create the root window."""
        gui = self.settings.gui
        
        # Use drag-drop enabled root if available
        if DRAG_DROP_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
            logger.warning("Drag-drop disabled. Install tkinterdnd2 for full functionality.")
        
        self.root.title(gui.window_title)
        self.root.geometry(gui.window_geometry)
        self.root.minsize(gui.min_width, gui.min_height)
        self.root.configure(bg=gui.background_color)
    
    def _create_layout(self) -> None:
        """Create the main application layout."""
        gui = self.settings.gui
        
        # Main container
        self._main_frame = ttk.Frame(self.root, style="Modern.TFrame")
        self._main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        self._create_header()
        
        # Content area (views will be swapped here)
        self._content_frame = ttk.Frame(self._main_frame, style="Modern.TFrame")
        self._content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Footer with status bar
        self._create_footer()
    
    def _create_header(self) -> None:
        """Create the application header."""
        gui = self.settings.gui
        
        header = ttk.Frame(self._main_frame, style="Modern.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        
        title = tk.Label(
            header,
            text=gui.window_title,
            font=(gui.font_family, gui.title_font_size, 'bold'),
            bg=gui.background_color,
            fg=gui.text_primary
        )
        title.pack(anchor=tk.W)
        
        subtitle = tk.Label(
            header,
            text="Detect and extract P&ID symbols from PDF documents",
            font=(gui.font_family, gui.subtitle_font_size),
            bg=gui.background_color,
            fg=gui.text_muted
        )
        subtitle.pack(anchor=tk.W, pady=(5, 0))
        
        separator = ttk.Separator(header, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=(15, 0))
    
    def _create_footer(self) -> None:
        """Create the footer with status bar."""
        gui = self.settings.gui
        
        footer = ttk.Frame(self._main_frame, style="Modern.TFrame")
        footer.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        status_frame = ttk.Frame(footer, style="Card.TFrame")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self._status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=(gui.font_family, gui.small_font_size),
            bg=gui.card_background,
            fg=gui.text_secondary,
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self._status_label.pack(fill=tk.X)
    
    def _clear_content(self) -> None:
        """Clear the content frame."""
        for widget in self._content_frame.winfo_children():
            widget.destroy()
    
    def _show_upload_view(self) -> None:
        """Show the upload/drag-drop view."""
        self._clear_content()
        
        self._upload_view = UploadView(
            self._content_frame,
            on_file_selected=self._handle_file,
            settings=self.settings
        )
        self._upload_view.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        self._upload_view.set_settings_vars(self.dpi_var, self.conf_var)
        
        self.status_var.set("Ready")
    
    def _show_progress_view(self) -> None:
        """Show the progress view."""
        self._clear_content()
        
        self._progress_view = ProgressView(
            self._content_frame,
            progress_var=self.progress_var,
            settings=self.settings
        )
        self._progress_view.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
    
    def _show_data_view(self, data: pd.DataFrame) -> None:
        """
        Show the data grid view.
        
        Args:
            data: DataFrame to display
        """
        self._clear_content()
        self._current_df = data
        
        self._data_view = DataView(
            self._content_frame,
            on_export=self._export_to_excel,
            on_reset=self._reset_ui,
            on_settings=self._show_settings,
            settings=self.settings
        )
        self._data_view.pack(fill=tk.BOTH, expand=True)
        self._data_view.load_data(data)
        
        self.status_var.set(f"Loaded {len(data)} rows × {len(data.columns)} columns")
    
    def _handle_file(self, file_path: str) -> None:
        """
        Handle a selected file.
        
        Args:
            file_path: Path to the selected file
        """
        # Clean and validate path
        file_path = self._file_handler.clean_path(file_path)
        
        if not self._file_handler.validate_pdf(file_path):
            messagebox.showerror("Invalid File", "Please select a valid PDF file.")
            return
        
        self._current_pdf_path = file_path
        self.status_var.set(f"Loading: {Path(file_path).name}")
        
        # Show progress and start processing
        self._show_progress_view()
        
        # Process in background thread
        thread = threading.Thread(
            target=self._process_pdf,
            args=(file_path,),
            daemon=True
        )
        thread.start()
    
    def _process_pdf(self, file_path: str) -> None:
        """
        Process a PDF file in background.
        
        Args:
            file_path: Path to PDF file
        """
        try:
            # Initialize processor if needed
            if self._processor is None:
                self._processor = PIDProcessor(settings=self.settings)
            
            # Update status
            dpi = self.dpi_var.get()
            conf = self.conf_var.get()
            
            self.root.after(0, lambda: self.status_var.set(
                f"Processing with DPI={dpi}, Confidence={conf:.2f}..."
            ))
            
            # Process PDF
            result = self._processor.process(
                pdf_path=file_path,
                dpi=dpi,
                confidence=conf,
                progress_callback=self._update_progress,
                show_progress=False
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(result.detections)
            
            # Update UI on main thread
            self.root.after(0, lambda: self._show_data_view(df))
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self.root.after(0, lambda: messagebox.showerror(
                "Error",
                f"Failed to process PDF:\n{str(e)}"
            ))
            self.root.after(0, self._show_upload_view)
    
    def _update_progress(self, percent: int) -> None:
        """
        Update progress bar (thread-safe).
        
        Args:
            percent: Progress percentage (0-100)
        """
        self.root.after(0, lambda: self.progress_var.set(percent))
    
    def _export_to_excel(self) -> None:
        """Export data to Excel file."""
        if self._current_df is None or self._current_df.empty:
            messagebox.showerror("No Data", "No data available to export.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")],
            initialdir=str(Path.home())
        )
        
        if not file_path:
            return
        
        try:
            self._current_df.to_excel(file_path, index=False)
            messagebox.showinfo("Success", f"Data exported successfully!\n{file_path}")
            self.status_var.set(f"Exported to: {Path(file_path).name}")
        except Exception as e:
            logger.error(f"Export failed: {e}")
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
    
    def _reset_ui(self) -> None:
        """Reset UI to initial state."""
        self._current_df = None
        self._current_pdf_path = None
        self.progress_var.set(0)
        self._show_upload_view()
    
    def _show_settings(self) -> None:
        """Show settings dialog."""
        dialog = SettingsDialog(
            self.root,
            dpi_var=self.dpi_var,
            conf_var=self.conf_var,
            settings=self.settings
        )
        dialog.show()
    
    def run(self) -> None:
        """Start the application main loop."""
        logger.info("Starting P&ID Detector application")
        self.root.mainloop()
    
    @classmethod
    def create_and_run(cls) -> None:
        """Create and run the application."""
        app = cls()
        app.run()
