"""
Upload view component for file selection and drag-drop.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Callable, Optional

from ...config import Settings

# Try to import drag-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False


class UploadView(ttk.Frame):
    """
    File upload view with drag-drop support.
    
    Provides:
    - Drag and drop file upload
    - File browser button
    - Visual feedback for drag operations
    - Settings display
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_file_selected: Callable[[str], None],
        settings: Optional[Settings] = None,
        **kwargs
    ):
        """
        Initialize the upload view.
        
        Args:
            parent: Parent widget
            on_file_selected: Callback when file is selected
            settings: Application settings
        """
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.settings = settings or Settings()
        self.on_file_selected = on_file_selected
        
        self._dpi_var: Optional[tk.IntVar] = None
        self._conf_var: Optional[tk.DoubleVar] = None
        
        self._create_widgets()
        self._setup_drag_drop()
    
    def set_settings_vars(
        self, 
        dpi_var: tk.IntVar, 
        conf_var: tk.DoubleVar
    ) -> None:
        """
        Set settings variables for display.
        
        Args:
            dpi_var: DPI setting variable
            conf_var: Confidence threshold variable
        """
        self._dpi_var = dpi_var
        self._conf_var = conf_var
        self._update_settings_label()
    
    def _create_widgets(self) -> None:
        """Create the upload UI widgets."""
        gui = self.settings.gui
        
        # Center content frame
        content = ttk.Frame(self, style="Modern.TFrame")
        content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Upload icon
        icon_label = tk.Label(
            content,
            text="📄",
            font=(gui.font_family, 48),
            bg=gui.card_background,
            fg=gui.primary_color
        )
        icon_label.pack(pady=(0, 20))
        
        # Main text
        upload_label = tk.Label(
            content,
            text="Drag and drop a PDF file here",
            font=(gui.font_family, 16),
            bg=gui.card_background,
            fg=gui.text_secondary
        )
        upload_label.pack()
        
        # "or" divider
        or_label = tk.Label(
            content,
            text="or",
            font=(gui.font_family, 12),
            bg=gui.card_background,
            fg=gui.text_muted
        )
        or_label.pack(pady=(10, 15))
        
        # Browse button
        browse_btn = ttk.Button(
            content,
            text="Browse Files",
            command=self._browse_file,
            style="Modern.TButton"
        )
        browse_btn.pack()
        
        # Supported formats
        formats_label = tk.Label(
            content,
            text="Supported format: PDF",
            font=(gui.font_family, 9),
            bg=gui.card_background,
            fg=gui.text_muted
        )
        formats_label.pack(pady=(15, 0))
        
        # Settings info
        self._settings_label = tk.Label(
            content,
            text="",
            font=(gui.font_family, 8),
            bg=gui.card_background,
            fg=gui.primary_color
        )
        self._settings_label.pack(pady=(5, 0))
        
        # Drag-drop status
        if not DRAG_DROP_AVAILABLE:
            dnd_label = tk.Label(
                content,
                text="(Drag-drop disabled - install tkinterdnd2)",
                font=(gui.font_family, 8),
                bg=gui.card_background,
                fg="#dc2626"
            )
            dnd_label.pack(pady=(5, 0))
    
    def _setup_drag_drop(self) -> None:
        """Setup drag and drop handling."""
        if not DRAG_DROP_AVAILABLE:
            return
        
        try:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self._on_drop)
        except Exception as e:
            print(f"Could not setup drag-and-drop: {e}")
    
    def _on_drop(self, event) -> None:
        """Handle dropped files."""
        try:
            files = self.winfo_toplevel().tk.splitlist(event.data)
            if files:
                file_path = files[0].strip('{}').strip('"').strip("'")
                if file_path:
                    self.on_file_selected(file_path)
        except Exception as e:
            print(f"Drop error: {e}")
    
    def _browse_file(self) -> None:
        """Open file browser dialog."""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=str(Path.home())
        )
        if file_path:
            self.on_file_selected(file_path)
    
    def _update_settings_label(self) -> None:
        """Update the settings display label."""
        if self._dpi_var and self._conf_var:
            text = f"DPI: {self._dpi_var.get()} | Confidence: {self._conf_var.get():.2f}"
            self._settings_label.config(text=text)
