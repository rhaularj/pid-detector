"""
Settings dialog for processing configuration.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable

from ...config import Settings


class SettingsDialog:
    """
    Modal settings dialog for processing configuration.
    
    Allows adjusting:
    - PDF rendering DPI
    - Detection confidence threshold
    - Quick presets
    """
    
    def __init__(
        self,
        parent: tk.Tk,
        dpi_var: tk.IntVar,
        conf_var: tk.DoubleVar,
        settings: Optional[Settings] = None,
        on_apply: Optional[Callable] = None
    ):
        """
        Initialize the settings dialog.
        
        Args:
            parent: Parent window
            dpi_var: DPI setting variable
            conf_var: Confidence threshold variable
            settings: Application settings
            on_apply: Optional callback when settings are applied
        """
        self.parent = parent
        self.dpi_var = dpi_var
        self.conf_var = conf_var
        self.settings = settings or Settings()
        self.on_apply = on_apply
        
        self.window: Optional[tk.Toplevel] = None
        
        # Store original values for cancel
        self._original_dpi = dpi_var.get()
        self._original_conf = conf_var.get()
    
    def show(self) -> None:
        """Show the settings dialog."""
        gui = self.settings.gui
        
        # Create modal window
        self.window = tk.Toplevel(self.parent)
        self.window.title("Processing Settings")
        self.window.geometry("400x320")
        self.window.resizable(False, False)
        self.window.configure(bg=gui.background_color)
        
        # Make modal
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # Center on parent
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() - 400) // 2
        y = self.parent.winfo_y() + (self.parent.winfo_height() - 320) // 2
        self.window.geometry(f"+{x}+{y}")
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """Create dialog widgets."""
        gui = self.settings.gui
        
        # Main frame
        main_frame = ttk.Frame(self.window, style="Modern.TFrame", padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(
            main_frame,
            text="Processing Settings",
            font=(gui.font_family, 16, 'bold'),
            bg=gui.background_color,
            fg=gui.text_primary
        )
        title.pack(pady=(0, 20))
        
        # DPI setting
        self._create_dpi_section(main_frame)
        
        # Confidence setting
        self._create_confidence_section(main_frame)
        
        # Presets
        self._create_presets_section(main_frame)
        
        # Buttons
        self._create_buttons(main_frame)
    
    def _create_dpi_section(self, parent: ttk.Frame) -> None:
        """Create DPI setting section."""
        gui = self.settings.gui
        
        frame = ttk.Frame(parent, style="Modern.TFrame")
        frame.pack(fill=tk.X, pady=10)
        
        label = tk.Label(
            frame,
            text="PDF Rendering DPI:",
            font=(gui.font_family, 10),
            bg=gui.background_color,
            fg=gui.text_secondary
        )
        label.pack(anchor=tk.W)
        
        info = tk.Label(
            frame,
            text="Higher DPI = better quality but slower processing",
            font=(gui.font_family, 8),
            bg=gui.background_color,
            fg=gui.text_muted
        )
        info.pack(anchor=tk.W, pady=(2, 5))
        
        spinbox = ttk.Spinbox(
            frame,
            from_=150,
            to=600,
            increment=50,
            textvariable=self.dpi_var,
            width=10,
            font=(gui.font_family, 10)
        )
        spinbox.pack(anchor=tk.W)
    
    def _create_confidence_section(self, parent: ttk.Frame) -> None:
        """Create confidence threshold section."""
        gui = self.settings.gui
        
        frame = ttk.Frame(parent, style="Modern.TFrame")
        frame.pack(fill=tk.X, pady=10)
        
        label = tk.Label(
            frame,
            text="Detection Confidence Threshold:",
            font=(gui.font_family, 10),
            bg=gui.background_color,
            fg=gui.text_secondary
        )
        label.pack(anchor=tk.W)
        
        info = tk.Label(
            frame,
            text="Lower = more detections, Higher = fewer false positives",
            font=(gui.font_family, 8),
            bg=gui.background_color,
            fg=gui.text_muted
        )
        info.pack(anchor=tk.W, pady=(2, 5))
        
        scale_frame = ttk.Frame(frame, style="Modern.TFrame")
        scale_frame.pack(fill=tk.X)
        
        scale = ttk.Scale(
            scale_frame,
            from_=0.1,
            to=0.9,
            variable=self.conf_var,
            orient=tk.HORIZONTAL
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.conf_label = tk.Label(
            scale_frame,
            text=f"{self.conf_var.get():.2f}",
            font=(gui.font_family, 10),
            bg=gui.background_color,
            fg=gui.text_primary,
            width=5
        )
        self.conf_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Update label on change
        self.conf_var.trace_add('write', self._update_conf_label)
    
    def _update_conf_label(self, *args) -> None:
        """Update confidence value label."""
        self.conf_label.config(text=f"{self.conf_var.get():.2f}")
    
    def _create_presets_section(self, parent: ttk.Frame) -> None:
        """Create quick presets section."""
        gui = self.settings.gui
        
        frame = ttk.Frame(parent, style="Modern.TFrame")
        frame.pack(fill=tk.X, pady=15)
        
        label = tk.Label(
            frame,
            text="Quick Presets:",
            font=(gui.font_family, 10),
            bg=gui.background_color,
            fg=gui.text_secondary
        )
        label.pack(anchor=tk.W, pady=(0, 5))
        
        btn_frame = ttk.Frame(frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X)
        
        presets = [
            ("Fast (200 DPI)", 200, 0.30),
            ("Balanced (300)", 300, 0.25),
            ("Quality (400)", 400, 0.20),
        ]
        
        for text, dpi, conf in presets:
            btn = ttk.Button(
                btn_frame,
                text=text,
                command=lambda d=dpi, c=conf: self._apply_preset(d, c),
                style="Secondary.TButton"
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))
    
    def _apply_preset(self, dpi: int, conf: float) -> None:
        """Apply a preset configuration."""
        self.dpi_var.set(dpi)
        self.conf_var.set(conf)
    
    def _create_buttons(self, parent: ttk.Frame) -> None:
        """Create dialog buttons."""
        frame = ttk.Frame(parent, style="Modern.TFrame")
        frame.pack(fill=tk.X, pady=(20, 0))
        
        apply_btn = ttk.Button(
            frame,
            text="Apply",
            command=self._on_apply,
            style="Modern.TButton"
        )
        apply_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_btn = ttk.Button(
            frame,
            text="Cancel",
            command=self._on_cancel,
            style="Secondary.TButton"
        )
        cancel_btn.pack(side=tk.RIGHT)
    
    def _on_apply(self) -> None:
        """Handle apply button click."""
        if self.on_apply:
            self.on_apply()
        self.window.destroy()
    
    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        # Restore original values
        self.dpi_var.set(self._original_dpi)
        self.conf_var.set(self._original_conf)
        self.window.destroy()
