"""
Progress view component for showing processing status.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

from ...config import Settings


class ProgressView(ttk.Frame):
    """
    Progress indicator view for long-running operations.
    
    Shows:
    - Processing status message
    - Progress bar
    - Percentage complete
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        progress_var: tk.DoubleVar,
        settings: Optional[Settings] = None,
        **kwargs
    ):
        """
        Initialize the progress view.
        
        Args:
            parent: Parent widget
            progress_var: Progress variable (0-100)
            settings: Application settings
        """
        super().__init__(parent, style="Card.TFrame", **kwargs)
        self.settings = settings or Settings()
        self.progress_var = progress_var
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """Create the progress UI widgets."""
        gui = self.settings.gui
        
        # Center content
        content = ttk.Frame(self, style="Modern.TFrame")
        content.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Processing icon
        icon_label = tk.Label(
            content,
            text="⚙️",
            font=(gui.font_family, 48),
            bg=gui.card_background,
            fg=gui.primary_color
        )
        icon_label.pack(pady=(0, 20))
        
        # Status text
        self.status_label = tk.Label(
            content,
            text="Processing PDF...",
            font=(gui.font_family, 16),
            bg=gui.card_background,
            fg=gui.text_secondary
        )
        self.status_label.pack()
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(
            content,
            variable=self.progress_var,
            maximum=100,
            length=400,
            style="Modern.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=20)
        
        # Percentage text
        self.percent_label = tk.Label(
            content,
            text="0%",
            font=(gui.font_family, 12),
            bg=gui.card_background,
            fg=gui.text_muted
        )
        self.percent_label.pack()
        
        # Bind to variable changes
        self.progress_var.trace_add('write', self._on_progress_change)
    
    def _on_progress_change(self, *args) -> None:
        """Update percentage label when progress changes."""
        value = int(self.progress_var.get())
        self.percent_label.config(text=f"{value}%")
    
    def set_status(self, message: str) -> None:
        """
        Update the status message.
        
        Args:
            message: New status message
        """
        self.status_label.config(text=message)
    
    def reset(self) -> None:
        """Reset the progress view to initial state."""
        self.progress_var.set(0)
        self.set_status("Processing PDF...")
