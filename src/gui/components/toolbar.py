"""
Toolbar component for action buttons.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable

from ...config import Settings


class Toolbar(ttk.Frame):
    """
    Reusable toolbar component with action buttons.
    
    Provides:
    - Export button
    - Settings button
    - Load new file button
    - Custom action slots
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_export: Optional[Callable] = None,
        on_reset: Optional[Callable] = None,
        on_settings: Optional[Callable] = None,
        settings: Optional[Settings] = None,
        **kwargs
    ):
        """
        Initialize the toolbar.
        
        Args:
            parent: Parent widget
            on_export: Callback for export action
            on_reset: Callback for reset/new file action
            on_settings: Callback for settings action
            settings: Application settings
        """
        super().__init__(parent, style="Modern.TFrame", **kwargs)
        self.settings = settings or Settings()
        self.on_export = on_export
        self.on_reset = on_reset
        self.on_settings = on_settings
        
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """Create toolbar widgets."""
        # Left side - primary actions
        left_frame = ttk.Frame(self, style="Modern.TFrame")
        left_frame.pack(side=tk.LEFT)
        
        if self.on_export:
            export_btn = ttk.Button(
                left_frame,
                text="📊 Export to Excel",
                command=self.on_export,
                style="Modern.TButton"
            )
            export_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Right side - secondary actions
        right_frame = ttk.Frame(self, style="Modern.TFrame")
        right_frame.pack(side=tk.RIGHT)
        
        if self.on_settings:
            settings_btn = ttk.Button(
                right_frame,
                text="⚙️ Settings",
                command=self.on_settings,
                style="Secondary.TButton"
            )
            settings_btn.pack(side=tk.RIGHT, padx=(0, 10))
        
        if self.on_reset:
            reset_btn = ttk.Button(
                right_frame,
                text="🔄 Load New PDF",
                command=self.on_reset,
                style="Secondary.TButton"
            )
            reset_btn.pack(side=tk.RIGHT)
