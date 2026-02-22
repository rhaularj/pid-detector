"""
Tkinter style configuration for the GUI.

Provides consistent modern styling across all GUI components.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

from ..config import Settings


class StyleManager:
    """
    Manages Tkinter/ttk styling for the application.
    
    Provides a modern, consistent look across all widgets.
    
    Usage:
        style_manager = StyleManager(root)
        style_manager.apply()
    """
    
    def __init__(self, root: tk.Tk, settings: Optional[Settings] = None):
        """
        Initialize the style manager.
        
        Args:
            root: Tkinter root window
            settings: Application settings
        """
        self.root = root
        self.settings = settings or Settings()
        self.gui_config = self.settings.gui
        self.style = ttk.Style()
    
    def apply(self) -> None:
        """Apply all styles to the application."""
        self._set_theme()
        self._configure_frame_styles()
        self._configure_button_styles()
        self._configure_treeview_styles()
        self._configure_progressbar_styles()
        self._configure_entry_styles()
        self._configure_scale_styles()
    
    def _set_theme(self) -> None:
        """Set the base theme."""
        available = self.style.theme_names()
        
        # Prefer modern themes
        for theme in ['vista', 'clam', 'alt', 'default']:
            if theme in available:
                self.style.theme_use(theme)
                break
    
    def _configure_frame_styles(self) -> None:
        """Configure frame styles."""
        self.style.configure(
            "Modern.TFrame",
            background=self.gui_config.background_color,
            relief="flat",
            borderwidth=0
        )
        
        self.style.configure(
            "Card.TFrame",
            background=self.gui_config.card_background,
            relief="solid",
            borderwidth=1
        )
    
    def _configure_button_styles(self) -> None:
        """Configure button styles."""
        # Primary button
        self.style.configure(
            "Modern.TButton",
            background=self.gui_config.card_background,
            foreground=self.gui_config.primary_color,
            borderwidth=0,
            focuscolor="none",
            padding=(20, 10),
            font=(self.gui_config.font_family, self.gui_config.body_font_size)
        )
        
        self.style.map(
            "Modern.TButton",
            background=[
                ('active', '#0088d1'),
                ('pressed', '#0071b3'),
                ('disabled', '#cccccc')
            ]
        )
        
        # Secondary button
        self.style.configure(
            "Secondary.TButton",
            background=self.gui_config.card_background,
            foreground=self.gui_config.primary_color,
            borderwidth=0,
            focuscolor="none",
            padding=(15, 8),
            font=(self.gui_config.font_family, self.gui_config.small_font_size)
        )
        
        self.style.map(
            "Secondary.TButton",
            background=[
                ('active', '#545b62'),
                ('pressed', '#3a3f44')
            ]
        )
    
    def _configure_treeview_styles(self) -> None:
        """Configure treeview styles."""
        self.style.configure(
            "Modern.Treeview",
            background=self.gui_config.card_background,
            foreground=self.gui_config.text_primary,
            rowheight=30,
            fieldbackground=self.gui_config.card_background,
            font=(self.gui_config.font_family, self.gui_config.small_font_size),
            borderwidth=1,
            relief="solid"
        )
        
        self.style.configure(
            "Modern.Treeview.Heading",
            background="#e9ecef",
            foreground=self.gui_config.text_secondary,
            font=(self.gui_config.font_family, self.gui_config.small_font_size, 'bold'),
            relief="flat",
            borderwidth=1
        )
        
        self.style.map(
            'Modern.Treeview',
            background=[('selected', self.gui_config.primary_color)],
            foreground=[('selected', '#ffffff')]
        )
    
    def _configure_progressbar_styles(self) -> None:
        """Configure progressbar styles."""
        self.style.configure(
            "Modern.Horizontal.TProgressbar",
            background=self.gui_config.primary_color,
            troughcolor="#e9ecef",
            borderwidth=0,
            lightcolor=self.gui_config.primary_color,
            darkcolor=self.gui_config.primary_color
        )
    
    def _configure_entry_styles(self) -> None:
        """Configure entry/spinbox styles."""
        self.style.configure(
            "Modern.TSpinbox",
            padding=5,
            font=(self.gui_config.font_family, self.gui_config.body_font_size)
        )
    
    def _configure_scale_styles(self) -> None:
        """Configure scale/slider styles."""
        self.style.configure(
            "Modern.Horizontal.TScale",
            background=self.gui_config.background_color,
            troughcolor="#e9ecef"
        )
    
    def create_label(
        self,
        parent: tk.Widget,
        text: str,
        style: str = "body",
        **kwargs
    ) -> tk.Label:
        """
        Create a styled label.
        
        Args:
            parent: Parent widget
            text: Label text
            style: Style type ('title', 'subtitle', 'body', 'muted')
            **kwargs: Additional label options
        
        Returns:
            Styled tk.Label widget
        """
        font_sizes = {
            'title': (self.gui_config.font_family, self.gui_config.title_font_size, 'bold'),
            'subtitle': (self.gui_config.font_family, self.gui_config.subtitle_font_size),
            'body': (self.gui_config.font_family, self.gui_config.body_font_size),
            'small': (self.gui_config.font_family, self.gui_config.small_font_size),
            'muted': (self.gui_config.font_family, self.gui_config.small_font_size),
        }
        
        colors = {
            'title': self.gui_config.text_primary,
            'subtitle': self.gui_config.text_muted,
            'body': self.gui_config.text_secondary,
            'small': self.gui_config.text_muted,
            'muted': self.gui_config.text_muted,
        }
        
        font = font_sizes.get(style, font_sizes['body'])
        fg = colors.get(style, colors['body'])
        
        return tk.Label(
            parent,
            text=text,
            font=font,
            bg=kwargs.pop('bg', self.gui_config.background_color),
            fg=kwargs.pop('fg', fg),
            **kwargs
        )
