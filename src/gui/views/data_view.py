"""
Data view component for displaying detection results.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, List, Dict, Any
import pandas as pd

from ...config import Settings
from ..components.toolbar import Toolbar


class DataView(ttk.Frame):
    """
    Data grid view for displaying detection results.
    
    Features:
    - Treeview table with sortable columns
    - Horizontal and vertical scrollbars
    - Export and reset toolbar
    - Row selection
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_export: callable,
        on_reset: callable,
        on_settings: callable,
        settings: Optional[Settings] = None,
        **kwargs
    ):
        """
        Initialize the data view.
        
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
        
        self._dataframe: Optional[pd.DataFrame] = None
        self._create_widgets()
    
    def _create_widgets(self) -> None:
        """Create the data view widgets."""
        # Toolbar
        self.toolbar = Toolbar(
            self,
            on_export=self.on_export,
            on_reset=self.on_reset,
            on_settings=self.on_settings,
            settings=self.settings
        )
        self.toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # Data grid container
        grid_frame = ttk.Frame(self, style="Card.TFrame")
        grid_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview
        self.tree = ttk.Treeview(grid_frame, style="Modern.Treeview")
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(grid_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scroll = ttk.Scrollbar(grid_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Layout
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        grid_frame.grid_rowconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(0, weight=1)
    
    def load_data(self, data: pd.DataFrame) -> None:
        """
        Load data into the treeview.
        
        Args:
            data: DataFrame to display
        """
        self._dataframe = data
        
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Configure columns
        columns = list(data.columns)
        self.tree["columns"] = columns
        self.tree["show"] = "headings"
        
        for col in columns:
            self.tree.heading(col, text=col, anchor=tk.W)
            self.tree.column(col, width=150, anchor=tk.W, minwidth=100)
        
        # Insert data rows
        for _, row in data.iterrows():
            values = []
            for val in row:
                # Convert lists to strings for display
                if isinstance(val, list):
                    val = ", ".join(str(v) for v in val)
                values.append(val)
            self.tree.insert("", "end", values=values)
    
    def load_detections(self, detections: List[Dict[str, Any]]) -> None:
        """
        Load detection results into the view.
        
        Args:
            detections: List of detection dictionaries
        """
        df = pd.DataFrame(detections)
        
        # Flatten nested fields for display
        if 'texts_inside' in df.columns:
            df['texts_inside'] = df['texts_inside'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else str(x)
            )
        
        if 'bbox_pdf' in df.columns:
            df['bbox_pdf'] = df['bbox_pdf'].apply(
                lambda x: str(x) if isinstance(x, list) else str(x)
            )
        
        if 'bbox_pixels' in df.columns:
            df['bbox_pixels'] = df['bbox_pixels'].apply(
                lambda x: str(x) if isinstance(x, list) else str(x)
            )
        
        self.load_data(df)
    
    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """Get the current dataframe."""
        return self._dataframe
    
    @property
    def row_count(self) -> int:
        """Get number of data rows."""
        return len(self._dataframe) if self._dataframe is not None else 0
    
    @property
    def column_count(self) -> int:
        """Get number of columns."""
        return len(self._dataframe.columns) if self._dataframe is not None else 0
    
    def get_selected_rows(self) -> List[Dict[str, Any]]:
        """Get currently selected rows as dictionaries."""
        selected = []
        for item in self.tree.selection():
            values = self.tree.item(item)['values']
            if self._dataframe is not None:
                row_dict = dict(zip(self._dataframe.columns, values))
                selected.append(row_dict)
        return selected
