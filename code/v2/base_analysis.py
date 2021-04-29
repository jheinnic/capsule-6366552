import os
from typing import List, Optional, Dict
from ipywidgets import Label, HBox, VBox, HTML

class BaseAnalysis():
    def __init__(self):
        self._return_widgets = []
        self.pixel_value_frequencies: Optional[Dict] = None
    
    def _add_result(self, file: str, label: str):
        stats = os.stat(str(file))
        size = stats.st_size
        name_label = Label(label)
        file_label = Label(file)
        size_label = HTML(f"<div class='badge'>{size}</div>")
        box = HBox(children=[name_label, file_label, size_label])
        
        self._return_widgets.append({
            "widget": box,
            "label": label,
            "file": file,
            "size": size,
            "frequencies": self.pixel_value_frequencies if self.pixel_value_frequencies is not None else {}
        })
        
    def get_results(self, widgets: List) -> None:
        widgets.extend(self._return_widgets)
