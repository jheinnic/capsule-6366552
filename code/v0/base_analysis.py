import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict
from ipywidgets import Label, HBox, VBox, HTML
import numpy as np


class VarintMode(Enum):
    USE_BYTES = 0,
    USE_PROTO_INT = 1,
    USE_PY_VARINT = 2;

    
class BaseAnalysis(ABC):
    def __init__(self, source_file_name: str):
        self._source_file_name = source_file_name
        self._returnable_results = []
        self._time_pending = []
        self._pixel_values_histogram: List = []
    
    def _add_result(self, file: str, label: str, data: Optional[np.ndarray] = None):
        file = str(file)
        stats = os.stat(file)
        size = stats.st_size
        
        self._time_pending.append({
            "msg_type": "reduction_result",
            "source": self._source_file_name,
            "label": label,
            "result": file,
            "size": size
        })
        if len(self._pixel_values_histogram) == 2:
            self._returnable_results.append({
                "msg_type": "histogram",
                "source": self._source_file_name,
                "label": label,
                "result": file,
                "histogram": self._pixel_values_histogram
            })
            self._pixel_values_histogram = []
        if not data is None:
            self._returnable_results.append({
                "msg_type": "shape_result",
                "source": self._source_file_name,
                "label": label,
                "result": file,
                "data": data
            })
        
    def _add_timing(self, time_points: List) -> bool:
        if len(time_points) < 2:
            print(f"Cannot use {time_points}--not enough values to make an interval...");
            return False
        time_clone = [ii for ii in time_points]
        previous_clock = time_clone.pop()
        while (len(time_clone) > 0 and len(self._time_pending) > 0):
            next_clock = previous_clock;
            previous_clock = time_clone.pop();
            this_result = self._time_pending.pop();
            this_result["time"] = next_clock - previous_clock;
            self._returnable_results.append(this_result);
        return len(time_clone) == len(self._time_pending) == 0
      
#     def _add_multi_file_result(self, files: List[str], label: str):
#         size = 0
#         for file in files:
#             size += os.stat(file).st_size
            
#         name_label = Label(label)
#         list_items = [f'<a href="#" class="list-group-item list-group-item-action active">{str(file)}</a><span class="badge badge-pill badge-primary pull-right">{os.stat(file).st_size}</span>' for file in files]
#         file_label = HTML('<div class="list-group">' + '\r'.join(list_items) + '</div>')
#         box = VBox(children=[name_label, file_label])
        
#         self._returnable_results.append({
#             "widget": box,
#             "name": name_label,
#             "files": files,
#             "size": size,
#             "frequencies": self.pixel_value_frequencies if self.pixel_value_frequencies is not None else {}
#         })

    def _compute_histogram(flat_source: np.ndarray, unique_source_sorted: Optional[np.ndarray] = None) -> (np.ndarray, np.ndarray):
        if unique_source_sorted is None:
            unique_source_sorted = flat_source
        bin_count = len(self.unique_values)
        np.sort(self.unique_values)
        bin_boundaries = self.unique_values.copy()
        bin_boundaries.resize(bin_count + 1)
        bin_boundaries[bin_count] = self.unique_values[bin_count - 1] + 1
        self._pixel_values_histogram = np.histogram(self.flattened_source, bins=bin_boundaries, density=False)
        return self._pixel_values_histogram
      
    @abstractmethod
    def run_analysis(self) -> bool:
        """
        The Analysis class is much like an Iterator.  Each call to this method performs an incrmenetal portion of analysis work,
        then returns True if there is more work to be done through additional calls, or False if no work remains and a call can
        now be made to get_results(result_list)
        """
        pass
    
    def get_results(self, result_list: List) -> None:
        result_list.extend(self._returnable_results)
