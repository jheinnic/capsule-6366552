import pandas as pd
from traitlets import Configurable, Boolean, Instance, Set, Unicode, UseEnum
from analysis_driver import AnalysisDriver

shape_labels = ["nomods", "naive", "sorted", "gcode"]
matrix_labels = ["dense", "sparse"]
encode_labels = ["protoint", "binary", "varint"]
matrix_prefixes = [
    f"{shape}_{matrix}_{encode}" 
        for matrix in matrix_labels
        for shape in shape_labels 
        for encode in encode_labels 
]
ALL_PREFIXES = [f"{shape}_png" for shape in shape_labels] + matrix_prefixes
MEASURES = ["size", "time"]
FEATURES = [
    f"{prefix}_{measure}"
        for prefix in ALL_PREFIXES
        for measure in MEASURES
]

# class PendingAnalysis():
#     def __init__(source_file, size, analysis, histogram=None):
#         self._source_file = source_file
#         self._histogram = histogram
#         self._analysis = analysis
#         self._size = size


class Store(Configurable):
    persistence_file = Unicode("/var/lib/file_tests.dat", config=True)
    results_df = Instance(klass=pd.DataFrame, args=({}))
    pending_analysis = Set(trait=Instance(klass=AnalysisDriver, default_value=set())
    show_log = Boolean(False)
    show_progress = Boolean(True)
    current_activity = UseEnum()                                 
                                      
    @default("results_df")
    def _reload_results(self):
        return pd.read_parquet(
            self.persistence_file,
            engine = "auto", 
            columns = FEATURES + ['source_file', 'unique_dir'])
