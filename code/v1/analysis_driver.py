from skimage.io import imread, imsave
from pathlib import Path
from traitlets import HasTraits
import tempfile

from digitize_analysis import DigitizeImageAnalysis

class AnalysisDriver(HasTraits):
    def __init__(self, source_file_name, source_file_data, progress_bar, width=1512, height=2016):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root_path = Path(self.temp_dir.name)
        self.original_content = source_file_data
        self.source_file_name = source_file_name
        self.progress_bar = progress_bar
        self.image_analysis = DigitizeImageAnalysis(
            source_file_name, width, height, "prime",
            self.original_content, False,
            str(self.temp_root_path / "baseline_content"),
            str(self.temp_root_path / "offset_reduced"),
            str(self.temp_root_path / "naive_digitized_reduced"),
            str(self.temp_root_path / "sorted_digitized_reduced")
        )
        # Varint and Sparse variants need the output of baseline tests to
        # define themselves, so we have to defer their construction until during analysis.
        self.varint_image_heuristcs = None
        self.sparse_image_analysis = None
        self.sparse_varint_image_analysis = None
        
    def run_analysis(self):
        self.progress_bar.value = 0
        self.image_analysis.run_analysis()
        self.progress_bar.value = 1
        self.varint_image_heuristcs = self.image_analysis.to_varint_sparse_analysis(True, False)
        self.sparse_varint_image_analysis = self.image_analysis.to_varint_sparse_analysis(True, True)
        self.sparse_image_analysis = self.image_analysis.to_varint_sparse_analysis(False, True)
        self.progress_bar.value = 2
        self.sparse_image_analysis.run_analysis()
        self.progress_bar.value = 3
        self.sparse_varint_image_analysis.run_analysis()
        self.progress_bar.value = 4
        self.varint_image_heuristcs.run_analysis()
        self.progress_bar.value = 5
     
    def collect_widgets(self):
        retval = []
        self.image_analysis.run_analysis(retval)
        self.sparse_image_analysis.run_analysis(retval)
        self.varint_image_heuristcs.run_analysis(retval)
        self.sparse_varint_image_analysis.run_analysis(retval)
        self.progress_bar.value = 6
        return retval