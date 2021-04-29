from skimage.io import imread, imsave
from pathlib import Path
from traitlets import HasTraits
import tempfile

from digitize_analysis import DigitizeImageAnalysis

class AnalysisDriver(HasTraits):
    def __init__(self, source_file_name, source_file_data, progress_bar, width=1512, height=2016):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root_path = Path(self.temp_dir.name)
        # self.source_file = self.temp_root_path / "original_source.png"
        # source_file_data = source_file_data.reshape([height, width])
        # imsave(self.source_file, source_file_data)
        # self.original_content = imread(self.source_file)
        # assert (self.original_content == source_file_data).all()
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
        self.delta_image_analysis = self.image_analysis.to_delta_analysis()
        # Varint and Sparse variants of baseline and delta transforms need the output of those tests to
        # define themselves, so we have to defer their construction until during the analysis pass.  The
        # delta variants may not be feasible for all inputs and so may not get created at all.
        self.varint_image_heuristcs = None
        self.sparse_image_analysis = None
        self.sparse_varint_image_analysis = None
        self.varint_delta_image_analysis = None
        self.sparse_delta_image_analysis = None
        self.sparse_varint_delta_image_analysis = None
        
    def run_analysis(self):
        self.progress_bar.value = 0
        self.image_analysis.run_analysis()
        self.varint_image_heuristcs = self.image_analysis.to_varint_sparse_analysis(True, False)
        self.sparse_varint_image_analysis = self.image_analysis.to_varint_sparse_analysis(True, True)
        self.sparse_image_analysis = self.image_analysis.to_varint_sparse_analysis(False, True)
        self.progress_bar.value = 1
        if not self.delta_image_analysis is None:
            self.delta_image_analysis.run_analysis()
            self.varint_delta_image_heuristcs = \
                self.delta_image_analysis.to_varint_sparse_analysis(True, False)
            self.sparse_varint_delta_image_analysis = \
                self.delta_image_analysis.to_varint_sparse_analysis(True, True)
            self.sparse_delta_image_analysis = \
                self.delta_image_analysis.to_varint_sparse_analysis(False, True)
        self.progress_bar.value = 2
        self.sparse_image_analysis.run_analysis()
        self.progress_bar.value = 3
        if not self.delta_image_analysis is None:
            self.sparse_delta_image_analysis.run_analysis()
        self.progress_bar.value = 4
        self.sparse_varint_image_analysis.run_analysis()
        self.progress_bar.value = 5
        if not self.delta_image_analysis is None:
            self.sparse_varint_delta_image_analysis.run_analysis()
        self.progress_bar.value = 6
        self.varint_image_heuristcs.run_analysis()
        self.progress_bar.value = 7
        if not self.delta_image_analysis is None:
            self.varint_delta_image_analysis.run_analysis()
        self.progress_bar.value = 8
     
    def collect_widgets(self):
        retval = []
        self.image_analysis.run_analysis(retval)
        self.sparse_image_analysis.run_analysis(retval)
        self.varint_image_heuristcs.run_analysis(retval)
        self.sparse_varint_image_analysis.run_analysis(retval)
        if not self.delta_image_analysis is None:
            self.delta_image_analysis.run_analysis(retval)
            self.sparse_delta_image_analysis.run_analysis(retval)
            self.varint_delta_image_analysis.run_analysis(retval)
            self.sparse_varint_delta_image_analysis.run_analysis(retval)
        return retval