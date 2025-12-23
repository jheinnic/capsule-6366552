from skimage.io import imread, imsave
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from traitlets import HasTraits
import varints
import sys
import numpy as np
from skimage.io import imread, imsave
from scipy import sparse
import numpy as np
import gzip
import os
from ipywidgets import Label, HBox, VBox, HTML
from pathlib import Path
from typing import Any
import tempfile
from PIL import Image
import io

BLOCK_SIZE = 8192
LAST_INT16_SAFE_UINT16 = pow(2, 16) - 1

class BaseHeuristic():
    def __init__(self):
        self._return_widgets = []
    
    def _add_result(self, file: str, label: str):
        stats = os.stat(str(file))
        size = stats.st_size
        name_label = Label(label)
        file_label = Label(file)
#         size_label = Label(str(size))
        size_label = HTML(f"<div class='badge'>{size}</div>")
        box = HBox(children=[name_label, file_label, size_label])
        
        self._return_widgets.append({
            "widget": box,
            "label": label,
            "file": file,
            "size": size
        })
        
    def get_results(self, widgets: List) -> None:
        widgets.extend(self._return_widgets)

    
class SparseVarIntHeuristics(BaseHeuristic):
    def __init__(
        self, base_label,
        flattened_source, offset_reduced, offset_key, unique_values, naive_digitized, naive_decoder, 
        sorted_digitize_map, sorted_digitized, sorted_decoder, is_offset_byte_sparing, is_digitize_byte_sparing,
        raw_result_file, signed_offset_result_file, offset_result_file, naive_digitized_result_file,
        sorted_digitized_result_file, apply_varints, apply_sparse
    ):
        BaseHeuristic.__init__(self)
        if (not apply_varints and not apply_sparse):
            raise ValueException("Must apply varints, sparse matrices, or both!")
        self.flattened_source = flattened_source
        self.offset_reduced = offset_reduced
        self.offset_key = offset_key
        self.unique_values = unique_values
        self.naive_digitized = naive_digitized
        self.naive_decoder = naive_decoder
        self.sorted_digitize_map = sorted_digitize_map
        self.sorted_digitized = sorted_digitized
        self.sorted_decoder = sorted_decoder
        self.is_offset_byte_sparing = is_offset_byte_sparing
        self.is_digitize_byte_sparing = is_digitize_byte_sparing
        self.signed_offset_result_file = signed_offset_result_file
        self.raw_result_file = raw_result_file
        self.offset_result_file = offset_result_file
        self.naive_digitized_result_file = naive_digitized_result_file
        self.sorted_digitized_result_file = sorted_digitized_result_file
        self.apply_varints = apply_varints
        self.apply_sparse = apply_sparse
        
        
    def run_analysis(self):
        flattened_source = self.flattened_source
        offset_reduced = self.offset_reduced
        sorted_digitized = self.sorted_digitized
        naive_digitized = self.naive_digitized
        
        if self.apply_sparse:
            flattened_source = sparse.bsr_matrix(flattened_source).data.flatten()
            offset_reduced = sparse.bsr_matrix(offset_reduced).data.flatten()
            naive_digitized = sparse.bsr_matrix(naive_digitized).data.flatten()
            sorted_digitized = sparse.bsr_matrix(sorted_digitized).data.flatten()
            
        if self.apply_varints:
            if self.apply_sparse:
                test_name = "sparse-varint"
            else:
                test_name = "varint"
            self.varint_strategies = [
                { "name": "sqliteu", "impl": varints.sqliteu },
                { "name": "dlugoszu", "impl": varints.dlugoszu }
            ]
            self.execute_trial(flattened_source, self.raw_result_file, test_name=test_name)
            
            if not self.is_offset_byte_sparing:
                self.execute_trial(offset_reduced, self.offset_result_file, test_name=f"{test_name}-offset")
            else:
                print("Offset strategy was byte sparing.  No reason to attempt varint reduction")
                
            if not self.is_digitize_byte_sparing:
                self.execute_trial(
                    sorted_digitized, self.sorted_digitized_result_file, test_name=f"{test_name}-sort-digi")
                self.execute_trial(
                    naive_digitized, self.naive_digitized_result_file, test_name=f"{test_name}-naive-digi")
            else:
                print("Digitize strategy was byte sparing.  No reason to attempt varint reduction")
        
            if self.sorted_digitize_map[0] > self.unique_values[0]:
                print("Most common value was not the smallest value.  Normalizing the smallest value to zero may not have produced as sparse of a file as normalizing to the most common value.  Attempting to normalize to the most common value will store negative values, which may require a specific varint format to tolerate.");
                self.varint_strategies = [
                    { "name": "sqliteu", "impl": varints.sqliteu },
                    { "name": "leb128s", "impl": varints.dlugoszu }
                ]
                self.apply_alternate_offset_strategy()
            else:
                print("Alternate offset reduction does not apply.  Skipping it...")
        else:
            self.save_sparse_data(flattened_source, self.raw_result_file, "sparse" )
            self.save_sparse_data(offset_reduced, self.offset_result_file, "sparse-offset")
            self.save_sparse_data(sorted_digitized, self.sorted_digitized_result_file, "sparse-sort-digi")
            self.save_sparse_data(naive_digitized, self.naive_digitized_result_file, "sparse-naive-digi")
        
    def apply_alternate_offset_strategy(self):
        print("This routine is TODO")
        
    def save_sparse_data(self, source_array, base_filename, label_suffix):
        with gzip.open(base_filename + ".dat", 'wb', compresslevel=9) as woo:
            woo.write(source_array.tobytes())
        self._add_result(base_filename + ".dat", label_suffix)
    
    def execute_trial(self, source_ndarray: np.ndarray, result_file: str, test_name:str="varint", read_verify:bool=True):
        """
        Apply each candidate varint algoritm to the given condensed byte sequence.  Report on the sizing
        of each any ensure that each algorithm's work is reversible without any knowledge other than
        which algorithm was used to condense the bit sequence.
        """
        as_list = source_ndarray.tolist()
        full_size = len(source_ndarray)
        padding = (full_size % BLOCK_SIZE)
        if (padding > 0):
            # Add padding to get an even multiple of BLOCK_SIZE.  We'll trim this away from the final buffer later.
            padding = BLOCK_SIZE - padding
            as_list.extend([0] * padding)
            full_size = full_size + padding
            source_ndarray = np.array(as_list, np.uint16)
        block_count: int = round(full_size / BLOCK_SIZE)
        algo_idx = 0
        for varint_algo in self.varint_strategies:
            algo_name = varint_algo["name"]
            varint_algo = varint_algo["impl"]
            algo_result_file = f"{str(result_file)}-{algo_name}.dat"
            approx = 0
            block_idx = 0
            last_block = block_count - 1
            with gzip.open(algo_result_file, 'wb', compresslevel=9) as woo:
                for chunk in np.split(source_ndarray, block_count):
                    block_idx += 1
                    if block_idx == last_block:
                        chunk = chunk[:(-1 * padding)]
                    var = varint_algo.encode(chunk.tolist())
                    approx += sys.getsizeof(var)
                    woo.write(var)
            print(f"|{approx}|{algo_name}|{result_file}|")
            if read_verify:
                with gzip.open(algo_result_file, 'rb', compresslevel=9) as woo:
                    var = woo.read()
                decoded_bytes = varint_algo.decode(var)
                rehydrated_png = np.array([i for i in decoded_bytes], np.uint16)
                print(source_ndarray.shape)
                print(rehydrated_png.shape)
                print(source_ndarray.dtype)
                print(rehydrated_png.dtype)
                #if not (source_ndarray == rehydrated_png):
                    #raise ValueError(f"Comparison on reading back varint encoding failed")
            self._add_result(algo_result_file, f"{test_name}-{algo_name}")
        

class ImageFileHeuristics(BaseHeuristic):
    def __init__(
        self, source_file_name, width: int, height: int, base_label: str, flattened_source, 
        raw_result_file,  offset_result_file, naive_digitized_result_file, sorted_digitized_result_file, 
    ):
        BaseHeuristic.__init__(self)
        self.source_file_name = source_file_name
        self.width = width
        self.height = height
        self.base_label = base_label
        self.flattened_source = flattened_source
        self.raw_result_file = raw_result_file
        self.offset_result_file = offset_result_file
        self.naive_digitized_result_file = naive_digitized_result_file
        self.sorted_digitized_result_file = sorted_digitized_result_file
        
        self.spread: int  = -1
        self.offset_reduced: Optional[np.ndarray] = None
        self.offset_key: Any = None
            
        self.unique_values: Optional[np.ndarray] = None
        self.num_unique_values: int = -1
            
        self.naive_digitize_map: Optional[np.ndarray] = None
        self.naive_digitized_reduced: Optional[np.ndarray] = None
        self.naive_decoder_key: Any = None
            
        self.sorted_digitize_map: Optional[np.ndarray] = None
        self.sorted_digitized_reduced: Optional[np.ndarray] = None
        self.sorted_decoder_key: Any = None
            
        self.is_offset_byte_sparing: bool = False
        self.is_digitize_byte_sparing = False
        self.is_signed_offset_more_sparse: bool = False
            
        self.signed_offset_reduced: Optional[np.ndarray]
        self.signed_offset_key: Any = None
            
    def to_delta_heuristics(self) -> "Optional[ImageFileHeuristics]":
        """
        Shift the stored representation into 'delta' format.  Pop one value off the end of the array and
        shift a 0 in the replace it on the other end.  Then take the difference between this array and its
        altered clone.  A cumulative sum operation on the resulting array will restore the original array.
        In cases where the source image is fairly homogenous, the common tendency is for each pixel's new
        value to be a litle bit above or below zero--the distribution of values tightens over a smaller 
        band.  For this operation to be safe, the array data type must become signed, and the magnitude of
        the greatest pixel value must not present a genuined thread to ovrflow/underflow .
        """
        if np.max(self.flattened_source) > LAST_INT16_SAFE_UINT16:
            return None
        offset = self.flattened_source.tolist().copy()
        check_val = offset.pop()
        offset.insert(0, 0)
        offset_array = np.array(offset, dtype=np.int16)
        data = self.flattened_source.astype(np.int16, 'C', 'unsafe')
        result = data - offset_array
        return ImageFileHeuristics(
            self.source_file_name,
            self.width, self.height,
            "delta", result,
            self.raw_result_file + "-delta",
            self.offset_result_file + "-delta",
            self.naive_digitized_result_file + "-delta",
            self.sorted_digitized_result_file + "-delta"
        )
        
    def run_analysis(self):
        self.spread = np.ptp(self.flattened_source)
        min = np.min(self.flattened_source)
        self.offset_key = {'algorithm': 'offset', 'param': self.spread}
        reduced = self.flattened_source - min
        print(f"Offset analysis: spread={self.spread}, min={min}, max={np.max(self.flattened_source)}")
        if self.spread < 256 and min >= 0:
            # Conversion from 16-bit ints to 8-bit ints has potential to
            # be lossy, but we have just confirmed the peak to peak value
            # spread is <256, so just normalizing to 0 yields a result that
            # fits nicely into unsigned 8-bit encoding.
            print(f"Reducing to 8-bit PNG because this image's dynamic range is {self.num_unique_values} from hi to lo")
            self.is_offset_byte_sparing = True
            self.offset_reduced = reduced.astype(np.uint8, 'C', 'unsafe')
            self._save_compact_image(self.offset_reduced, self.offset_result_file, f"{self.base_label}-offset")
        elif min != 0:
            self.is_offset_byte_sparing = False
            self.offset_reduced = reduced
            self._save_compact_image(self.offset_reduced, self.offset_result_file, f"{self.base_label}-offset")
        else:
            # Ignore the offset method entirely if its just going to be identical to the
            # unprocessed baseline and not save on pixel depth.
            self.offset_reduced = None
            self.offset_key = None
        
        self.unique_values = np.unique(self.flattened_source)
        self.num_unique_values = len(self.unique_values)
        self.naive_digitize_map = {self.unique_values[ii]: ii for ii in range(0, self.num_unique_values)}
        reduced = [self.naive_digitize_map[p] for p in self.flattened_source]
        if self.num_unique_values < 256:
            # peak-to-peak spread went beyond 256, but fewer than 256 unique values are used within that spread, so
            # we can build can eight-bit digitzation map and order does not matter because every value used will get
            # an eight-bit assignment.
            print(f"Reducing to 8-bit PNG because this image only uses {self.num_unique_values} distinct pixel values")
            self.is_digitize_byte_sparing = True
            self.naive_digitized_reduced = np.array(reduced, dtype=np.uint8)
            self.naive_decoder_key = {'algorithm': 'digitize8', 'param': self.unique_values.copy()}
        else:
            self.is_digitize_byte_sparing = False
            self.naive_digitized_reduced = np.array(reduced, dtype=np.uint16)
            self.naive_decoder_key = {'algorithm': 'digitize16', 'param': self.unique_values.copy()}
        self._save_compact_image(
            self.naive_digitized_reduced, self.naive_digitized_result_file, f"{self.base_label}-naive-digi")
            
        # No eight-bit solutions exist, so are left with one that will take 16 bits with a fixed encoding.  There
        # will be no point using this with an image container encoding where every pixel must have the same bit
        # count, but we calculare it anyhow for the benefit of varint encoders that will use these routines to
        # begin optimzation from a local optimum.  Also because we curiously enough do still see a small benefit
        # in the data reporting even though we probably should not see any benefit at all.
        #.In order to get the most out of varint encoding, we do want to sort the unique domain of pixel values
        # in use in decreasing order of frequency, so those values that appear most are the ones first selected to
        # receive an eight bit value in the pixel map.
        (self.sorted_digitize_map, decoder) = self._compute_pixel_map()
        reduced = [self.sorted_digitize_map[p] for p in self.flattened_source]
        if self.num_unique_values < 256:
            print(f"Reducing to 8-bit PNG because this image only uses {self.num_unique_values} distinct pixel values")
            self.sorted_digitized_reduced = np.array(reduced, dtype=np.uint8)
            self.sorted_decoder_key = {'algorithm': 'digitize8', 'param': decoder}
        else:
            self.sorted_digitized_reduced = np.array(reduced, dtype=np.uint16)
            self.sorted_decoder_key = {'algorithm': 'digitize16', 'param': decoder}
        self._save_compact_image(
            self.sorted_digitized_reduced, self.sorted_digitized_result_file, f"{self.base_label}-sorted-digi")

    def _compute_pixel_map(self) -> Tuple:
        np.sort(self.unique_values)
        bin_boundaries = self.unique_values.tolist().copy()
        bin_boundaries.append(np.max(self.unique_values) + 5)
        histo = np.histogram(self.flattened_source, bins=bin_boundaries, density=False)
        value_frequency = np.array([(ii[0], 0-ii[1]) for ii in zip(histo[1], histo[0])], 
                                   dtype=[("value", int), ("freq", int)])
        value_frequency.sort(order="freq")
        decoder = [ii[0] for ii in value_frequency]
        digitize_map = {decoder[ii]: ii for ii in range(0, self.num_unique_values)}
        return (digitize_map, decoder) 

    def _alt_compute_pixel_map(self, flattened_source) -> Tuple:
        unique_values = np.unique(flattened_source)
        np.sort(unique_values)
        bin_boundaries = unique_values.tolist().copy()
        bin_boundaries.append(np.max(unique_values) + 1)
        histo = np.histogram(flattened_source, bins=bin_boundaries, density=False)
        value_frequency = np.array(
            [(ii[0], 0-ii[1]) for ii in zip(histo[1], histo[0])], 
            dtype=[("value", int), ("freq", int)])
        value_frequency.sort(order="freq")
        decoder = [ii[0] for ii in value_frequency]
        digitize_map = {decoder[ii]: ii for ii in range(0, len(unique_values))}
        return (digitize_map, decoder, value_frequency) 
    
    def _save_compact_image(self, reduced: np.ndarray, dst: str, label: str) -> None:
        reduced: np.ndarray = reduced.reshape([self.height, self.width])
        imsave(str(dst) + ".png", reduced)
        self._add_result(str(dst) + ".png", f"{self.base_label}-offset")

    def expand_offset(self, compressed: np.ndarray, offset: int) -> np.ndarray:
        return compressed + offset

    def expand_digitize8(self, compressed: np.ndarray, decoder_map: List) -> np.ndarray:
        return decoder_map[compressed]

    def expand_digitize16(self, compressed: np.ndarray, decoder_map: List) -> np.ndarray:
        return decoder_map[compressed]

    EXPANDERS = {
        'offset': expand_offset,
        'digitize8': expand_digitize8,
        'digitize16': expand_digitize16
    }

    def expand(self, tgt: str, dst: str, key) -> None:
        if key['algorithm'] is None:
            return
        compressed_data = imread(tgt)
        try:
            expanded_data = EXPANDERS[key['algorithm']](
                compressed_data, key['param']
            )
            expanded_data = expanded_data.astype(np.uint16, 'C', 'safe')
        except KeyError as e:
            return
        imsave(dst, expanded)
        
    def to_varint_sparse_heuristics(self, is_varint_used: bool, is_sparse_used: bool):
        if is_varint_used:
            if is_sparse_used:
                suffix = "-sparse-varint"
            else:
                suffix = "-varint"
        elif is_sparse_used:
            suffix = "-sparse"
        else:
            raise ValueException("Either varint or sparse mode or both is requires")
            
        return SparseVarIntHeuristics(
            self.base_label,
            self.flattened_source,
            self.offset_reduced,
            self.offset_key,
            self.unique_values,
            self.naive_digitized_reduced,
            self.naive_decoder_key, 
            self.sorted_digitize_map,
            self.sorted_digitized_reduced,
            self.sorted_decoder_key,
            self.is_offset_byte_sparing,
            self.is_digitize_byte_sparing,
            self.raw_result_file + suffix,
            self.offset_result_file + "-signed" + suffix,
            self.offset_result_file + suffix,
            self.naive_digitized_result_file + suffix,
            self.sorted_digitized_result_file + suffix,
            is_varint_used, is_sparse_used
        )
    

class VisibleOptimizer(HasTraits):
    def __init__(self, source_file_name, source_file_data, progress_bar, width=1512, height=2016):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_root_path = Path(self.temp_dir.name)
        self.source_file = self.temp_root_path / "original_source.png"
        source_file_data = Image.open(io.BytesIO(source_file_data))
        source_file_data = np.asarray(source_file_data)
        source_file_data = source_file_data.reshape([height, width])
        imsave(self.source_file, source_file_data)
        self.original_content = imread(self.source_file)
        assert (self.original_content == source_file_data).all()
        self.source_file_name = source_file_name
        self.progress_bar = progress_bar
        self.image_heuristics = ImageFileHeuristics(
            source_file_name, width, height, "prime",
            self.original_content.flatten(),
            str(self.temp_root_path / "baseline_content"),
            str(self.temp_root_path / "offset_reduced"),
            str(self.temp_root_path / "naive_digitized_reduced"),
            str(self.temp_root_path / "sorted_digitized_reduced")
        )
        self.delta_image_heuristics = self.image_heuristics.to_delta_heuristics()
        # Varint and Sparse variants of baseline and delta transforms need the output of those tests to
        # define themselves, so we have to defer their construction until during the analysis pass.  The
        # delta variants may not be feasible for all inputs and so may not get created at all.
        self.varint_image_heuristcs = None
        self.sparse_image_heuristics = None
        self.sparse_varint_image_heuristics = None
        self.varint_delta_image_heuristics = None
        self.sparse_delta_image_heuristics = None
        self.sparse_varint_delta_image_heuristics = None
        
    def run_analysis(self):
        self.progress_bar.value = 0
        self.image_heuristics.run_analysis()
        self.varint_image_heuristcs = self.image_heuristics.to_varint_sparse_heuristics(True, False)
        self.sparse_varint_image_heuristics = self.image_heuristics.to_varint_sparse_heuristics(True, True)
        self.sparse_image_heuristics = self.image_heuristics.to_varint_sparse_heuristics(False, True)
        self.progress_bar.value = 1
        if not self.delta_image_heuristics is None:
            self.delta_image_heuristics.run_analysis()
            self.varint_delta_image_heuristics = \
                self.delta_image_heuristics.to_varint_sparse_heuristics(True, False)
            self.sparse_varint_delta_image_heuristics = \
                self.delta_image_heuristics.to_varint_sparse_heuristics(True, True)
            self.sparse_delta_image_heuristics = \
                self.delta_image_heuristics.to_varint_sparse_heuristics(False, True)
        self.progress_bar.value = 2
        self.sparse_image_heuristics.run_analysis()
        self.progress_bar.value = 3
        if not self.delta_image_heuristics is None:
            self.sparse_delta_image_heuristics.run_analysis()
        self.progress_bar.value = 4
        self.sparse_varint_image_heuristics.run_analysis()
        self.progress_bar.value = 5
        if not self.delta_image_heuristics is None:
            self.sparse_varint_delta_image_heuristics.run_analysis()
        self.progress_bar.value = 6
        self.varint_image_heuristcs.run_analysis()
        self.progress_bar.value = 7
        if not self.delta_image_heuristics is None:
            self.varint_delta_image_heuristics.run_analysis()
        self.progress_bar.value = 8
     
    def collect_widgets(self):
        retval = []
        self.image_heuristics.run_analysis(retval)
        self.sparse_image_heuristics.run_analysis(retval)
        self.varint_image_heuristcs.run_analysis(retval)
        self.sparse_varint_image_heuristics.run_analysis(retval)
        if not self.delta_image_heuristics is None:
            self.delta_image_heuristics.run_analysis(retval)
            self.sparse_delta_image_heuristics.run_analysis(retval)
            self.varint_delta_image_heuristics.run_analysis(retval)
            self.sparse_varint_delta_image_heuristics.run_analysis(retval)
        return retval
        
