from typing import Any, List, Optional, Tuple
from skimage.io import imsave
import numpy as np

from base_analysis import BaseAnalysis
from sparse_varint import SparseVarintAnalysis

LAST_INT16_SAFE_UINT16 = pow(2, 16) - 1

class DigitizeImageAnalysis(BaseAnalysis):
    def __init__(
        self, source_file_name, width: int, height: int, base_label: str, flattened_source, is_signed: bool, 
        raw_result_file,  offset_result_file, naive_digitized_result_file, sorted_digitized_result_file, 
    ):
        BaseAnalysis.__init__(self)
        self.source_file_name = source_file_name
        self.width = width
        self.height = height
        self.base_label = base_label
        self.flattened_source = flattened_source
        self.raw_result_file = raw_result_file
        self.naive_digitized_result_file = naive_digitized_result_file
        self.sorted_digitized_result_file = sorted_digitized_result_file
        self.unique_values: Optional[np.ndarray] = None
        self.num_unique_values: int = -1
        self.naive_digitize_map: Optional[np.ndarray] = None
        self.naive_digitized_reduced: Optional[np.ndarray] = None
        self.naive_decoder_key: Any = None
        self.sorted_digitize_map: Optional[np.ndarray] = None
        self.sorted_digitized_reduced: Optional[np.ndarray] = None
        self.sorted_decoder_key: Any = None
        self.is_digitize_byte_sparing = False
            
    def run_analysis(self):        
        self.unique_values = np.unique(self.flattened_source)
        self.num_unique_values = len(self.unique_values)
        self.naive_digitize_map = {self.unique_values[ii]: ii for ii in range(0, self.num_unique_values)}
        reduced = [self.naive_digitize_map[p] for p in self.flattened_source]
        if self.num_unique_values < 256:
            # Fewer than 256 unique values are used within this file.
            # We can build an eight-bit digitzation map and order should not matter because no value
            # will get a sixteen-bit assignment.
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
            
        # In order to get most out of varint encoding, we sort unique domain of pixel values
        # in decreasing order of frequency, so those values that appear most often ones first selected to
        # receive smallest values in the pixel map.
        # There is a case to be made to use a Hamiltonian cycle of single bit toggles instead, presuming that
        # spreading out the entropy might favor better compression.  Or perhaps using all the values with a
        # single one bit, then those with two, then three, etc. etc.
        (self.sorted_digitize_map, decoder, self.pixel_value_frequencies) = self._compute_pixel_map()
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
        bin_boundaries.append(np.max(self.unique_values) + 1)
        histo = np.histogram(self.flattened_source, bins=bin_boundaries, density=False)
        value_frequency = np.array(
            [(ii[0], 0-ii[1]) for ii in zip(histo[1], histo[0])], 
            dtype=[("value", int), ("freq", int)])
        value_frequency.sort(order="freq")
        decoder = [ii[0] for ii in value_frequency]
        digitize_map = {decoder[ii]: ii for ii in range(0, self.num_unique_values)}
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
        
    def to_varint_sparse_analysis(self, is_varint_used: bool, is_sparse_used: bool):
        if is_varint_used:
            if is_sparse_used:
                pass
        elif is_sparse_used:
            pass
        else:
            raise ValueException("Either varint or sparse mode or both is requires")
            
        return SparseVarintAnalysis(
            self.base_label,
            self.flattened_source,
            self.width, self.height,
            self.unique_values,
            self.naive_digitized_reduced,
            self.naive_decoder_key, 
            self.sorted_digitize_map,
            self.sorted_digitized_reduced,
            self.sorted_decoder_key,
            self.is_digitize_byte_sparing,
            self.raw_result_file,
            self.naive_digitized_result_file,
            self.sorted_digitized_result_file,
            is_varint_used, is_sparse_used
        )
    
