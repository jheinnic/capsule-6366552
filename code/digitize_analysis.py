import asyncio
import math
from typing import Any, List, Optional, Tuple
from skimage.io import imread, imsave
import numpy as np
import timeit

from output_logger import getLogger
from base_analysis import BaseAnalysis, VarintMode
from sparse_varint import SparseVarintAnalysis


LOGGER = getLogger(__name__)

LAST_INT16_SAFE_UINT16 = pow(2, 16) - 1

def build_scatter_digi_map():
    """
    Build a gray code inspired Hamiltonian cycle through all 16-bit numbers.  This will be used in descending frequency
    order to buid the 'gcode' style bitmap.  Low order bits toggle much earlier than high order bits to preserve a
    measure of varint-friendly encoding.
    """
    count_up = [(ii ^ (ii >> 1)) for ii in range(0, 65536)]
    count_down = [ii for ii in count_up]
    down_runs = [count_down[64*i:64*(i+1)] for i in range(0,1024)]
    alt_down = []
    for ii in range(0, 1024):
        down_runs[ii].reverse()
        alt_down.extend(down_runs[ii])
    return [count_up[ii] if (ii%2 == 0) else alt_down[ii-1] for ii in range(0,65536)]


SCATTER_DIGI_MAP = build_scatter_digi_map()

class DigitizeImageAnalysis(BaseAnalysis):
    def __init__(
        self, source_file_name: str, source_file_bytes: bytes, result_file_template: str, read_validate: bool = True
    ):
        BaseAnalysis.__init__(self, source_file_name)
        self._source_file_bytes: bytes = source_file_bytes
        self._result_file_template: str = result_file_template
        self._read_validate = read_validate
        self.flattened_source: Optional[np.ndarray] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.unique_values: Optional[np.ndarray] = None
        self.num_unique_values: int = -1
        self._value_freq_by_desc_freq: Optional[List] = None
        self.naive_digitize_map: Optional[np.ndarray] = None
        self.naive_digitized_reduced: Optional[np.ndarray] = None
        self.naive_decoder_key: Any = None
        self.gcode_digitize_map: Optional[np.ndarray] = None
        self.gcode_digitized_reduced: Optional[np.ndarray] = None
        self.gcode_decoder_key: Any = None
        self.sorted_digitize_map: Optional[np.ndarray] = None
        self.sorted_digitized_reduced: Optional[np.ndarray] = None
        self.sorted_decoder_key: Any = None
        self.is_digitize_byte_sparing: Optinal[bool] = None
            
    def run_analysis(self):        
        timings = [timeit.default_timer()]
        self._save_original_image()
        timings.append(timeit.default_timer())
        self.unique_values = np.unique(self.flattened_source)
        self.num_unique_values = len(self.unique_values)
        self.naive_digitize_map = {self.unique_values[ii]: ii for ii in range(0, self.num_unique_values)}
        reduced = [self.naive_digitize_map[p] for p in self.flattened_source]
        if self.num_unique_values < 256:
            # Fewer than 256 unique values are used within this file.
            # We can build an eight-bit digitzation map and order should not matter because no value
            # will get a sixteen-bit assignment.
            LOGGER.info(f"Reducing to 8-bit PNG because this image only uses {self.num_unique_values} distinct pixel values")
            self.is_digitize_byte_sparing = True
            self.naive_digitized_reduced = np.array(reduced, dtype=np.uint8)
            self.naive_decoder_key = {'algorithm': 'digitize8', 'param': self.unique_values.copy()}
        else:
            self.is_digitize_byte_sparing = False
            self.naive_digitized_reduced = np.array(reduced, dtype=np.uint16)
            self.naive_decoder_key = {'algorithm': 'digitize16', 'param': self.unique_values.copy()}
        self._save_compact_image(self.naive_digitized_reduced, "naive_png")
        timings.append(timeit.default_timer())
            
        # In order to get most out of varint encoding, we sort unique domain of pixel values in decreasing
        # order of frequency, so those values that appear most often ones first selected to receive smallest
        # values in the pixel map.  There is a case to be made to use a Hamiltonian cycle that toggles bits
        # in a gray code like order--spreading out entropy could favor better compression.
        (self.sorted_digitize_map, decoder) = self._compute_sorted_pixel_map()
        reduced = [self.sorted_digitize_map[p] for p in self.flattened_source]
        if self.num_unique_values < 256:
            LOGGER.info(f"Reducing to 8-bit PNG because this image uses only {self.num_unique_values} distinct pixel values")
            self.sorted_digitized_reduced = np.array(reduced, dtype=np.uint8)
            self.sorted_decoder_key = {'algorithm': 'digitize8', 'param': decoder}
        else:
            self.sorted_digitized_reduced = np.array(reduced, dtype=np.uint16)
            self.sorted_decoder_key = {'algorithm': 'digitize16', 'param': decoder}
        self._save_compact_image(self.sorted_digitized_reduced, "sorted_png")
        timings.append(timeit.default_timer())
        self._add_timing(timings)
        
        (self.gcode_digitize_map, decoder) = self._compute_gcode_pixel_map()
        reduced = [self.gcode_digitize_map[p] for p in self.flattened_source]
        if self.num_unique_values < 256:
            self.gcode_digitized_reduced = np.array(reduced, dtype=np.uint8)
            self.gcode_decoder_key = {'algorithm': 'digitize8', 'param': decoder}
        else:
            self.gcode_digitized_reduced = np.array(reduced, dtype=np.uint16)
            self.gcode_decoder_key = {'algorithm': 'digitize16', 'param': decoder}
        self._save_compact_image(self.gcode_digitized_reduced, "gcode_png")
        timings.append(timeit.default_timer())
        self._add_timing(timings)
        
        naive_validate = np.fromiter(
            [self.naive_decoder_key["param"][value] for value in self.naive_digitized_reduced], np.uint16
        )
        sorted_validate = np.fromiter(
            [self.sorted_decoder_key["param"][value] for value in self.sorted_digitized_reduced], np.uint16
        )
        gcode_validate = np.fromiter(
            [self.gcode_decoder_key["param"][value] for value in self.gcode_digitized_reduced], np.uint16
        )
        assert((self.flattened_source == naive_validate).all())
        assert((self.flattened_source == sorted_validate).all())
        assert((self.flattened_source == gcode_validate).all())
        
#         if self.num_unique_values >= 256 and self.num_unique_values < 512:
#             LOGGER.info(f"Too many unique values for a single 8-bit PNG reduction.  Not too many for a 2-way split.")
#             (self.split_encoders, self.split_decoders) = self._compute_split_pixel_map()
            
    def to_varint_sparse_analysis(self, is_varint_used: bool, is_sparse_used: bool):
        return SparseVarintAnalysis(
            self._source_file_name,
            self.flattened_source,
            self.width, self.height,
            self.unique_values,
            self.naive_digitized_reduced,
            self.naive_decoder_key, 
            self.gcode_digitized_reduced,
            self.gcode_decoder_key,
            self.sorted_digitized_reduced,
            self.sorted_decoder_key,
            self.is_digitize_byte_sparing,
            self._result_file_template,
            self._read_validate,
            is_varint_used, is_sparse_used
        )
        
    def _sort_value_and_freq_by_desc_freq(self):
        if self._value_freq_by_desc_freq is not None:
            return
        bin_count = len(self.unique_values)
        np.sort(self.unique_values)
        bin_boundaries = self.unique_values.copy()
        bin_boundaries.resize(bin_count + 1)
        bin_boundaries[bin_count] = self.unique_values[bin_count - 1] + 1
        self._pixel_values_histogram = np.histogram(self.flattened_source, bins=bin_boundaries, density=False)
        value_frequency = np.array(
            [(ii[0], 0-ii[1]) for ii in zip(self._pixel_values_histogram[1], self._pixel_values_histogram[0])], 
            dtype=[("value", int), ("freq", int)])
        value_frequency.sort(order="freq")
        self._value_freq_by_desc_freq = value_frequency
            
#     def _compute_split_pixel_map(self) -> Tuple:
#         decoder = self.sorted_decoder_key["param"]
#         decoder_len = len(decoder)
#         if (decoder_len % 2) == 1:
#             decoder_len = decoder_len + 1
#             decoder.resize(decoder_len)
#             decoder[decoder_len - 1] = np.max(decoder) + 2
#         decoders = decoder.reshape([decoder_len/2, 2]).T
#         digitize_maps = []
#         digitize_maps[0] = {decoders[0][ii]: SCATTER_DIGI_MAP[ii] for ii in range(0, decoder_len/2)}
#         # digitize_maps[1] = {decoders[1][ii]: SCATTER_DIGI_MAP[ii] for ii in range(0, decoder_len/2)}
#         digitize_map = {decoder[ii]: (ii ^ (ii >> 1)) for ii in range(0, self.num_unique_values)}
#         return (digitize_maps, decoders)
                            
    def _compute_gcode_pixel_map(self) -> Tuple:
        self._sort_value_and_freq_by_desc_freq()
        digitize_map = {
            self._value_freq_by_desc_freq[ii][0]: SCATTER_DIGI_MAP[ii]
            for ii in range(0, self.num_unique_values)
        }
        max_decoder_size = int(math.pow(2, math.ceil(math.log(((self.num_unique_values + 63) // 64), 2))) * 64)
        decoder = [0] * max_decoder_size
        LOGGER.warning(f"Allocated decoder array of size {max_decoder_size} for {self.num_unique_values} values")
        for ii in digitize_map:
            if (digitize_map[ii] >= len(decoder)):
                print(f"About to crash due to {digitize_map[ii]} into {len(decoder)} for {ii}")
            decoder[digitize_map[ii]] = ii
        return (digitize_map, decoder) 

    def _compute_naive_gcode_pixel_map(self) -> Tuple:
        np.sort(self.unique_values)
        digitize_map = {
            self.unique_values[ii]: SCATTER_DIGI_MAP[ii]
            for ii in range(0, self.num_unique_values)
        }
        max_decoder_size = int(math.pow(2, math.ceil(math.log(((self.num_unique_values + 63) // 64), 2))) * 64)
        decoder = [0] * max_decoder_size
        LOGGER.warning(f"Allocated decoder array of size {max_decoder_size} for {self.num_unique_values} values")
        for ii in digitize_map:
            decoder[digitize_map[ii]] = ii
        return (digitize_map, decoder) 

    def _compute_sorted_pixel_map(self) -> Tuple:
        self._sort_value_and_freq_by_desc_freq()
        decoder = [ii[0] for ii in self._value_freq_by_desc_freq]
        digitize_map = {
            decoder[ii]: ii 
            for ii in range(0, self.num_unique_values)}
        return (digitize_map, decoder) 

    def _save_original_image(self):
        filename = self._result_file_template.format("nomods_png", "png")
        with open(filename, "wb") as foo:
            foo.write(self._source_file_bytes)
        pixel_data: np.ndarray = imread(filename)
        (self.height, self.width) = pixel_data.shape
        self.flattened_source = pixel_data.flatten()
        LOGGER.info(f"Result file <{filename}> was written for nomods_png")
        self._add_result(filename, "nomods_png", data = self.flattened_source);

    def _save_compact_image(self, reduced: np.ndarray, label: str) -> None:
        shaped: np.ndarray = reduced.reshape([self.height, self.width])
        filename = self._result_file_template.format(label, "png")
        LOGGER.info(f"Result file <{filename}> was written for {label}")
        imsave(filename, shaped)
        self._add_result(filename, label, data = reduced);

    def expand_digitize8(self, compressed: np.ndarray, decoder_map: List) -> np.ndarray:
        return decoder_map[compressed]

    def expand_digitize16(self, compressed: np.ndarray, decoder_map: List) -> np.ndarray:
        return decoder_map[compressed]

    EXPANDERS = {
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
