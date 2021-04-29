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
        self.is_signed = is_signed
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
            
        self.is_base_image_viable: bool = True
        self.is_offset_image_viable: bool = True
        self.is_offset_byte_sparing: bool = False
        self.is_digitize_byte_sparing = False
        self.is_signed_offset_more_sparse: bool = False
            
        self.signed_offset_reduced: Optional[np.ndarray]
        self.signed_offset_key: Any = None
            
    def to_delta_analysis(self) -> "Optional[DigitizeImageAnalysis]":
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
        return DigitizeImageAnalysis(
            self.source_file_name,
            self.width, self.height,
            "delta", result, True, 
            self.raw_result_file + "-delta",
            self.offset_result_file + "-delta",
            self.naive_digitized_result_file + "-delta",
            self.sorted_digitized_result_file + "-delta"
        )
        
    def run_analysis(self):
        print(self.flattened_source.dtype)
        print(self.flattened_source.shape)
        self.spread = np.ptp(self.flattened_source)
        min = np.min(self.flattened_source)
        if self.spread < 256 and min >= 0:
            # Conversion from 16-bit ints to 8-bit ints has potential to
            # be lossy, but we have just confirmed the peak to peak value
            # spread is <256, so just normalizing to 0 yields a result that
            # fits nicely into unsigned 8-bit encoding.
            print(f"Reducing to 8-bit PNG because this image's dynamic range is {self.spread} from hi={np.max(self.flattened_source)} to lo={min}.  Unmodified image may still be viable, but hardly likely to perform better.")
            self.is_offset_byte_sparing = True
            self.is_offset_image_viable = True
            self.is_base_image_usable = True
            reduced = self.flattened_source - min
            self.offset_reduced = reduced.astype(np.uint8, 'C', 'unsafe')
            self.offset_key = {'algorithm': 'offset', 'param': min}
            self._save_compact_image(self.offset_reduced, self.offset_result_file, f"{self.base_label}-offset")
            self._save_compact_image(self.flattened_source, self.raw_result_file, f"{self.base_label}-baseline")
        elif not self.is_signed and min > 0:
            print(f"Linearly decreasing 16-bit PNG with no zeros by {-1 * min} to normalize smallest pixel value to 0.  Was min={min}, max={np.max(self.flattened_source)}, spread={self.spread}.  Unmodified image may still be viable." )
            self.is_offset_byte_sparing = False
            self.is_offset_image_viable = True
            self.is_base_image_usable = True
            reduced = self.flattened_source - min
            self.offset_reduced = reduced.astype(np.uint16, 'K', 'unsafe')
            self.offset_key = {'algorithm': 'offset', 'param': min}
            self._save_compact_image(self.offset_reduced, self.offset_result_file, f"{self.base_label}-offset")
            self._save_compact_image(self.flattened_source, self.raw_result_file, f"{self.base_label}-baseline")
        elif self.is_signed and min < 0:
            print(f"Linearly increasing signed 16-bit PNG by {-1 * min} to ensure matrix is non-negative.  Was min={min}, max={np.max(self.flattened_source)}, spread={self.spread}.  Unaltered base image is NOT usable.")
            self.is_offset_byte_sparing = False
            self.is_offset_image_viable = True
            self.is_base_image_usable = False
            reduced = self.flattened_source - min
            self.offset_reduced = reduced.astype(np.uint16, 'K', 'unsafe')
            self.offset_key = {'algorithm': 'offset', 'param': min}
            self._save_compact_image(self.offset_reduced, self.offset_result_file, f"{self.base_label}-offset")
        else:
            # Ignore the offset method entirely if its just going to be identical to the
            # unprocessed baseline and not save on pixel depth.
            print(f"No relevant linear normalization applicable to PNG with spread={self.spread}, min==0, max=={np.max(self.flattened_source)}.  Retaining only the base image.")
            self.is_offset_byte_sparing = False
            self.is_offset_image_viable = False
            self.is_base_image_usable = True
            self.offset_reduced = None
            self.offset_key = None
            self._save_compact_image(self.flattened_source, self.raw_result_file, f"{self.base_label}-baseline")
        
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

#     def _alt_compute_pixel_map(self, flattened_source) -> Tuple:
#         unique_values = np.unique(flattened_source)
#         np.sort(unique_values)
#         bin_boundaries = unique_values.tolist().copy()
#         bin_boundaries.append(np.max(unique_values) + 1)
#         histo = np.histogram(flattened_source, bins=bin_boundaries, density=False)
#         value_frequency = np.array(
#             [(ii[0], 0-ii[1]) for ii in zip(histo[1], histo[0])], 
#             dtype=[("value", int), ("freq", int)])
#         value_frequency.sort(order="freq")
#         decoder = [ii[0] for ii in value_frequency]
#         digitize_map = {decoder[ii]: ii for ii in range(0, len(unique_values))}
#         return (digitize_map, decoder, value_frequency) 
    
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
                suffix = "-sparse-varint"
            else:
                suffix = "-varint"
        elif is_sparse_used:
            suffix = "-sparse"
        else:
            raise ValueException("Either varint or sparse mode or both is requires")
            
        return SparseVarintAnalysis(
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
            self.is_base_image_viable,
            self.is_offset_image_viable,
            self.is_offset_byte_sparing,
            self.is_digitize_byte_sparing,
            self.raw_result_file + suffix,
            self.offset_result_file + "-signed" + suffix,
            self.offset_result_file + suffix,
            self.naive_digitized_result_file + suffix,
            self.sorted_digitized_result_file + suffix,
            is_varint_used, is_sparse_used
        )
    
