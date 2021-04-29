from typing import Dict, List, Tuple, Optional, Any
from scipy import sparse
import varints
import sys
import numpy as np
import gzip
import os

from base_analysis import BaseAnalysis

WRITE_BLOCK_SIZE = 2097152 # 8192
READ_BLOCK_SIZE = 2097152

class SparseVarintAnalysis(BaseAnalysis):
    def __init__(
        self, base_label,
        flattened_source, offset_reduced, offset_key, unique_values, naive_digitized, naive_decoder, 
        sorted_digitize_map, sorted_digitized, sorted_decoder, is_base_image_viable, is_offset_image_viable,
        is_offset_byte_sparing, is_digitize_byte_sparing, raw_result_file, signed_offset_result_file,
        offset_result_file, naive_digitized_result_file, sorted_digitized_result_file, apply_varints, apply_sparse
    ):
        BaseAnalysis.__init__(self)
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
        self.is_base_image_viable = is_base_image_viable
        self.is_offset_image_viable = is_offset_image_viable
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
            if self.is_base_image_viable:
                flattened_source = sparse.bsr_matrix(flattened_source).data.flatten()
            if self.is_offset_image_viable and not offset_reduced is None:
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
            if self.is_base_image_viable:
                self.execute_trial(flattened_source, self.raw_result_file, test_name=test_name)
            
            if not self.offset_reduced is None and not self.is_offset_byte_sparing and self.is_offset_image_viable:
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
        
#             if self.sorted_digitize_map[0] > np.min(self.unique_values):
#                 print("Most common value was not the smallest value.  Normalizing the smallest value to zero may not have produced as sparse of a file as normalizing to the most common value.  Attempting to normalize to the most common value will store negative values, which may require a specific varint format to tolerate.");
#                 self.varint_strategies = [
#                     { "name": "sqliteu", "impl": varints.sqliteu },
#                     { "name": "leb128s", "impl": varints.dlugoszu }
#                 ]
#                 self.apply_alternate_offset_strategy()
#             else:
#                 print("Alternate offset reduction does not apply.  Skipping it...")
        else:
            if self.is_base_image_viable:
                self.save_sparse_data(flattened_source, self.raw_result_file, "sparse" )
            if self.is_offset_image_viable and not offset_reduced is None:
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
#         as_list = source_ndarray.tolist()
        byte_count = source_ndarray.nbytes
        item_count = len(source_ndarray)
#         padding = (item_count % WRITE_BLOCK_SIZE)
#         print(f"Compare {item_count} items to {sys.getsizeof(as_list)} or {len(as_list)}")
#         if (padding > 0):
            # Add padding to get an even multiple of WRITE_BLOCK_SIZE.  We'll trim this away from the final buffer later.
#             padding = WRITE_BLOCK_SIZE - padding
#             as_list.extend([0] * padding)
#             item_count = item_count + padding
#             source_ndarray = np.array(as_list, np.uint16)
#         block_count: int = round(item_count / WRITE_BLOCK_SIZE)
#         algo_idx = 0
        for varint_algo in self.varint_strategies:
            algo_name = varint_algo["name"]
            varint_algo = varint_algo["impl"]
            algo_result_file = f"{str(result_file)}-{algo_name}.dat"
#             approx = 0
#             block_idx = 0
            bytes_written = 0
#             last_block = block_count - 1
            with gzip.open(algo_result_file, 'wb', compresslevel=9) as woo:
#                 for chunk in np.split(source_ndarray, block_count):
#                     block_idx += 1
#                     if block_idx == last_block:
#                         chunk = chunk[:(-1 * padding)]
                var = varint_algo.encode(source_ndarray) # .tolist())
                bytes_written += woo.write(var)
            print(f"|{bytes_written}|{algo_name}|{result_file}|")
            if read_verify:
                bytes_to_read = byte_count
                rehydrated_png = np.ndarray([bytes_to_read], dtype=np.byte)
                next_read = bytes_to_read if bytes_to_read < READ_BLOCK_SIZE else READ_BLOCK_SIZE
                bytes_read = 0
                with gzip.open(algo_result_file, 'rb', compresslevel=9) as woo:
#                     while next_read > 0:
                    chunk = woo.read() # size=next_read)
                    decoded_bytes = np.fromiter(varint_algo.decode(chunk), np.uint16, count=item_count)
#                         next_bytes_read = bytes_read + decoded_bytes.nbytes
#                         rehydrated_png[bytes_read:next_bytes_read] = decoded_bytes[0:decoded_bytes.nbytes]
#                         bytes_read = next_bytes_read
#                         next_read = bytes_to_read - bytes_read
#                         next_read = next_read if next_read < READ_BLOCK_SIZE else READ_BLOCK_SIZE
                rehydrated_png = np.frombuffer(decoded_bytes.tobytes(), dtype=np.uint16)
                if not (source_ndarray.shape == rehydrated_png.shape):
                    raise ValueError(
                        f"Source and rehydrated arrays have different flat shapes? {source_ndarray.shape} != {rehydrated_png.shape}")
                if not (source_ndarray.dtype == rehydrated_png.dtype): 
                    raise ValueError(
                        f"Source and rehydrated arrays have different data types? {source_ndarray.dtype} != {rehydrated_png.dtype}")
                element_wise_compare = (source_ndarray == rehydrated_png)
                if not element_wise_compare.all():
                    raise ValueError(f"Decoded varint was not value-identitcal with its original source!")
            self._add_result(algo_result_file, f"{test_name}-{algo_name}")