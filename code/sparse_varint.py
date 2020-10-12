from typing import Dict, List, Tuple, Optional, Any
from scipy import sparse
import varints
import sys
import numpy as np
import bz2
import os

from base_analysis import BaseAnalysis
from sample_image_pb2 import ImageSampleContainer, SparseBsrMatrix, FlatDenseMatrix, XmpHeaderContent


class SparseVarintAnalysis(BaseAnalysis):
    def __init__(
        self, base_label, shaped_original_source, width, height, unique_values, naive_digitized, naive_decoder, 
        sorted_digitize_map, sorted_digitized, sorted_decoder, is_digitize_byte_sparing, baseline_source_file,
        naive_digitized_result_file, sorted_digitized_result_file, apply_varints, apply_sparse
    ):
        BaseAnalysis.__init__(self)
        if (not apply_varints and not apply_sparse):
            raise ValueException("Must apply varints, sparse matrices, or both!")
        self.shaped_original_source = shaped_original_source.reshape([height, width])
        self.width = width
        self.height = height
        self.unique_values = unique_values
        self.naive_digitized = naive_digitized.reshape([height, width])
        self.naive_decoder = naive_decoder
        self.sorted_digitize_map = sorted_digitize_map
        self.sorted_digitized = sorted_digitized.reshape([height, width])
        self.sorted_decoder = sorted_decoder
        self.is_digitize_byte_sparing = is_digitize_byte_sparing
        self.baseline_source_file = baseline_source_file
        self.naive_digitized_result_file = naive_digitized_result_file
        self.sorted_digitized_result_file = sorted_digitized_result_file
        self.apply_varints = apply_varints
        self.apply_sparse = apply_sparse
        self.varint_strategies = [
            { "name": "sqliteu", "impl": varints.sqliteu },
        ]
        
        
    def run_analysis(self):
        shaped_original_source = self.shaped_original_source
        sorted_digitized = self.sorted_digitized
        naive_digitized = self.naive_digitized
        
        if self.apply_sparse:
            sparse_original_source = sparse.bsr_matrix(shaped_original_source)
            sparse_naive_digitized = sparse.bsr_matrix(naive_digitized)
            sparse_sorted_digitized = sparse.bsr_matrix(sorted_digitized)
            
            if self.apply_varints:
                test_name = "sparse-varint"
                with_varints = True
            else:
                test_name = "sparse"
                with_varints = False
                
            self.save_sparse_data(
                sparse_original_source, with_varints, self.baseline_source_file, test_name )
            self.save_sparse_data(
                sparse_sorted_digitized, with_varints, self.sorted_digitized_result_file, test_name )
            self.save_sparse_data(
                sparse_naive_digitized, with_varints, self.naive_digitized_result_file, test_name )
        elif self.apply_varints:
            test_name = "varint"
            self.execute_trial(
                shaped_original_source.flatten(), self.baseline_source_file, test_name )
            self.execute_trial(
                sorted_digitized.flatten(), self.sorted_digitized_result_file, test_name )
            self.execute_trial(
                naive_digitized.flatten(), self.naive_digitized_result_file, test_name )
        
    def save_sparse_data(self, sparse_matrix: sparse.bsr, with_varints: bool, base_filename: str, label_prefix: str):
        """
        Index pointers are small in number, but require 32-bit addressing because they must address each unique location
        with non-zero data across the entire matrix.  Indices count the number of blocks in a row, so until we are 
        storing images with at least 2^16 pixels in a single dimension, 2^16th bits of addressability will suffices,
        and 2^8 is too small since most images have more than 256 pixels in each dimension.  Data values store the 
        actual pixel content at each non-sparse location, and since the original data set is encoded using 16-bit
        grayscale, 2^16 is also suitable here.
        
        The sparse format alone is capable of yielding ~25% reduction in storage requirements, but varint encoding
        yields will be able to cut the storeage cost for a majority of the data bytes in half, rouhgly half of the
        index bytes, and all of the index pointer values will reduce from 4 bytes to 1, 2, or 3 bytes.
        
        The fourth array of six values captures all dimensions needed to rehydrate this content to the original
        PNG image matrix.
        """
        pixel_width = self.width
        pixel_height = self.height
        nvals = sparse_matrix.indices.shape[0]
        block_row_count = sparse_matrix.indptr.shape[0]
        block_width = sparse_matrix.data.shape[2]
        block_height = sparse_matrix.data.shape[1]
        
        indptr_array = sparse_matrix.indptr.flatten()
        indices_array = sparse_matrix.indices.flatten()
        data_matrix = sparse_matrix.data.flatten()
        shape_sizes = np.array(
            [nvals, block_width, block_height, block_row_count, pixel_width, pixel_height], np.uint32)
        
        indptr_filename = base_filename + label_prefix + "-indptr.dat"
        indices_filename = base_filename + label_prefix + "-indices.dat"
        data_matrix_filename = base_filename + label_prefix + "-data_matrix.dat"
        shape_sizes_filename = base_filename + label_prefix + "-shape_sizes.dat"
            
        if with_varints:
            indptr_array = varints.sqliteu.encode(indptr_array)
            indices_array = varints.sqliteu.encode(indices_array)
            data_matrix = varints.sqliteu.encode(data_matrix)
            shape_sizes = varints.sqliteu.encode(shape_sizes)
            
#             with bz2.open(indptr_filename, 'wb', compresslevel=9) as woo:
#                 woo.write(indptr_array)
#             with bz2.open(indices_filename, 'wb', compresslevel=9) as woo:
#                 woo.write(indices_array)
#             with bz2.open(data_matrix_filename, 'wb', compresslevel=9) as woo:
#                 woo.write(data_matrix)
#             with bz2.open(shape_sizes_filename, 'wb', compresslevel=9) as woo:
#                 woo.write(shape_sizes)
            smdata = SparseBsrMatrix()
            smdata.blockHeight = block_height
            smdata.blockWidth = block_width
            smdata.indPtrLen = len(indptr_array)
            smdata.indPtr = indptr_array.
        else:
            with bz2.open(indptr_filename, 'wb', compresslevel=9) as woo:
                woo.write(indptr_array.tobytes())
            with bz2.open(indices_filename, 'wb', compresslevel=9) as woo:
                woo.write(indices_array.tobytes())
            with bz2.open(data_matrix_filename, 'wb', compresslevel=9) as woo:
                woo.write(data_matrix.tobytes())
            with bz2.open(shape_sizes_filename, 'wb', compresslevel=9) as woo:
                woo.write(shape_sizes.tobytes())
        self._add_multi_file_result(
            [indptr_filename, indices_filename, data_matrix_filename, shape_sizes_filename], label_prefix)
        
    
    def execute_trial(
        self, source_ndarray: np.ndarray, result_file: str, test_name:str, dtype = np.uint16, read_verify:bool = True):
        """
        Apply each candidate varint algoritm to the given condensed byte sequence.  Report on the sizing
        of each any ensure that each algorithm's work is reversible without any knowledge other than
        which algorithm was used to condense the bit sequence.
        """
        byte_count = source_ndarray.nbytes
        item_count = len(source_ndarray)
        for varint_algo in self.varint_strategies:
            algo_name = varint_algo["name"]
            varint_algo = varint_algo["impl"]
            algo_result_file = f"{str(result_file)}-{algo_name}.dat"
            bytes_written = 0
            with bz2.open(algo_result_file, 'wb', compresslevel=9) as woo:
                var = varint_algo.encode(source_ndarray) # .tolist())
                bytes_written += woo.write(var)
            print(f"|{bytes_written}|{algo_name}|{result_file}|")
            if read_verify:
                bytes_to_read = byte_count
                with bz2.open(algo_result_file, 'rb', compresslevel=9) as woo:
                    chunk = woo.read() 
                rehydrated_png = np.fromiter(varint_algo.decode(chunk), np.uint16, count=item_count)
#                 rehydrated_png = np.frombuffer(decoded_bytes.tobytes(), dtype=np.uint16)
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