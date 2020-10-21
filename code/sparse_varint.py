from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import asyncio
from scipy import sparse
from uuid import uuid1
import numpy as np
import timeit
import sys
import bz2
import os

from output_logger import debug_capture, getLogger
from base_analysis import BaseAnalysis, VarintMode
from sample_image_pb2 import ImageSampleContainer
import varints


LOGGER = getLogger(__name__)


class SparseVarintAnalysis(BaseAnalysis):
    def __init__(
        self, source_file_name, original_source, width, height, unique_values_map,
        naive_digitized, naive_decoder, gcode_digitized, gcode_decoder,  sorted_digitized, sorted_decoder, 
        is_digitize_byte_sparing, result_file_template,
        read_validate: bool, apply_varints: VarintMode, apply_sparse: bool 
    ):
        BaseAnalysis.__init__(self, source_file_name)
        self.original_source = original_source.reshape([height, width])
        self.width = width
        self.height = height
        self.unique_values = unique_values_map
        self.naive_digitized = naive_digitized.reshape([height, width])
        self.naive_decoder = None if naive_decoder['param'] is None else np.array(naive_decoder['param'])
        self.gcode_digitized = gcode_digitized.reshape([height, width])
        self.gcode_decoder = None if gcode_decoder['param'] is None else np.array(gcode_decoder['param'])
        self.sorted_digitized = sorted_digitized.reshape([height, width])
        self.sorted_decoder = None if sorted_decoder['param'] is None else np.array(sorted_decoder['param'])
        self.is_digitize_byte_sparing = is_digitize_byte_sparing
        self._result_file_template = result_file_template
        self.read_validate = read_validate
        self.apply_varints = apply_varints
        self.apply_sparse = apply_sparse
        self.varint_algo = varints.sqliteu
        self.varint_name = "sqliteu"
        
    def run_analysis(self):
        original_source = self.original_source
        sorted_digitized = self.sorted_digitized
        gcode_digitized = self.gcode_digitized
        naive_digitized = self.naive_digitized
        dense_digi_dtype = np.uint8 if self.is_digitize_byte_sparing else np.uint16
        
        timing = [0, 0, 0, 0, 0];
        if self.apply_sparse:
            sparse_original_source = sparse.bsr_matrix(original_source, blocksize=(1, 1))
            sparse_naive_digitized = sparse.bsr_matrix(naive_digitized, blocksize=(1, 1))
            sparse_gcode_digitized = sparse.bsr_matrix(gcode_digitized, blocksize=(1, 1))
            sparse_sorted_digitized = sparse.bsr_matrix(sorted_digitized, blocksize=(1, 1))
            
            if self.apply_varints == VarintMode.USE_PY_VARINT:
                test_name = "sparse_varint";
            elif self.apply_varints == VarintMode.USE_PROTO_INT:
                test_name = "sparse_protoint";
            elif self.apply_varints == VarintMode.USE_BYTES:
                test_name = "sparse_binary";
            else:
                raise ValueException(f'{self.apply_varints} is not a valid value for the VarintMode enum')
                
            timing[0] = timeit.default_timer();
            self.save_sparse_data(
                sparse_original_source, self.apply_varints, None, "nomods", test_name )
            timing[1] = timeit.default_timer();
            if self.apply_varints == VarintMode.USE_PY_VARINT and self.is_digitize_byte_sparing:
                timing = [timing[0], timing[1]]
            else:
                self.save_sparse_data(
                    sparse_naive_digitized, self.apply_varints, self.naive_decoder, "naive", test_name )
                timing[2] = timeit.default_timer();
                self.save_sparse_data(
                    sparse_sorted_digitized, self.apply_varints, self.sorted_decoder, "sorted", test_name )
                timing[3] = timeit.default_timer();
                self.save_sparse_data(
                    sparse_gcode_digitized, self.apply_varints, self.naive_decoder, "gcode", test_name )
                timing[4] = timeit.default_timer();
        elif self.apply_varints == VarintMode.USE_PY_VARINT:
            timing[0] = timeit.default_timer();
            self.run_dense_varint_trial(
                original_source.flatten(), None, "nomods")
            timing[1] = timeit.default_timer();
            if self.is_digitize_byte_sparing:
                timing = [timing[0], timing[1]]
            else:
                self.run_dense_varint_trial(
                    naive_digitized.flatten(), self.naive_decoder, "naive", dtype=dense_digi_dtype)
                timing[2] = timeit.default_timer();
                self.run_dense_varint_trial(
                    sorted_digitized.flatten(), self.sorted_decoder, "sorted", dtype=dense_digi_dtype)
                timing[3] = timeit.default_timer();
                self.run_dense_varint_trial(
                    gcode_digitized.flatten(), self.gcode_decoder, "gcode", dtype=dense_digi_dtype)
                timing[4] = timeit.default_timer();
        elif self.apply_varints == VarintMode.USE_PROTO_INT:
            timing[0] = timeit.default_timer();
            self.run_dense_proto_trial(
                original_source.flatten(), None, "nomods")
            timing[1] = timeit.default_timer();
            self.run_dense_proto_trial(
                naive_digitized.flatten(), self.naive_decoder, "naive", dtype=dense_digi_dtype)
            timing[2] = timeit.default_timer();
            self.run_dense_proto_trial(
                sorted_digitized.flatten(), self.sorted_decoder, "sorted", dtype=dense_digi_dtype)
            timing[3] = timeit.default_timer();
            self.run_dense_proto_trial(
                gcode_digitized.flatten(), self.naive_decoder, "gcode", dtype=dense_digi_dtype)
            timing[4] = timeit.default_timer();
        elif self.apply_varints == VarintMode.USE_BYTES:
            timing[0] = timeit.default_timer();
            self.run_dense_binary_trial(
                original_source.flatten(), None, "nomods")
            timing[1] = timeit.default_timer();
            self.run_dense_binary_trial(
                naive_digitized.flatten(), self.naive_decoder, "naive", dtype=dense_digi_dtype)
            timing[2] = timeit.default_timer();
            self.run_dense_binary_trial(
                sorted_digitized.flatten(), self.sorted_decoder, "sorted", dtype=dense_digi_dtype)
            timing[3] = timeit.default_timer();
            self.run_dense_binary_trial(
                gcode_digitized.flatten(), self.naive_decoder, "gcode", dtype=dense_digi_dtype)
            timing[4] = timeit.default_timer();
        else:
            raise ValueException(f"Could not recognize varint mode argument, {self.apply_varints}")
        self._add_timing(timing)

            
    def save_sparse_data(
        self, sparse_matrix: sparse.bsr, with_varints: VarintMode,
        decode_matrix: np.ndarray, source_label: str, variant_label: str):
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
        indices_array = sparse_matrix.indices.flatten().astype(np.uint16)
        data_matrix = sparse_matrix.data.flatten()
        
        scenario_label = f"{source_label}_{variant_label}"
        task_log_filename = self._result_file_template.format(scenario_label, "dat")
        
        container: ImageSampleContainer = ImageSampleContainer()
        container.header.uuid = str(uuid1())
        container.header.dataType = 'png'
        container.header.captureTime = datetime.now().timestamp()
        container.header.exposureTime = 25.2
        container.pixelHeight = pixel_height
        container.pixelWidth = pixel_width
        if not decode_matrix is None:
            container.digitizedBy = decode_matrix.tobytes()
            
        if with_varints == VarintMode.USE_PY_VARINT:
            indptr_array = varints.sqliteu.encode(indptr_array)
            indices_array = varints.sqliteu.encode(indices_array)
            data_matrix = varints.sqliteu.encode(data_matrix)
        elif with_varints == VarintMode.USE_BYTES:
            indptr_array = indptr_array.tobytes()
            indices_array = indices_array.tobytes()
            data_matrix = data_matrix.tobytes()
            
        if with_varints == VarintMode.USE_BYTES or with_varints == VarintMode.USE_PY_VARINT:
            container.sparseVarintBsr.blockHeight = block_height
            container.sparseVarintBsr.blockWidth = block_width
            container.sparseVarintBsr.indPtrLen = len(indptr_array)
            container.sparseVarintBsr.indicesLen = len(indices_array)
            container.sparseVarintBsr.dataLen = len(data_matrix)
            container.sparseVarintBsr.indPtr = indptr_array
            container.sparseVarintBsr.indices = indices_array
            container.sparseVarintBsr.data = data_matrix
        else:
            container.sparseBsr.blockHeight = block_height
            container.sparseBsr.blockWidth = block_width
            container.sparseBsr.indPtr.extend(indptr_array)
            container.sparseBsr.indices.extend(indices_array)
            container.sparseBsr.data.extend(data_matrix)
            
        with bz2.open(task_log_filename, 'wb', compresslevel=9) as woo:
            woo.write(container.SerializeToString())
        self._add_result(task_log_filename, scenario_label)
        
    
    def run_dense_proto_trial(
        self, source_ndarray: np.ndarray, decode_matrix: np.ndarray,
        source_label: str, dtype = np.uint16, read_verify:bool = True):
        """
        Apply each candidate varint algoritm to the given condensed byte sequence.  Report on the sizing
        of each any ensure that each algorithm's work is reversible without any knowledge other than
        which algorithm was used to condense the bit sequence.
        """
        file_suffix = f"{source_label}_dense_protoint"
        algo_result_file = self._result_file_template.format(file_suffix, "dat")
        
        container: ImageSampleContainer = ImageSampleContainer()
        container.header.uuid = str(uuid1())
        container.header.dataType = 'png'
        container.header.captureTime = datetime.now().timestamp()
        container.header.exposureTime = 25.2
        container.pixelHeight = self.height
        container.pixelWidth = self.width
        container.densePng.data.extend(source_ndarray)
        if not decode_matrix is None:
            container.digitizedBy = decode_matrix.tobytes()
        with bz2.open(algo_result_file, 'wb', compresslevel=9) as woo:
            woo.write(container.SerializeToString())
        if read_verify:
            with bz2.open(algo_result_file, 'rb', compresslevel=9) as woo:
                chunk = woo.read() 
            container.ParseFromString(chunk)
            rehydrated_png = np.array(container.densePng.data, dtype=dtype)
            self.assert_comparison(source_ndarray, rehydrated_png)
        self._add_result(algo_result_file, file_suffix)
    
    
    def run_dense_varint_trial(
        self, source_ndarray: np.ndarray, decode_matrix: np.ndarray,
        source_label: str, dtype = np.uint16, read_verify:bool = True):
        """
        Apply each candidate varint algoritm to the given condensed byte sequence.  Report on the sizing
        of each any ensure that each algorithm's work is reversible without any knowledge other than
        which algorithm was used to condense the bit sequence.
        """
        file_suffix = f"{source_label}_dense_varint"
        algo_result_file = self._result_file_template.format(file_suffix, "dat")
        
        item_count = len(source_ndarray)
        data_matrix = self.varint_algo.encode(source_ndarray)
        
        container: ImageSampleContainer = ImageSampleContainer()
        container.header.uuid = str(uuid1())
        container.header.dataType = 'png'
        container.header.captureTime = datetime.now().timestamp()
        container.header.exposureTime = 25.2
        container.pixelHeight = self.height
        container.pixelWidth = self.width
        container.denseVarintPng.dataLen = item_count
        container.denseVarintPng.data = data_matrix
        if not decode_matrix is None:
            container.digitizedBy = decode_matrix.tobytes()
        with bz2.open(algo_result_file, 'wb', compresslevel=9) as woo:
            woo.write(container.SerializeToString())
        if read_verify:
            with bz2.open(algo_result_file, 'rb', compresslevel=9) as woo:
                chunk = woo.read() 
            container.ParseFromString(chunk)
            rehydrated_png = np.fromiter(
                self.varint_algo.decode(container.denseVarintPng.data),
                dtype=dtype, count=container.denseVarintPng.dataLen)
            self.assert_comparison(source_ndarray, rehydrated_png)
        self._add_result(algo_result_file, file_suffix)
    
    
    def run_dense_binary_trial(
        self, source_ndarray: np.ndarray, decode_matrix: np.ndarray,
        source_label: str, dtype = np.uint16, read_verify:bool = True):
        """
        Apply each candidate varint algoritm to the given condensed byte sequence.  Report on the sizing
        of each any ensure that each algorithm's work is reversible without any knowledge other than
        which algorithm was used to condense the bit sequence.
        """
        file_suffix = f"{source_label}_dense_binary"
        algo_result_file = self._result_file_template.format(file_suffix, "dat")

        item_count = len(source_ndarray)
        data_matrix = source_ndarray.tobytes()
         
        container: ImageSampleContainer = ImageSampleContainer()
        container.header.uuid = str(uuid1())
        container.header.dataType = 'png'
        container.header.captureTime = datetime.now().timestamp()
        container.header.exposureTime = 25.2
        container.pixelHeight = self.height
        container.pixelWidth = self.width
        container.denseVarintPng.dataLen = item_count
        container.denseVarintPng.data = data_matrix
        if not decode_matrix is None:
            container.digitizedBy = decode_matrix.tobytes()
        with bz2.open(algo_result_file, 'wb', compresslevel=9) as woo:
            woo.write(container.SerializeToString())
        if read_verify:
            with bz2.open(algo_result_file, 'rb', compresslevel=9) as woo:
                chunk = woo.read() 
            container.ParseFromString(chunk)
            rehydrated_png = np.frombuffer(container.denseVarintPng.data, dtype=dtype)
            self.assert_comparison(source_ndarray, rehydrated_png)
        self._add_result(algo_result_file, file_suffix)
    
    
    def assert_comparison(self, source_ndarray, rehydrated_png):
        if not (source_ndarray.shape == rehydrated_png.shape):
             raise ValueError(
                 f"Source and rehydrated arrays have different flat shapes? {source_ndarray.shape} != {rehydrated_png.shape}")
        if not (source_ndarray.dtype == rehydrated_png.dtype): 
             raise ValueError(
                 f"Source and rehydrated arrays have different data types? {source_ndarray.dtype} != {rehydrated_png.dtype}")
        element_wise_compare = (source_ndarray == rehydrated_png)
        if not element_wise_compare.all():
             raise ValueError(f"Decoded varint was not value-identitcal with its original source!")
 