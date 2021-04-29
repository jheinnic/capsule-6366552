#!/usr/bin/python

import io
import sys
from typing import Iterable, Sized

if sys.version_info[0] > 2:
    from typing import Iterable, Iterator, Generator
    
    def is_iterable(x):
        return isinstance(x, Iterable)
    def is_sizable(x):
        return isinstance(x, Sized)
    def pre_decode(buf):
        return io.BytesIO(buf)
    def empty_varint_storage(initial_size):
        return io.BytesIO(
            bytes(initial_size))
#     def varint_storage(b):
#         return bytes((b, ))
    def store_to_num(b):
        return b
    def num_types():
        return (int)
else:
    def store_from_generator(gen):
        return ''.join(gen)
    def is_iterable(x):
        return hasattr(x, '__iter__')
    def is_sizable(x):
        return hasattr(x, '__len__')
    def pre_decode(buf):
#         if HAS_NUMPY:
#             return np.frombuffer(buf, '|S1')
        return buf
    def empty_varint_storage():
        # TODO!
        return ""
#     def varint_storage(b):
#         return chr(b)
    def store_to_num(b):
        return ord(b)
    def num_types():
        return (int,long)
    
def dump( num ):
    print( "Len: {}".format( len(num) ))
    for element in num:
        print( "B: {}".format( store_to_num(element) ))

# def generic_encode( num, funcs ):
#     ret_val = None
#     if( isinstance(num, list)):
#         source_generator = (item for item in num)
#     elif( isinstance( num, num_types() )):
#         source_generator = (item for item in [num])
#     encoding_generator = funcs['encode_iterator']( num ))
#     return store_from_generator(encoding_generator)

def generic_encode( num, funcs ):
    ret_val = None
    if is_iterable(num):
        if is_sizable(num):
            buf = empty_varint_storage(len(num))
        else:
            buf = empty_varint_storage(1)
    elif isinstance(num, num_types()):
        buf = empty_varint_storage(1)
        num = [num]
    return drive_encoder(num, funcs, buf)

def drive_encoder( num, funcs, buf ):
    encoder_iterator = funcs['encoding_generator']()
    count = 0
    next(encoder_iterator)
    for val in num:
        byte_out = encoder_iterator.send(val)
        while not byte_out is None:
            buf.write(bytes((byte_out,)))
            byte_out = next(encoder_iterator)
    return buf.getbuffer().tobytes()
    

# def generic_decode( num, funcs, max_val_len = None ):
#     ret_val = None
#     decode_val = funcs['decode_val']
#     if( isinstance(num, (str,bytes))):
#         len_num = len(num)
#         if len_num > 0:
#             if max_val_len is None:
#                 max_val_len = len_num
#                 (ret_val, ptr) = decode_val(num)
#             else:
#                 (ret_val, ptr) = decode_val(num[0:max_val_len])
#             if ptr < len_num:
#                 ret_val = [ret_val]
#             while ptr < len_num:
#                 (int_val, bytes_used) = decode_val( num[ptr:ptr+max_val_len] )
#                 ptr = ptr + bytes_used
#                 ret_val.append( int_val )
#     return ret_val

# def feed_decoder( num ):
#     if( isinstance(num, (str,bytes))):
#         for value in num:
#             yield value
    
# def generic_decode( num, funcs, max_val_len = None ):
#     return [x for x in _generic_decode_generator(num, funcs, max_val_len=max_val_len])]
    
def generic_decode( num, funcs, max_val_len = None):
    ret_val = None
    decoding_generator = funcs['decoding_generator']
    if( isinstance(num, bytes)):
        num = io.BytesIO(num).getbuffer()
        if hasattr(num, 'toreadonly'):
            num = num.toreadonly()
        len_num = len(num)
        ptr = 0
        if len_num > 0:
            if max_val_len is None:
                max_val_len = len_num
            worker = decoding_generator()
            next(worker)
        while ptr < len_num:
            (int_val, bytes_used) = worker.send( num[ptr:ptr+max_val_len] )
            ptr = ptr + bytes_used
            yield int_val
