from ..varints import num_types,generic_encode,generic_decode
from ..varints import store_to_num


def measure(num):
    """
    Instead of encoding a message, generate a sequence of single integers, encoding the number of bytes that would have
    been written in an actual encoding.  This makes it easy to scan the resulting byte sequence and report the encoder's
    actual measured efficiency.
    """
    return generic_encode( num, measure_funcs )

def measurement_generator( num, functs ):
    num = yield None
    while not num is None:
        if num < 0:
            raise ValueError("Negative numbers not handled")
        elif num > maxint:
            raise ValueError(f"{num} exceeds maximum encodable value, {maxint}")
        elif( num <= ONE_BYTE_LIMIT ):
            yield 1
        elif( num <= TWO_BYTE_LIMIT ):
            yield 2
        elif( num <= THREE_BYTE_LIMIT ):
            yield 3
        else:
            start = 0
            while(( start < len( buckets )) and
                  ( num > buckets[start]['limit'])):
                start = start + 1
            if( start == len( buckets )):
                raise ValueError("Too large")
            yield start + 4
        num = yield None
                
                
def report( num ):
    return generic_decode( num, funcs, max_val_len=9 )


def reporting_generator( ):
    num = yield None
    while(not num is None):
        if( first <= ONE_BYTE_LIMIT ):
            bytes_used = 1
        elif( first < THREE_BYTE_HEADER ):
            bytes_used = 2
        elif( first == THREE_BYTE_HEADER ):
            bytes_used = 3
        else:
            bytes_used = first - 246
        num = yield (bytes_used, bytes_used)
    yield None

funcs = {
    'encoding_generator': measuring_generator,
    'decoding_generator': reporting_generator
}