#!/usr/bin/python

#   Copyright 2017 John Bailey
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# Based on the encoding method described at
#  https://sqlite.org/src4/doc/trunk/www/varint.wiki

from io import BytesIO
from typing import Iterable
from struct import pack, unpack
try:
    from numpy import integer, uint8
    NUMPY_FOUND = True
except Exception:
    NUMPY_FOUND = False

ONE_BYTE_LIMIT = 240
TWO_BYTE_LIMIT = 2287
THREE_BYTE_LIMIT = 67823

FOUR_BYTE_LIMIT = 16777215
FIVE_BYTE_LIMIT = 4294967295
SIX_BYTE_LIMIT = 1099511627775
SEVEN_BYTE_LIMIT = 281474976710655
EIGHT_BYTE_LIMIT = 72057594037927935
NINE_BYTE_LIMIT = 18446744073709551615
THREE_BYTE_HEADER = 249
FOUR_BYTE_HEADER = 250
FIVE_BYTE_HEADER = 251
SIX_BYTE_HEADER = 252
SEVEN_BYTE_HEADER = 253
EIGHT_BYTE_HEADER = 254
NINE_BYTE_HEADER = 255
BYTE_VALS = 256
SHORT_VALS = 65536
# FOUR_BYTE_MOD = (FOUR_BYTE_LIMIT + 1) // BYTE_VALS
# FIVE_BYTE_MOD = (FIVE_BYTE_LIMIT + 1) // BYTE_VALS
# SIX_BYTE_MOD = (SIX_BYTE_LIMIT + 1) // BYTE_VALS
# SEVEN_BYTE_MOD = (SEVEN_BYTE_LIMIT + 1) // BYTE_VALS
# EIGHT_BYTE_MOD = (EIGHT_BYTE_LIMIT + 1) // BYTE_VALS
# NINE_BYTE_MOD = (NINE_BYTE_LIMIT + 1) // BYTE_VALS
LARGE_BYTE_MODS = [72057594037927936, 281474976710656, 1099511627776, 4294967296, 16777216, 65536, 256, 1]

BUCKET_OFFSET = 2

minint = 0
maxint = NINE_BYTE_LIMIT

buckets = [ { 'limit': FOUR_BYTE_LIMIT,
              'header': FOUR_BYTE_HEADER },
            { 'limit': FIVE_BYTE_LIMIT,
              'header': FIVE_BYTE_HEADER },
            { 'limit': SIX_BYTE_LIMIT,
              'header': SIX_BYTE_HEADER },
            { 'limit': SEVEN_BYTE_LIMIT,
              'header': SEVEN_BYTE_HEADER },
            { 'limit': EIGHT_BYTE_LIMIT,
              'header': EIGHT_BYTE_HEADER },
            { 'limit': NINE_BYTE_LIMIT,
              'header': NINE_BYTE_HEADER },
          ]

def encode( nums: Iterable[int] ):
    for num in nums:
        if num < 0:
            raise ValueError("Negative numbers not handled")
        if num > maxint:
            raise ValueError(f"{num} exceeds maximum encodable value, {maxint}")

        if( num <= ONE_BYTE_LIMIT ):
            yield num
        elif( num <= TWO_BYTE_LIMIT ):
            top = num-ONE_BYTE_LIMIT
            yield (top // BYTE_VALS)+ONE_BYTE_LIMIT+1
            yield top % BYTE_VALS
        elif( num <= THREE_BYTE_LIMIT ):
            top = num-(TWO_BYTE_LIMIT+1)
            yield THREE_BYTE_HEADER
            yield top // BYTE_VALS
            yield top % BYTE_VALS
        else:
            start = 0

            # Work out how many bytes are needed to store this value
            while(( start < len( buckets )) and
                  ( num > buckets[start]['limit'])):
                start = start + 1

            if( start == len( buckets )):
                raise ValueError("Too large")

            yield buckets[start]['header']
            mod = (buckets[start]['limit']+1) // BYTE_VALS
            start = start + BUCKET_OFFSET

            while( start >= 0 ):
                start = start - 1
                yield num // mod
                num = num % mod
                mod = mod // BYTE_VALS
                
                
def encode2( nums: Iterable[int] ) -> BytesIO:
    buf = BytesIO(bytes(len(nums)))
    if NUMPY_FOUND:
        def write(byte_num):
            if isinstance(byte_num, integer):
                buf.write(
                    byte_num.astype(uint8).tobytes()
                )
            else:
                buf.write(
                    byte_num.tobyte(1, 'little'))
    else:
        def write(byte_num):
            buf.write(
                byte_num.tobyte(1, 'little'))
        
    for num in nums:
        if num < 0:
            raise ValueError("Negative numbers not handled")
        if num > maxint:
            raise ValueError(f"{num} exceeds maximum encodable value, {maxint}")

        if( num <= ONE_BYTE_LIMIT ):
            write(num)
        elif( num <= TWO_BYTE_LIMIT ):
            top = num-ONE_BYTE_LIMIT
            buf.write(
                pack(
                    'BB', 
                    (top // BYTE_VALS)+ONE_BYTE_LIMIT+1, 
                    top % BYTE_VALS))
        elif( num <= THREE_BYTE_LIMIT ):
            top = num-(TWO_BYTE_LIMIT+1)
            buf.write(
                pack(
                    'BBB',
                    THREE_BYTE_HEADER,
                    top // BYTE_VALS,
                    top % BYTE_VALS))
        else:
            # Work out how many bytes are needed to store this value
            if num <= SIX_BYTE_LIMIT:
                if num <= FOUR_BYTE_LIMIT:
                    header = FOUR_BYTE_HEADER
                    start = 5
                elif num <= FIVE_BYTE_LIMIT:
                    header = FIVE_BYTE_HEADER
                    start = 4
                else:
                    header = SIX_BYTE_HEADER
                    start = 3
            elif num <= EIGHT_BYTE_LIMIT:
                if num <= SEVEN_BYTE_LIMIT:
                    header = SEVEN_BYTE_HEADER
                    start = 2
                else:
                    header = EIGHT_BYTE_HEADER
                    start = 1
            elif num <= NINE_BYTE_LIMIT:
                header = NINE_BYTE_HEADER
                start = 0
            else:
                raise ValueError("Too large")

            write(header)
            for ii in range(start, 8):
                mod = LARGE_BYTE_MODS[ii]
                write(num // mod)
                num = num % mod
    return buf

def decode( bytes_in: bytes ):
    idx = 0
    end = len(bytes_in)
    while idx < end:
        ret_val = None
        bytes_used = 1
        first = store_to_num( num[idx] )
        if( first <= ONE_BYTE_LIMIT ):
            ret_val = first
        elif( first < THREE_BYTE_HEADER ):
            second = store_to_num( num[idx + 1] )
            ret_val = ONE_BYTE_LIMIT+(BYTE_VALS*(first-(ONE_BYTE_LIMIT+1)))+second
            bytes_used = 2
        elif( first == THREE_BYTE_HEADER ):
            second = store_to_num( num[idx + 1] )
            third = store_to_num( num[idx + 2] )
            ret_val = (TWO_BYTE_LIMIT+1)+(BYTE_VALS*second)+third
            bytes_used = 3
        else:
            data_bytes = first-247
            start = data_bytes - 1
            ret_val = 0
            i = 1

            mod = (buckets[start-BUCKET_OFFSET]['limit']+1) // BYTE_VALS

            while( start >= 0 ):
                ret_val = ret_val + (mod * store_to_num( num[idx + i] )) 
                i = i + 1
                start = start - 1
                mod = mod // BYTE_VALS

            bytes_used = data_bytes + 1
        yield ret_val
        idx += bytes_used
