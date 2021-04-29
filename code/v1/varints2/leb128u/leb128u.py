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

# Based on the unsigned integer encoding method described at
#  https://en.wikipedia.org/wiki/LEB128

from ..varints import num_types,generic_encode,generic_decode
from ..varints import store_to_num

def encode( num ):
    return generic_encode( num, funcs )

def decode( num ):
    return generic_decode( num, funcs )

def encode_generator( ):
    working = yield None
    while (not working is None):
        if working < 0:
            raise ValueError("Negative numbers not handled")
        byte = working & 0x7F
        working = working >> 7
        if( working != 0 ):
            byte = byte | 0x80
        yield byte
        while( working ):
            byte = working & 0x7F
            working = working >> 7
            if( working != 0 ):
                byte = byte | 0x80
            yield byte
        working = yield None

def decoding_generator( ):
    num = yield None
    while (not num is None):
        ret_val = None
        bytes_used = 0
        cont = True
        while cont:
            val = store_to_num( num[ bytes_used ] )
            if(( val & 0x80 ) == 0):
                cont = False
            val = val & 0x7F

            if ret_val is None:
                ret_val = 0

            ret_val = ret_val | (val << (7*bytes_used))
            bytes_used = bytes_used + 1
        num = yield (ret_val, bytes_used)
    yield None

funcs = { 'decoding_generator': decoding_generator,
          'encode_generator': encode_generator }
