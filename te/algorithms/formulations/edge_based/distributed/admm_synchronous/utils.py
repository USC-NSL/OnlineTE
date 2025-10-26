import grpc
import struct
import numpy as np
import protos.array.array_pb2 as array_messages
from collections.abc import Iterator as IteratorABC
from typing import Optional, Iterator, Union, List, Tuple, Dict, Type, Generator
from te.algorithms.array_utils.cpu_utils import cpu_frombuffer, cpu_frombuffer_serial


GRPC_ARRAY_STREAM_MAX_LEN = 2**20


# TODO: Move this file out of this package into a utility module, we'll need it in a lot of places

ARRAY_TYPE_MAP: Dict[Type, int] = {
    None: 0,
    bool: 1,
    np.int32: 2
}
"""
Maps a type to a `char` sized value that will be used to pack the array into a struct.
A `None` type means to cast to the current global CPU/GPU data type.
"""
REVERSE_ARRAY_TYPE_MAP: Dict[int, Type] = {v: k for k, v in ARRAY_TYPE_MAP.items()}


def array_to_serialized_message(array: Optional[np.ndarray]) -> Optional[array_messages.SerializedNumpyArrayMessage]:
    """
    Serialize a Numpy array into a single chunk of bytes to send (this can be a really large packet).
    The output RPC message contains the array shape too.
    """
    if array is not None:
        return array_messages.SerializedNumpyArrayMessage(array=array.tobytes(), dims=list(array.shape))

def serialized_message_to_array(message: Optional[array_messages.SerializedNumpyArrayMessage]) -> Optional[np.ndarray]:
    """Deserialize a chunk of bytes into a Numpy array (the input RPC request contains the shape as well)"""
    if message is not None:
        return cpu_frombuffer(message.array, tuple(message.dims))

get_optional_field = lambda request, field_name: getattr(request, field_name) if request.HasField(field_name) else None


ARRAY_PREAMBLE_STRUCT_FORMAT = "BB"
"""
First unsigned `char` is the number of dimensions for the array, and the
next unsigned `char` is the data type flag based on `ARRAY_TYPE_MAP`.
"""

def parse_array_preamble(premble: bytes) -> Tuple[int, Type, str]:
    ndims, type_num = struct.unpack(ARRAY_PREAMBLE_STRUCT_FORMAT, premble)
    return ndims, REVERSE_ARRAY_TYPE_MAP[type_num], "I"*ndims


def chunk_big_array(array: np.ndarray, chunk_size: int, dtype: Optional[Type] = None) -> Generator[array_messages.Chunk, None, None]:
    """
    Chunk a large array into a sequence of bytes of at most `chunk_size` length.
    Records the number of dimensions and the data type of the array.
    It returns a generator of `Chunk` messages, where:
    - The first chunk is the preamble, encoding the number of dimensions and data type
    - The second chunk encodes the array shape
    - The rest are the array itself
    """
    # TODO: Make this more lazy (e.g. avoid copying). This array can be really really big!
    # First, send the array preamble
    yield array_messages.Chunk(data=struct.pack(ARRAY_PREAMBLE_STRUCT_FORMAT, array.ndim, ARRAY_TYPE_MAP[dtype]))
    # Now, the shape
    yield array_messages.Chunk(data=struct.pack("I"*array.ndim, *(array.shape)))
    # Now the array itself
    buffer = array.tobytes()
    total = len(buffer)
    while total > 0:
        pos = -total + chunk_size
        if pos >= 0:
            yield array_messages.Chunk(data=buffer[-total:])
        else:
            yield array_messages.Chunk(data=buffer[-total:pos])
        total -= chunk_size


def rebuild_chunked_array(chunks: Union[Iterator[array_messages.Chunk], List[array_messages.Chunk]]) -> np.ndarray:
    """Rebuild an array from gathered byte chunks (This assumes a synchronous stream)"""
    arrays = []
    if isinstance(chunks, list):
        _, dtype, shape_struct_str = parse_array_preamble(chunks[0].data)
        shape = struct.unpack(shape_struct_str, chunks[1].data)
        for chunk in chunks[2:]:
            arrays.append(cpu_frombuffer_serial(chunk.data, dtype=dtype))
    elif isinstance(chunks, IteratorABC):
        _, dtype, shape_struct_str = parse_array_preamble(next(chunks).data)
        shape = struct.unpack(shape_struct_str, next(chunks).data)
        for chunk in chunks:
            arrays.append(cpu_frombuffer_serial(chunk.data, dtype=dtype))
    else:
        raise ValueError(f'Unexpected type: {type(chunks)}')
    return np.hstack(arrays).reshape(shape)


async def async_rebuild_chunked_array(chunk_async_stream: grpc.aio._call.UnaryStreamCall) -> np.ndarray:
    """Rebuild an array from gathered byte chunks (This assumes am asymchronous stream)"""
    arrays = []
    preamble_chunk: array_messages.Chunk = await chunk_async_stream.read()
    _, dtype, shape_struct_str = parse_array_preamble(preamble_chunk.data)
    shape_chunk: array_messages.Chunk = await chunk_async_stream.read()
    shape = struct.unpack(shape_struct_str, shape_chunk.data)
    next_chunk: array_messages.Chunk = await chunk_async_stream.read()
    while next_chunk != grpc.aio.EOF:
        arrays.append(cpu_frombuffer_serial(next_chunk.data, dtype=dtype))
        next_chunk = await chunk_async_stream.read()
    return np.hstack(arrays).reshape(shape)
