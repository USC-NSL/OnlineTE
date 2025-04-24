import grpc
import struct
import numpy as np
import protos.array.array_pb2 as array_messages
from collections.abc import Iterator as IteratorABC
from typing import Optional, Iterator, Union, List
from te.algorithms.array_utils.cpu_utils import cpu_frombuffer, cpu_frombuffer_serial


GRPC_ARRAY_STREAM_MAX_LEN = 2**20


# TODO: Move this file out of this package into a utility module, we'll need it in a lot of places


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


array_2d_dim_struct_format = "II"


def chunk_big_array(array: np.ndarray, chunk_size: int):
    """Chunk a large array into a sequence of bytes of at most `chunk_size` length"""
    assert array.ndim == 2
    # TODO: Make this more lazy (e.g. avoid copying). This array can be really really big!
    # First, send the array dimenssions ..
    yield array_messages.Chunk(data=struct.pack(array_2d_dim_struct_format, *(array.shape)))
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
        shape = struct.unpack(array_2d_dim_struct_format, chunks[0].data)
        for chunk in chunks[1:]:
            arrays.append(cpu_frombuffer_serial(chunk.data))
    elif isinstance(chunks, IteratorABC):
        shape = struct.unpack(array_2d_dim_struct_format, next(chunks).data)
        for chunk in chunks:
            arrays.append(cpu_frombuffer_serial(chunk.data))
    else:
        raise ValueError(f'Unexpected type: {type(chunks)}')
    return np.hstack(arrays).reshape(shape)


async def async_rebuild_chunked_array(chunk_async_stream: grpc.aio._call.UnaryStreamCall) -> np.ndarray:
    """Rebuild an array from gathered byte chunks (This assumes am asymchronous stream)"""
    arrays = []
    shape_chunk: array_messages.Chunk = await chunk_async_stream.read()
    shape = struct.unpack(array_2d_dim_struct_format, shape_chunk.data)
    next_chunk: array_messages.Chunk = await chunk_async_stream.read()
    while next_chunk != grpc.aio.EOF:
        arrays.append(cpu_frombuffer_serial(next_chunk.data))
        next_chunk = await chunk_async_stream.read()
    return np.hstack(arrays).reshape(shape)
