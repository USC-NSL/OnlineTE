import struct
import numpy as np
import protos.distributed_lp.distributed_lp_pb2 as distributed_lp_messages
from typing import Optional, Iterator


def array_to_serialized_message(array: Optional[np.ndarray]) -> Optional[distributed_lp_messages.SerializedNumpyArrayMessage]:
    if array is not None:
        return distributed_lp_messages.SerializedNumpyArrayMessage(array=array.tobytes(), dims=list(array.shape))

def serialized_message_to_array(message: Optional[distributed_lp_messages.SerializedNumpyArrayMessage]) -> Optional[np.ndarray]:
    if message is not None:
        return np.reshape(np.frombuffer(message.array), tuple(message.dims))

get_optional_field = lambda request, field_name: getattr(request, field_name) if request.HasField(field_name) else None


array_2d_dim_struct_format = "II"


def chunk_big_array(array: np.ndarray, chunk_size: int):
    assert array.ndim == 2
    # TODO: Make this more lazy (e.g. avoid copying). This array can be really really big!
    # First, send the array dimenssions ..
    yield distributed_lp_messages.Chunk(data=struct.pack(array_2d_dim_struct_format, *(array.shape)))
    # Now the array itself
    buffer = array.tobytes()
    total = len(buffer)
    while total > 0:
        pos = -total + chunk_size
        if pos >= 0:
            yield distributed_lp_messages.Chunk(data=buffer[-total:])
        else:
            yield distributed_lp_messages.Chunk(data=buffer[-total:pos])
        total -= chunk_size


def rebuild_chunked_array(chunks: Iterator[distributed_lp_messages.Chunk]) -> np.ndarray:
    shape = struct.unpack(array_2d_dim_struct_format, next(chunks).data)
    buffer = b''
    for chunk in chunks:
        buffer += chunk.data
    return np.frombuffer(buffer).reshape(shape)
