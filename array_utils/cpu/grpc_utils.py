import grpc
import struct
import numpy as np
import networkx as nx
import protos.array.array_pb2 as array_messages
import protos.graph.graph_pb2 as graph_messages
from collections.abc import Iterator as IteratorABC
from typing import Optional, Iterator, Union, List, Tuple, Dict, Type, Generator
from .. import get_global_precision
from .types import *
from .buffer_ops import *
from .sparse.types import *
from .sparse.buffer_ops import *


ARRAY_TYPE_MAP: Dict[Type, int] = {
    None: 0,
    np.dtype('bool'): 1,
    np.dtype('int32'): 2,
    np.dtype('float16'): 3,
    np.dtype('float32'): 4,
    np.dtype('float64'): 5
}
"""
Maps a type to a `char` sized value that will be used to pack the array into a struct.
A `None` type means to cast to the current global CPU/GPU data type.
"""
REVERSE_ARRAY_TYPE_MAP: Dict[int, Type] = {v: k for k, v in ARRAY_TYPE_MAP.items()}

get_array_type_index = lambda tp: ARRAY_TYPE_MAP[np.result_type(tp if tp is not None else get_global_precision())]


def array_to_serialized_message(array: Optional[np.ndarray]) -> Optional[array_messages.SerializedNumpyArrayMessage]:
    """
    Serialize a Numpy array into a single chunk of bytes to send (this can be a really large packet).
    The output RPC message contains the array shape too.
    """
    if array is not None:
        return array_messages.SerializedNumpyArrayMessage(array=array.tobytes(), dims=array.shape, dtype=get_array_type_index(array.dtype))

def serialized_message_to_array(message: Optional[array_messages.SerializedNumpyArrayMessage]) -> Optional[np.ndarray]:
    """Deserialize a chunk of bytes into a Numpy array (the input RPC request contains the shape as well)"""
    if message is not None:
        return cpu_frombuffer(message.array, tuple(message.dims), REVERSE_ARRAY_TYPE_MAP[message.dtype])

def array_list_to_serialized_message(array_list: List[CPUArray]) -> Generator[array_messages.SerializedNumpyArrayMessage, None, None]:
    for array in array_list:
        yield array_to_serialized_message(array)

def serialized_message_to_array_list(message_stream: Iterator[array_messages.SerializedNumpyArrayMessage]) -> List[CPUArray]:
    return [serialized_message_to_array(array) for array in message_stream]

get_optional_field = lambda request, field_name: getattr(request, field_name) if request.HasField(field_name) else None


ARRAY_PREAMBLE_STRUCT_FORMAT = "BB?"
"""
First unsigned `char` is the number of dimensions for the array, and the
next unsigned `char` is the data type flag based on `ARRAY_TYPE_MAP`.
The final byte is a boolean, representing whether or not the array is `CSR`.
"""

def parse_array_preamble(premble: bytes) -> Tuple[int, Type, str, bool]:
    ndims, type_num, is_csr = struct.unpack(ARRAY_PREAMBLE_STRUCT_FORMAT, premble)
    return ndims, REVERSE_ARRAY_TYPE_MAP[type_num], "I"*ndims, is_csr


def get_csr_lengths(pack: bytes) -> Tuple[int, int, int]:
    return struct.unpack("I"*3, pack)


def chunk_big_array(array: Union[CPUArray, CPUCSRArray], chunk_size: int,
                    dtype: Optional[Type] = None) -> Generator[array_messages.Chunk, None, None]:
    """
    Chunk a large array into a sequence of bytes of at most `chunk_size` length.
    Records the number of dimensions and the data type of the array.
    We have two cases, one for `np.ndarray` and one for `scipy.sparse.csr_array`:
    In both cases, we return a generator of `Chunk` messages, where:
    - The first chunk is the preamble, encoding the number of dimensions and data type
    - The second chunk encodes the array shape
    For `np.ndarray`:
    - The rest are the array itself
    For `scipy.sparse.csr_array`:
    - A third chunk encodes the lengths of the data/index/pointer arrays.
    - The final chunk contains the concatanation of data/index/pointer array buffers.
    """
    # TODO: Make this more lazy (e.g. avoid copying). This array can be really really big!
    # First, send the array preamble
    is_csr = isinstance(array, CPUCSRArray)
    yield array_messages.Chunk(data=struct.pack(ARRAY_PREAMBLE_STRUCT_FORMAT, array.ndim, get_array_type_index(dtype), is_csr))
    # Now, the shape
    yield array_messages.Chunk(data=struct.pack("I"*array.ndim, *(array.shape)))
    if not is_csr:
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
    else:
        data = array.data.tobytes()
        indices = array.indices.tobytes()
        pointers = array.indptr.tobytes()
        # Send the lengths
        yield array_messages.Chunk(data=struct.pack("I"*3, *((len(data), len(indices), len(pointers)))))
        # Now the array itself
        data += indices
        del indices
        data += pointers
        del pointers
        total = len(data)
        while total > 0:
            pos = -total + chunk_size
            if pos >= 0:
                yield array_messages.Chunk(data=data[-total:])
            else:
                yield array_messages.Chunk(data=data[-total:pos])
            total -= chunk_size


def rebuild_chunked_array(chunks: Union[Iterator[array_messages.Chunk], List[array_messages.Chunk]]) -> Union[CPUArray, CPUCSRArray]:
    """Rebuild an array from gathered byte chunks (This assumes a synchronous stream)"""
    if isinstance(chunks, list):
        chunk_iter = iter(chunks)
    elif isinstance(chunks, IteratorABC):
        chunk_iter = chunks
    else:
        raise ValueError(f'Unexpected type: {type(chunks)}')
    _, dtype, shape_struct_str, is_csr = parse_array_preamble(next(chunk_iter).data)
    shape = struct.unpack(shape_struct_str, next(chunk_iter).data)
    if not is_csr:
        arrays = []
        for chunk in chunk_iter:
            arrays.append(cpu_frombuffer_serial(chunk.data, dtype=dtype))
        return np.hstack(arrays).reshape(shape)
    else:
        buffer = b''
        lens = get_csr_lengths(next(chunk_iter).data)
        for chunk in chunk_iter:
            buffer += chunk.data
        return cpu_csr_frombuffer(buffer, shape, lens, dtype)


async def async_rebuild_chunked_array(chunk_async_stream: grpc.aio._call.UnaryStreamCall) -> Union[CPUArray, CPUCSRArray]:
    """Rebuild an array from gathered byte chunks (This assumes am asymchronous stream)"""
    preamble_chunk: array_messages.Chunk = await chunk_async_stream.read()
    _, dtype, shape_struct_str, is_csr = parse_array_preamble(preamble_chunk.data)
    shape_chunk: array_messages.Chunk = await chunk_async_stream.read()
    shape = struct.unpack(shape_struct_str, shape_chunk.data)
    if not is_csr:
        arrays = []
        next_chunk: array_messages.Chunk = await chunk_async_stream.read()
        while next_chunk != grpc.aio.EOF:
            arrays.append(cpu_frombuffer_serial(next_chunk.data, dtype=dtype))
            next_chunk = await chunk_async_stream.read()
        return np.hstack(arrays).reshape(shape)
    else:
        buffer = b''
        lens_chunk: array_messages.Chunk = await chunk_async_stream.read()
        lens = get_csr_lengths(lens_chunk.data)
        next_chunk: array_messages.Chunk = await chunk_async_stream.read()
        while next_chunk != grpc.aio.EOF:
            # arrays.append(cpu_frombuffer_serial(next_chunk.data, dtype=dtype))
            buffer += next_chunk.data
            next_chunk = await chunk_async_stream.read()
        return cpu_csr_frombuffer(buffer, shape, lens, dtype)


def graph_to_serialized_message(graph: nx.DiGraph) -> graph_messages.Topology:
    payload = graph_messages.Topology()
    payload.node_ids.extend([n for n in graph.nodes(data=False)])
    # TODO: Add other attributes to support here!!!
    for u, v, data in graph.edges(data=True):
        edge = payload.edges.add(source=u, target=v)
        for key, val in data.items():
            if key in {'capacity'}:
                edge.attributes[key] = val
    return payload


def serialized_message_to_graph(message: graph_messages.Topology) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(message.node_ids)
    for edge in message.edges:
        graph.add_edge(edge.source, edge.target, **dict(edge.attributes))
    return graph


__all__ = [
    'ARRAY_TYPE_MAP', 'REVERSE_ARRAY_TYPE_MAP', 'ARRAY_PREAMBLE_STRUCT_FORMAT',
    'array_to_serialized_message', 'serialized_message_to_array', 'parse_array_preamble',
    'chunk_big_array','rebuild_chunked_array', 'async_rebuild_chunked_array',
    'array_list_to_serialized_message', 'serialized_message_to_array_list',
    'graph_to_serialized_message', 'serialized_message_to_graph'
]