import numpy as np
import cupy as cp
from te.algorithms.array_utils import get_global_precision, DOUBLE_PRECISION, SINGLE_PRECISION, HALF_PRECISION
from te.algorithms.array_utils.cpu_utils import CPUArray
from typing import Union, Tuple, Callable, NewType, List

"""
Gurobi cannot utilize a GPU (and it really cannot benefit from it as-is
anyway, since Barrier/Simplex run well enough on a CPU). However, our
algorithm does benefit greatly from a GPU in certain parts.

We need to make explicit what lives in RAM and what lives on the GPU
memory, since going over the bus to convert them would be huge hit
in terms of performance.
"""

GPUArray = NewType('GPUArray', cp.ndarray)
"""Alias for `cupy.ndarray`, an array that lives on the GPU memory. Usually quite limited."""
ScatteredGPUArray = NewType('ScatteredGPUArray', Tuple[GPUArray])
"""Designates arrays that are shared among all devices as-is"""
PartitionedGPUArray = NewType('PartitionedGPUArray', List[GPUArray])
"""A 2D matrix that has been partitioned column-wise over GPU devices"""


_GPU_DTYPE = None
"""Every CuPy array that we instantiate must adhere to this data type"""
NUMBER_OF_GPU_DEVICES: int = cp.cuda.runtime.getDeviceCount()
"""Number of available GPU devices (assumed to be of the same kind ... for now)"""
GPU_MEM_MANAGER: cp.cuda.MemoryPool = cp.get_default_memory_pool()
"""A global GPU memory manager (just to query memory usage, nothing too fancy)"""

def set_precision():
    global _CPU_DTYPE
    assert _CPU_DTYPE is None
    _CPU_DTYPE = ({
        DOUBLE_PRECISION: cp.float64,
        SINGLE_PRECISION: cp.float32,
        HALF_PRECISION: cp.float16
    })[get_global_precision()]


def get_total_reserved_gpu_memory_usage() -> int:
    """The average amount of allocated GPU memory usage across all devices (including caches ...)"""
    total = 0
    for i in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(i):
            total += GPU_MEM_MANAGER.total_bytes()
    return total // NUMBER_OF_GPU_DEVICES


def get_total_used_gpu_memory_usage() -> int:
    """The average GPU working set, across all devices"""
    total = 0
    for i in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(i):
            total += GPU_MEM_MANAGER.used_bytes()
    return total // NUMBER_OF_GPU_DEVICES


cpu_zeros: Callable[[Tuple[int]], CPUArray]  = lambda shape: np.zeros(shape=shape, dtype=_CPU_DTYPE)
"""Wrapper for `nump.zeros`. Enforces the global data type."""
gpu_zeros: Callable[[Tuple[int]], GPUArray]  = lambda shape: cp.zeros(shape=shape, dtype=_GPU_DTYPE)
"""Wrapper for `cupy.zeros`. Enforces the global data type."""
as_gpu_array: Callable[[CPUArray], GPUArray] = lambda array: cp.array(array, dtype=_GPU_DTYPE)
"""Move array into GPU memory."""
as_cpu_array: Callable[[GPUArray], CPUArray] = lambda array: cp.asnumpy(array)
"""Move array into CPU memory."""
cpu_memmap: Callable[[str, Tuple[int], str], CPUArray] = \
    lambda path, shape, mode: np.memmap(shape=shape, filename=path, mode=mode, dtype=_CPU_DTYPE)


synchronize_to_device: Callable[[int], None] = lambda dev: cp.cuda.Device(dev).synchronize()
"""Wait until all operations on the slected device finish"""
def synchronize_to_all():
    """Wait until all operations on all GPU devices finish"""
    for dev in range(NUMBER_OF_GPU_DEVICES):
        synchronize_to_device(dev)


"""Multi-GPU functions"""


def scattered_shape(array: Union[ScatteredGPUArray, PartitionedGPUArray]) -> Tuple[int]:
    return tuple([len(array)] + [a.shape for a in array])


def partitions(total: int, parts: int) -> List[int]:
    assert (total > parts)
    out = [total // parts for _ in range(parts)]
    out[0] += (total % parts)
    return out


def as_scattered_gpu_arrray(array: CPUArray) -> ScatteredGPUArray:
    out = []
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(as_gpu_array(array))
    return tuple(out)


def as_partitioned_gpu_array(array: CPUArray) -> PartitionedGPUArray:
    shape = array.shape
    assert len(shape) == 2
    columns = partitions(shape[-1], NUMBER_OF_GPU_DEVICES)
    out = []
    sum_col = 0
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(as_gpu_array(array[:, sum_col:sum_col+columns[dev]]))
            sum_col += columns[dev]
    assert sum_col == shape[-1]
    return out


def gpu_partitioned_zeros(shape: Tuple[int]) -> PartitionedGPUArray:
    assert len(shape) == 2
    columns = partitions(shape[-1], NUMBER_OF_GPU_DEVICES)
    out = []
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(gpu_zeros((shape[0], columns[dev])))
    return out


def gpu_scattered_zeros(shape: Tuple[int]) -> ScatteredGPUArray:
    out = []
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(gpu_zeros(shape))
    return tuple(out)


def zip_map(arrays: List[Union[ScatteredGPUArray, PartitionedGPUArray]], f: Callable,
            *args, **kwargs) -> Union[ScatteredGPUArray, PartitionedGPUArray]:
    l = len(arrays[0])
    assert all(len(array) == l for array in arrays), f'Len = {[len(array) for array in arrays]}'
    out = []
    for dev, partition in enumerate(zip(*arrays)):
        with cp.cuda.Device(dev):
            out.append(f(*partition, *args, **kwargs))
    return out


def reduce_to_cpu(array: PartitionedGPUArray, f: Callable, *args, **kwargs) -> CPUArray:
    return f(np.array([as_cpu_array(item) for item in array]).T, *args, **kwargs)


def rebuild_to_cpu(array: PartitionedGPUArray) -> CPUArray:
    return np.hstack([as_cpu_array(item) for item in array])
