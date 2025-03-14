import numpy as np
import cupy as cp
from typing import Optional, Union, Tuple, Callable, NewType, List

"""
Gurobi cannot utilize a GPU (and it really cannot benefit from it as-is
anyway, since Barrier/Simplex run well enough on a CPU). However, our
algorithm does benefit greatly from a GPU in certain parts.

We need to make explicit what lives in RAM and what lives on the GPU
memory, since going over the bus to convert them would be huge hit
in terms of performance.
"""

CPUArray = np.ndarray
"""Alias for `numpy.ndarray`, an array that lives on the RAM. Plenty of space usually."""
GPUArray = NewType('GPUArray', cp.ndarray)
"""Alias for `cupy.ndarray`, an array that lives on the GPU memory. Usually quite limited."""
ScatteredGPUArray = NewType('ScatteredGPUArray', Tuple[GPUArray])
"""Designates arrays that are shared among all devices as-is"""
PartitionedGPUArray = NewType('PartitionedGPUArray', List[GPUArray])
"""A 2D matrix that has been partitioned column-wise over GPU devices"""

"""
For very large topologies, GPU memory becomes very tight.
We need to switch to half precision for such cases.
"""

DOUBLE_PRECISION = 'double'  # float 64
SINGLE_PRECISION = 'single'  # float 32
HALF_PRECISION = 'half'      # float 16

NUMBER_OF_GPU_DEVICES: int = cp.cuda.runtime.getDeviceCount()
GPU_MEM_MANAGER: cp.cuda.MemoryPool = cp.get_default_memory_pool()


_GLOBAL_PRECISION: Optional[int] = None
"""
Any algorithm that uses GPU features MUST explicitly set this, otherwise it will
get an error later (which we actually want, since it was probably never intended
to be this way)
"""

_CPU_DTYPE = None
_GPU_DTYPE = None


def set_global_precision(precision: str):
    global _GLOBAL_PRECISION, _CPU_DTYPE, _GPU_DTYPE
    assert _GLOBAL_PRECISION is None and _CPU_DTYPE is None and _GPU_DTYPE is None
    _GLOBAL_PRECISION = precision
    if precision == DOUBLE_PRECISION:
        _CPU_DTYPE = np.float64
        _GPU_DTYPE = cp.float64
    elif precision == SINGLE_PRECISION:
        _CPU_DTYPE = np.float32
        _GPU_DTYPE = cp.float32
    elif precision == HALF_PRECISION:
        _CPU_DTYPE = np.float16
        _GPU_DTYPE = cp.float16
    else:
        raise ValueError


def get_total_reserved_gpu_memory_usage() -> int:
    """The amount of allocated GPU memory usage across all devices (including caches ...)"""
    total = 0
    for i in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(i):
            total += GPU_MEM_MANAGER.total_bytes()
    return total


def get_total_used_gpu_memory_usage() -> int:
    """The GPU working set, across all devices"""
    total = 0
    for i in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(i):
            total += GPU_MEM_MANAGER.used_bytes()
    return total


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
            out.append(as_gpu_array(array[:, sum_col:columns[dev]]))
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
    return [f(*partition, *args, **kwargs) for partition in zip(*arrays)]


def reduce_to_cpu(array: PartitionedGPUArray, f: Callable, *args, **kwargs) -> CPUArray:
    return f(np.array([as_cpu_array(item) for item in array]).T, *args, **kwargs)


def rebuild_to_cpu(array: PartitionedGPUArray) -> CPUArray:
    return np.hstack([as_cpu_array(item) for item in array])
