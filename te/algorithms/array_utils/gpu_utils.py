import cupy as cp
import numpy as np
import cupyx.scipy.sparse as cps
from te.algorithms.array_utils import get_global_precision, DOUBLE_PRECISION, SINGLE_PRECISION, HALF_PRECISION
from te.algorithms.array_utils.cpu_utils import CPUArray, CPUCOOArray, CPUCSRArray, CPUCSCArray, is_float_array
from typing import Union, Tuple, Callable, Any, List

"""
Gurobi cannot utilize a GPU (and it really cannot benefit from it as-is
anyway, since Barrier/Simplex run well enough on a CPU). However, our
algorithm does benefit greatly from a GPU in certain parts.

We need to make explicit what lives in RAM and what lives on the GPU
memory, since going over the bus to convert them would be huge hit
in terms of performance.

This script also handles using _multiple_ GPUs on the same machine. By
default, it will assume that all GPUs are essentially the same and evenly
devide everything over them.
"""

GPUArray = cp.ndarray
"""Alias for `cupy.ndarray`, an array that lives on the GPU memory. Usually quite limited."""
GPUCOOArray = cps.coo_matrix
"""Alias for `cupyx.scipy.sparse.coo_matrix`"""
GPUCSRArray = cps.csr_matrix
"""Alias for `cupyx.scipy.sparse.csr_matrix`"""
GPUCSCArray = cps.csc_matrix
"""Alias for `cupyx.scipy.sparse.csc_matrix`"""
ScatteredGPUArray = Tuple[Union[GPUArray, GPUCSCArray, GPUCSRArray]]
"""Designates arrays that are shared among all devices as-is"""
PartitionedGPUArray = List[Union[GPUArray, GPUCSCArray]]
"""A 2D matrix that has been partitioned column-wise over GPU devices (so no CSR arrays!)"""
DoublePrecisionGPUArray = cp.ndarray
IntegerGPUArray = cp.ndarray
BooleanGPUArray = cp.ndarray


_GPU_DTYPE = None
"""Every CuPy array that we instantiate must adhere to this data type unless specified otherwise"""
NUMBER_OF_GPU_DEVICES: int = cp.cuda.runtime.getDeviceCount()
"""Number of available GPU devices (assumed to be of the same kind ... for now)"""
GPU_MEM_MANAGER: cp.cuda.MemoryPool = cp.get_default_memory_pool()
"""A global GPU memory manager (just to query memory usage, nothing too fancy)"""

def set_gpu_float_precision():
    global _GPU_DTYPE
    assert _GPU_DTYPE is None
    _GPU_DTYPE = ({
        DOUBLE_PRECISION: cp.float64,
        SINGLE_PRECISION: cp.float32,
        HALF_PRECISION: cp.float16
    })[get_global_precision()]


def _partitions(total: int, parts: int) -> List[int]:
    assert (total > parts)
    out = [total // parts for _ in range(parts)]
    out[0] += (total % parts)
    return out


gpu_zeros: Callable[[Tuple[int]], GPUArray]  = lambda shape: cp.zeros(shape=shape, dtype=_GPU_DTYPE)
"""Wrapper for `cupy.zeros`. Enforces the global data type."""

def gpu_partitioned_zeros(shape: Tuple[int]) -> PartitionedGPUArray:
    """
    Create a 2D array of zeros in the GPU memory that is column-wise partitioned
    over all available devices.
    """
    assert len(shape) == 2
    columns = _partitions(shape[-1], NUMBER_OF_GPU_DEVICES)
    out = []
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(gpu_zeros((shape[0], columns[dev])))
    return out


def gpu_scattered_zeros(shape: Tuple[int]) -> ScatteredGPUArray:
    """
    Create an identical array of zeros on each available GPU device.
    """
    out = []
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(gpu_zeros(shape))
    return tuple(out)


def as_gpu_array(
    cpu_thing: Union[CPUCSRArray, CPUCSCArray, CPUCOOArray, CPUArray],
    partition: bool = False
) -> Union[GPUArray, GPUCOOArray, GPUCSRArray, GPUCSCArray, PartitionedGPUArray]:
    """
    Copy array into GPU memory.
    The behavior of this function depends on input type and arguments.
    - The data type depends on whether the input array is a float array
      (i.e. float16/32/64, 128 is NOT expected). If the input is a float
      array, then it will be cast to the current global GPU data type.
      If it is non-float, the data type will be kept as is.
    - The output format depends on the array format and arguments:
        - For a dense input, the output is also a dense array, and if
          `partition`, will be partitioned over all available GPUs, yeilding
          a `PartitionedGPUArray`. Otherwise, the output is just a regular
          `GPUArray` on device index 0.
        - For a sparse (COO/CSC/CSR) array, the output will have the same
          sparsity pattern. If `partition` and the input is CSC, then the
          output is a `PartitionedGPUArray`, with each partition also a
          CSC array. If `partition` is `False`, then the output is a dense
          `GPUArray` on device index 0.
        - If `partition` is `True` and the input is CSR/COO, then this raises
          a `ValueError`, as a column-wise partition of the array like this
          is almost certainly a mistake.
    """
    if isinstance(cpu_thing, list): # Just a simple list, NOT a partitioned array
        return cp.array(cpu_thing, dtype=_GPU_DTYPE)
    
    copy_dtype = _GPU_DTYPE if is_float_array(cpu_thing) else cpu_thing.dtype
    if isinstance(cpu_thing, CPUArray):
        if partition:
            return _as_partitioned_gpu_array(cpu_thing)
        return cp.array(cpu_thing, dtype=copy_dtype)
    elif isinstance(cpu_thing, CPUCOOArray):
        if partition:
            raise ValueError('Column-wise partitioning of a COO is probably unintended!')
        return cps.coo_matrix(cpu_thing, dtype=copy_dtype)
    elif isinstance(cpu_thing, CPUCSRArray):
        if partition:
            raise ValueError('Column-wise partitioning of a CSR is probably unintended!')
        return cps.csr_matrix(cpu_thing, dtype=copy_dtype)
    elif isinstance(cpu_thing, CPUCSCArray):
        if partition:
            return _as_partitioned_gpu_array(cpu_thing)
        return cps.csc_matrix(cpu_thing, dtype=copy_dtype)
    else:
        raise ValueError(f'Unknown matrix type: {type(cpu_thing)}')


def as_cpu_array(
    gpu_thing: Union[GPUArray, GPUCOOArray, GPUCSRArray, GPUCSCArray, PartitionedGPUArray, ScatteredGPUArray],
    dense: bool = False
) -> Union[CPUCSRArray, CPUCSCArray, CPUCOOArray, CPUArray]:
    """
    Copy array into RAM with similar sparsity.
    If the input array has been scattered, then it will be rebuilt in host
    memory using `np.hstack`.
    If the input array is sparse (partitioned or not) and `dense` is `True`,
    then the output will be a regular `CPUArray` instead.
    """
    if isinstance(gpu_thing, cp.ndarray):
        return cp.asnumpy(gpu_thing)
    elif isinstance(gpu_thing, (CPUCOOArray, CPUCSRArray, CPUCSCArray)):
        res = gpu_thing.get()
        if dense:
            return res.toarray()
        return res
    elif isinstance(gpu_thing, tuple): # type:ScatteredGPUArray
        return as_cpu_array(gpu_thing[0], dense=dense)
    elif isinstance(gpu_thing, list): # type:PartitionedGPUArray
        return np.hstack([as_cpu_array(item, dense=dense) for item in gpu_thing])
    else:
        raise ValueError(f'Unknown matrix type: {type(gpu_thing)}')


def scattered_shape(array: Union[ScatteredGPUArray, PartitionedGPUArray]) -> Tuple[int]:
    """
    Get the shape of a scattered/partitioned GPU array.
    """
    if isinstance(array, tuple): # type:ScatteredGPUArray
        return array[0].shape
    rows = array[0].shape[0]
    cols = sum([a.shape[1] for a in array])
    return (rows, cols)


def as_scattered_gpu_arrray(array: Union[CPUArray, CPUCSCArray, CPUCSRArray, CPUCOOArray]) -> ScatteredGPUArray:
    """Scatter an array in host memory across all GPU devices."""
    out = []
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(as_gpu_array(array))
    return tuple(out)


def _scatter_from_index_0(array: GPUArray) -> ScatteredGPUArray:
    if NUMBER_OF_GPU_DEVICES == 1:
        return (array,)
    out = []
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(array)
    return tuple(out)


def _as_partitioned_gpu_array(array: Union[CPUArray, CPUCSCArray]) -> PartitionedGPUArray:
    assert not isinstance(array, CPUCSRArray), "Cannot column partition CSR arrays!"
    shape = array.shape
    assert len(shape) == 2
    columns = _partitions(shape[-1], NUMBER_OF_GPU_DEVICES)
    out = []
    sum_col = 0
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(as_gpu_array(array[:, sum_col:sum_col+columns[dev]]))
            sum_col += columns[dev]
    assert sum_col == shape[-1]
    return out


def gpu_sparse_to_dense(
    gpu_thing: Union[GPUCSCArray, GPUCOOArray, GPUCSRArray, ScatteredGPUArray, PartitionedGPUArray]
) -> Union[GPUArray, ScatteredGPUArray, PartitionedGPUArray]:
    if isinstance(gpu_thing, (GPUCSCArray, GPUCOOArray, GPUCSRArray)):
        return gpu_thing.toarray()
    elif isinstance(gpu_thing, tuple): # type:ScatteredGPUArray
        if isinstance(gpu_thing[0], GPUArray):
            return gpu_thing
        return tuple([a.toarray() for a in gpu_thing])
    elif isinstance(gpu_thing, list): # type:PartitionedGPUArray
        if isinstance(gpu_thing[0], GPUArray):
            return gpu_thing
        return [a.toarray() for a in gpu_thing]
    raise ValueError(f'Unknown sparse matrix type: {type(gpu_thing)}')


def zip_map(arrays: List[Union[ScatteredGPUArray, PartitionedGPUArray]], f: Callable,
            *args, **kwargs) -> Union[ScatteredGPUArray, PartitionedGPUArray]:
    """
    Function that performs operations on scattered/partitioned GPU arrays.
    These functions (`f`) must satisfy the following conditions:
    - Column-wise parallelization, where we can perform operations on any arbitrary
      set of columns and stitch them together.
    - Only use partitioned/scattered arrays already on the same devices.
    
    Example
    -------
    Assume we have two devices. The call:
    ```
    zip_map([A, B], lambda a, b: a + b)
    ```
    Is equivalent to:
    ```
    [a[0] + b[0], a[1] + b[1]]
    ```

    Note
    ----
    For matrix multiplication, this is only every useful if the multiplication is
    of the form `S @ P`, where `S` is a scattered array and `P` is a partitioned
    array.
    For our algorithm, this is always the case, since the only multiplication that
    we need is `N @ Y`, where `N` is the null-space basis of the topology, which 
    is small enough and can be scattered.
    """
    l = len(arrays[0])
    assert all(len(array) == l for array in arrays), f'Len = {[len(array) for array in arrays]}'
    out = []
    for dev, partition in enumerate(zip(*arrays)):
        with cp.cuda.Device(dev):
            out.append(f(*partition, *args, **kwargs))
    return out


def reduce_in_place(array: PartitionedGPUArray, f: Callable, gather: bool = False, *args, **kwargs) -> Union[ScatteredGPUArray, CPUArray]:
    """
    Perform a reducing function `f` on a partitioned array and return a scattered
    result on all devices.
    If `gather` is `True`, then the result is instead loaded into host memory
    instead of being scattered again.
    """
    partitioned_reduce = zip_map([array], lambda a: f(a, *args, **kwargs))
    aggregate = partitioned_reduce[0] if len(partitioned_reduce) == 1 else f(cp.array(partitioned_reduce).T, *args, **kwargs)
    if gather:
        return np.squeeze(as_cpu_array(aggregate))
    return _scatter_from_index_0(aggregate)


synchronize_to_device: Callable[[int], None] = lambda dev: cp.cuda.Device(dev).synchronize()
"""Wait until all operations on the slected device finish"""
def synchronize_to_all():
    """Wait until all operations on all GPU devices finish"""
    for dev in range(NUMBER_OF_GPU_DEVICES):
        synchronize_to_device(dev)


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


gpu_coo_array: Callable[[List[int], List[int], List[Any], Tuple[int]], GPUCOOArray] = \
    lambda rows, cols, data, shape: cps.coo_matrix(
        (as_gpu_array(data), (as_gpu_array(rows), as_gpu_array(cols))),
        shape=shape, dtype=_GPU_DTYPE
    )
gpu_coo_to_csr: Callable[[GPUCOOArray], GPUCSRArray] = lambda inp: inp.tocsr()


def gpu_coo_to_csc(coo: GPUCOOArray, partition: bool = False) -> Union[GPUCSCArray, PartitionedGPUArray]:
    res = coo.tocsc()
    if not partition:
        return res
    shape = res.shape
    columns = _partitions(shape[-1], NUMBER_OF_GPU_DEVICES)
    out = []
    sum_col = 0
    for dev in range(NUMBER_OF_GPU_DEVICES):
        with cp.cuda.Device(dev):
            out.append(as_gpu_array(res[:, sum_col:sum_col+columns[dev]]))
            sum_col += columns[dev]
    assert sum_col == shape[-1]
    return out
