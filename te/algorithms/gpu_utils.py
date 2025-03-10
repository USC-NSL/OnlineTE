import numpy as np
import cupy as cp
from typing import Optional, Tuple, Callable, NewType

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

"""
For very large topologies, GPU memory becomes very tight.
We need to switch to half precision for such cases.
"""

DOUBLE_PRECISION = 'double'  # float 64
SINGLE_PRECISION = 'single'  # float 32
HALF_PRECISION = 'half'      # float 16


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


cpu_zeros: Callable[[Tuple[int]], CPUArray]  = lambda shape: np.zeros(shape=shape, dtype=_CPU_DTYPE)
gpu_zeros: Callable[[Tuple[int]], GPUArray]  = lambda shape: cp.zeros(shape=shape, dtype=_GPU_DTYPE)
as_gpu_array: Callable[[CPUArray], GPUArray] = lambda array: cp.array(array, dtype=_GPU_DTYPE)
as_cpu_array: Callable[[GPUArray], CPUArray] = lambda array: cp.asnumpy(array)
cpu_memmap: Callable[[str, Tuple[int], str], CPUArray] = \
    lambda path, shape, mode: np.memmap(shape=shape, filename=path, mode=mode, dtype=_CPU_DTYPE)
