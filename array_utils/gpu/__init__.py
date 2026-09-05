import cupy as cp
from array_utils import *


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