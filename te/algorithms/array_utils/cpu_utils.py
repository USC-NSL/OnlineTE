import numpy as np
from te.algorithms.array_utils import get_global_precision, DOUBLE_PRECISION, SINGLE_PRECISION, HALF_PRECISION
from typing import Tuple, Callable, Any

CPUArray = np.ndarray
"""Alias for `numpy.ndarray`, an array that lives on the RAM. Plenty of space usually."""


_CPU_DTYPE = None
"""Every Numpy array that we instantiate must adhere to this data type"""


def set_cpu_float_precision():
    global _CPU_DTYPE
    assert _CPU_DTYPE is None
    _CPU_DTYPE = ({
        DOUBLE_PRECISION: np.float64,
        SINGLE_PRECISION: np.float32,
        HALF_PRECISION: np.float16
    })[get_global_precision()]


cpu_zeros: Callable[[Tuple[int]], CPUArray]  = lambda shape: np.zeros(shape=shape, dtype=_CPU_DTYPE)
"""Wrapper for `nump.zeros`. Enforces the global data type"""
cpu_mmap: Callable[[str, Tuple[int], str], CPUArray] = \
    lambda path, shape, mode: np.lib.format.open_memmap(shape=shape, filename=path, mode=mode, dtype=_CPU_DTYPE)
"""Alias for MMAP"""
cpu_array: Callable[[Any], CPUArray] = lambda input: np.array(input, dtype=_CPU_DTYPE)
"""Create a copy of an array-like thing"""
cpu_frombuffer: Callable[[bytes, Tuple[int]], CPUArray] = \
    lambda data, shape: np.frombuffer(data, dtype=_CPU_DTYPE).reshape(shape)
"""Alias for `np.frombuffer`, but expects the shape as an argument"""
cpu_frombuffer_serial: Callable[[bytes], CPUArray] = \
    lambda data: np.frombuffer(data, dtype=_CPU_DTYPE)
"""Alias for `np.frombuffer`. Always returns a 1D array"""
cpu_dump: Callable[[str, CPUArray], None] = lambda path, data: np.save(path, data, allow_pickle=True)
"""Replacement for Joblib `dump`, it seems to not do what I expect it to"""
