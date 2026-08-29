import numpy as np
from array_utils import *
from typing import Optional


_CPU_DTYPE = None
"""Every Numpy array that we instantiate must adhere to this data type unless specified otherwise"""


def set_cpu_float_precision(precision: Optional[str] = None):
    global _CPU_DTYPE
    assert _CPU_DTYPE is None
    if precision is not None:
        set_global_precision(precision)
    _CPU_DTYPE = ({
        DOUBLE_PRECISION: np.float64,
        SINGLE_PRECISION: np.float32,
        HALF_PRECISION: np.float16
    })[get_global_precision()]


def get_cpu_float_precision() -> np.typing.DTypeLike:
    assert _CPU_DTYPE is not None
    return _CPU_DTYPE


__all__ = [
    'set_cpu_float_precision', 'get_cpu_float_precision',
    'HALF_PRECISION', 'SINGLE_PRECISION', 'DOUBLE_PRECISION'
]