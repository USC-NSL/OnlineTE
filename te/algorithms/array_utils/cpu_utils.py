import numpy as np
import scipy.sparse as sp
from te.algorithms.array_utils import get_global_precision, DOUBLE_PRECISION, SINGLE_PRECISION, HALF_PRECISION
from typing import Tuple, Callable, Any, Optional, Type, List

CPUArray = np.ndarray
"""Alias for `numpy.ndarray`, an array that lives on the RAM. Plenty of space usually."""
CPUCOOArray = sp.coo_array
"""Alias for `scipy.sparse.coo_matrix`"""
CPUCSRArray = sp.csr_array
"""Alias for `scipy.sparse.csr_matrix`"""
DoublePrecisionCPUArray = np.ndarray
"""
Alias for `numpy.ndarray`.
The reason for this one being used is because any array that interacts with the solver
for Gurobi, needs to be double precision (otherwise, it is possible that we end up
passing `Inf` or `NaN` to Gurobi).
These arrays are guaranteed to be `np.float64` regardless of the global CPU array
data type.
Having a separate type alias for them helps with knowing which array can safely be
passed into Gurobi objective expressions.
"""
IntegerCPUArray = np.ndarray
"""
Alias for a `numpy.ndarray` instance that is known to be `np.int32`.
This is mostly for the path-based solution that projects onto the pinned probability
simplex. We need to keep certain arrays as integers to prevent the projection from
returning noisy output.
"""
BooleanCPUArray = np.ndarray
"""
Alias for a `numpy.ndarray` instance that is known to be a `bool` type.
This is for the path indicator matrix for path-based solvers. We need to be careful
with this, since if we mistakenly cast it to float, things still work fine but suddenly
everything takes forever and eats a lot of memory.
"""


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
"""Wrapper for `np.zeros`. Enforces the global data type"""

def cpu_mmap(path: str, shape: Tuple[int], mode: str, dtype: Optional[Type] = None):
    """Alias for MMAP"""
    if dtype is None:
        return np.lib.format.open_memmap(shape=shape, filename=path, mode=mode, dtype=_CPU_DTYPE)
    else:
        return np.lib.format.open_memmap(shape=shape, filename=path, mode=mode, dtype=dtype)

def cpu_array(thing: Any) -> CPUArray:
    """Create a copy of an array-like thing"""
    if isinstance(thing, (CPUArray, BooleanCPUArray, DoublePrecisionCPUArray, IntegerCPUArray)):
        return np.array(thing, dtype=_CPU_DTYPE)
    elif isinstance(thing, (list, tuple)):
        return np.array(thing, dtype=_CPU_DTYPE)
    elif isinstance(thing, (CPUCOOArray, CPUCSRArray)):
        return thing.toarray()

def cpu_frombuffer(buffer: bytes, shape: Tuple[int], dtype: Optional[Type] = None) -> CPUArray:
    """Alias for `np.frombuffer`, but expects the shape as an argument"""
    if dtype is None:
        return np.frombuffer(buffer, dtype=_CPU_DTYPE).reshape(shape)
    else:
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

def cpu_frombuffer_serial(buffer: bytes, dtype: Optional[Type] = None) -> CPUArray:
    """Alias for `np.frombuffer`. Always returns a 1D array"""
    if dtype is None:
        return np.frombuffer(buffer, dtype=_CPU_DTYPE)
    else:
        return np.frombuffer(buffer, dtype=dtype)

def cpu_csr_frombuffer(buffer: bytes, shape: Tuple[int], lens: Tuple[int], dtype: Optional[Type] = None) -> CPUCSRArray:
    """Given the 1D buffer of a CSR array, rebuilds it from the given lengths and shape"""
    assert len(lens) == 3
    d_len, i_len, p_len = lens
    assert len(buffer) == d_len + i_len + p_len
    data = cpu_frombuffer_serial(buffer[:d_len], dtype)
    indices = cpu_frombuffer_serial(buffer[d_len:d_len+i_len], np.int64)
    pointers = cpu_frombuffer_serial(buffer[d_len+i_len:], np.int64)
    return sp.csr_array((data, indices, pointers), shape=shape)

cpu_dump: Callable[[str, CPUArray], None] = lambda path, data: np.save(path, data, allow_pickle=True)
"""Replacement for Joblib `dump`, it seems to not do what I expect it to"""

cpu_double_array: Callable[[Any], DoublePrecisionCPUArray] = lambda input: np.array(input, dtype=np.float64)
"""Create a copy of an array-like thing that is always double precision, regardles of global data type"""
cpu_double_zeros: Callable[[Tuple[int]], DoublePrecisionCPUArray] = lambda shape: np.zeros(shape=shape, dtype=np.float64)
"""Always returns zero array with double precision, regardless of global data type"""

cpu_int_array: Callable[[Any], IntegerCPUArray] = lambda input: np.array(input, dtype=np.int32)
"""Create a copy of an array-like thing that is always `np.int32`, regardles of global data type"""
cpu_int_zeros: Callable[[Tuple[int]], IntegerCPUArray] = lambda shape: np.zeros(shape=shape, dtype=np.int32)
"""Always returns zero array with `np.int32`, regardless of global data type"""

cpu_cast_float: Callable[[Any], Any] = lambda val: _CPU_DTYPE(val)

cpu_coo_array: Callable[[List[int], List[int], List[Any], Tuple[int]], CPUCOOArray] = \
    lambda rows, cols, data, shape: sp.coo_array((data, (rows, cols)), shape=shape, dtype=_CPU_DTYPE)
cpu_coo_to_csr: Callable[[CPUCOOArray], CPUCSRArray] = lambda inp: inp.tocsr()
