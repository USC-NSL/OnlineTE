import numpy as np
from .. import get_global_precision
from .sparse.types import *
from typing import Callable, Tuple, Any, Union


CPUArray = np.ndarray
"""Alias for `numpy.ndarray`, an array that lives on the RAM. Plenty of space usually."""
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

cpu_zeros: Callable[[Tuple[int]], CPUArray] = lambda shape: np.zeros(shape=shape, dtype=get_global_precision())
"""Wrapper for `np.zeros`. Enforces the global data type"""
cpu_fill: Callable[[Tuple[int], Any], CPUArray] = lambda shape, fill: np.full(shape=shape, fill_value=fill, dtype=get_global_precision())
"""Wrapper for `np.full`. Enforces the global data type"""


def is_float_array(thing: CPUArray) -> bool:
    """
    Return `True`, if the array contains float-16/32/64 data type.
    We implicitly assume that any such array needs to be cast to the 
    current global float precision.
    """
    dt = thing.dtype
    return dt == np.float16 or dt == np.float32 or dt == np.float64


cpu_cast_float: Callable[[Any], Any] = lambda val: get_global_precision()(val)
"""Cast a float value to the current global CPU data type"""


def cpu_array(thing: Any) -> Union[CPUArray, CPUCOOArray, CPUCSRArray]:
    """
    Create a copy of an array-like thing with similar sparsity and
    enforce the current global data type.
    """
    if isinstance(thing, CPUArray):
        if is_float_array(thing):
            return np.array(thing, dtype=get_global_precision())
        return thing.copy()
    elif isinstance(thing, (list, tuple)):
        return np.array(thing, dtype=get_global_precision())
    elif isinstance(thing, (CPUCOOArray, CPUCSRArray)):
        return thing.copy()
    raise ValueError(f'Unknown matrix type: {type(thing)}')


__all__ = [
    'CPUArray', 'DoublePrecisionCPUArray', 'IntegerCPUArray', 'BooleanCPUArray',
    # These are not types and should technically be in `wrapper`, but are used too
    # frequently, so we put them here
    'cpu_array', 'cpu_zeros', 'cpu_fill', 'cpu_cast_float'
]