import numpy as np
import torch
from typing import Optional, Tuple, Callable

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
Tensor = torch.Tensor
"""Alias for `torch.Tensor`, an array that lives on the GPU memory. Usually quite limited."""

"""
For very large topologies, GPU memory becomes very tight.
We need to switch to half precision for such cases.
"""

DOUBLE_PRECISION = 'double'  # float 64
SINGLE_PRECISION = 'single'  # float 32
HALF_PRECISION = 'half'      # float 16

NUMBER_OF_GPU_DEVICES: int = torch.cuda.device_count()


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
        _GPU_DTYPE = torch.float64
    elif precision == SINGLE_PRECISION:
        _CPU_DTYPE = np.float32
        _GPU_DTYPE = torch.float32
    elif precision == HALF_PRECISION:
        _CPU_DTYPE = np.float16
        _GPU_DTYPE = torch.float16
    else:
        raise ValueError


def get_total_reserved_gpu_memory_usage() -> int:
    """The amount of allocated GPU memory usage across all devices (including caches ...)"""
    raise NotImplementedError


def get_total_used_gpu_memory_usage() -> int:
    """The GPU working set, across all devices"""
    raise NotImplementedError


cpu_zeros: Callable[[Tuple[int]], CPUArray]  = lambda shape: np.zeros(shape=shape, dtype=_CPU_DTYPE)
"""Wrapper for `nump.zeros`. Enforces the global data type."""
gpu_zeros: Callable[[Tuple[int]], Tensor]  = lambda shape: torch.zeros(*shape, dtype=_GPU_DTYPE, requires_grad=False, device=torch.device(0))
"""Wrapper for `cupy.zeros`. Enforces the global data type."""
as_gpu_array: Callable[[CPUArray], Tensor] = lambda array: torch.from_numpy(array).type(dtype=_GPU_DTYPE).cuda(device=0)
"""Move array into GPU memory."""
as_cpu_array: Callable[[Tensor], CPUArray] = lambda array: array.detach().numpy(force=True)
"""Move array into CPU memory."""
cpu_memmap: Callable[[str, Tuple[int], str], CPUArray] = \
    lambda path, shape, mode: np.memmap(shape=shape, filename=path, mode=mode, dtype=_CPU_DTYPE)


synchronize_to_device: Callable[[int], None] = lambda dev: torch.cuda.synchronize(device=dev)
"""Wait until all operations on the slected device finish"""
def synchronize_to_all():
    """Wait until all operations on all GPU devices finish"""
    for dev in range(NUMBER_OF_GPU_DEVICES):
        synchronize_to_device(dev)
