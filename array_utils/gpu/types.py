import cupy as cp
import cupyx.scipy.sparse as cps
from typing import Tuple, List, Union


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
