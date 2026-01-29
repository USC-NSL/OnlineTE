"""
Different Projected Gradient Descent (PGD) algorithms for
handling non-negativity constraints.
"""

import numpy as np
from numba.typed import List as NumbaList
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_cast_float
from te.algorithms.sub_algorithms.simplex_projection import project_onto_probability_simplex, project_onto_probability_orthant
from .paths import path_based_projection_nnz, path_based_transpose_vector_product_nnz


def do_memory_efficient_pgd(lambda_block: CPUArray, x_block: CPUArray, nnt: CPUArray,
                            bias: CPUArray, x_block_0: CPUArray,
                            step_size: float, n_iter: int, mask: CPUArray) -> CPUArray:
    """
    A variant of our plain PGD algorithm that does the bare minimum to handle
    the sharing problem.
    It is more memory efficient, as it gets away with calculating the null-space
    bias (i.e. Y_tk) between iterations.
    """
    for _ in range(n_iter):
        # TODO: This can be optimized slightly if `x_block_0` is a least-squares solution ...
        lambda_block -= step_size * (nnt @ (lambda_block + x_block - np.expand_dims(bias, axis=1)) + x_block_0)
        np.maximum(lambda_block, 0, out=lambda_block, where=~mask)
    return lambda_block


def do_dual_pgd(lambda_block: CPUArray, lambda_sum_block: CPUArray,
                nnt: CPUArray, bias: CPUArray, x_block_0: CPUArray,
                step_size: float, n_iter: int, mask: CPUArray) -> CPUArray:
    """
    A variant of our plain PGD algorithm that only uses dual variables.
    This gets away with one giant matrix multiplication for updating the primal
    solution, which can basically cut the runtime in half.
    On the other hand, it increases memory usage by one-third, since we must keep
    the old dual solution instead of the old primal solution.

    Note
    ----
    The solver that calls this _MUST_ use `PSEUDO_INVERSE` solution for the
    initial feasible assignment.
    """
    for _ in range(n_iter):
        lambda_block -= step_size * (nnt @ (lambda_block + lambda_sum_block - np.expand_dims(bias, axis=1)) + x_block_0)
        np.maximum(lambda_block, 0, out=lambda_block, where=~mask)
    return lambda_block


def do_path_based_pgd(y_block: CPUArray, y_block_old: CPUArray, alpha_rows: NumbaList, alpha_cols: NumbaList,
                      sharing_bias: CPUArray, beta_block: CPUArray, demand_block: CPUArray, num_edges: int, 
                      num_paths: int, step_size: float, n_iter: int) -> CPUArray:
    for _ in range(n_iter):
        grad_block = path_based_projection_nnz(y_block - y_block_old, alpha_rows, alpha_cols, num_edges, demand_block) + \
                     path_based_transpose_vector_product_nnz(sharing_bias, alpha_rows, alpha_cols, num_paths, demand_block)
        y_block = project_onto_probability_simplex(y_block - step_size * grad_block, beta_block)
    return y_block


def do_path_based_maxflow_pgd(y_block: CPUArray, A_block: CPUArray, C_block: CPUArray, D_block: CPUArray,
                              beta_block: CPUArray, step_size: float, n_iter: int, eta: float) -> CPUArray:
    for _ in range(n_iter):
        grad_block = eta * (np.einsum('kij,jk->ik', A_block, y_block) - C_block) - D_block[np.newaxis, :]
        y_block = project_onto_probability_orthant(y_block - step_size * grad_block, beta_block)
    return y_block


try:
    """
    This module might be imported on systems that just do not have a GPU (
    let alone one from Nvidia that can use CUDA).
    Thus, we try to import this but fall-back on plain Numpy if it fails.
    The main thing that we need to monkey-patch is `get_array_module` since
    we use it to write code that works on both CPU and GPU. We handle this by
    just hard-coding it to return Numpy whenever called.
    """
    import cupy as cp
    from te.algorithms.array_utils.gpu_utils import zip_map, PartitionedGPUArray, ScatteredGPUArray

    def do_pgd_gpu(lambda_block: PartitionedGPUArray, c_block: PartitionedGPUArray, nnt: ScatteredGPUArray, 
                step_size: float, n_iter: int, mask: PartitionedGPUArray) -> PartitionedGPUArray:
        """
        A variant of `do_memory_efficient_pgd` suited for running on one or more GPUs.
        """
        for _ in range(n_iter):
            lambda_block = zip_map([nnt, lambda_block, c_block], lambda nn, l, c: l - step_size * (nn @ l + c))
            zip_map([lambda_block, mask], lambda l, m: np.maximum(l, 0, out=l, where=~m))
        return lambda_block

except ModuleNotFoundError:
    do_pgd_gpu = None
