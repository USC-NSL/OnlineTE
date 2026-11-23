"""
Different Projected Gradient Descent (PGD) algorithms for
handling non-negativity constraints.
"""

import numpy as np
from typing import Union, Optional
from numba.typed import List as NumbaList
from array_utils.cpu.types import *
from te.algorithms.sub_algorithms.simplex_projection import (
    project_onto_probability_simplex, project_onto_probability_orthant
)
from te.path_providers.sparse_ops import (
    path_based_projection_nnz, path_based_transpose_vector_product_nnz
)


def do_memory_efficient_pgd(
    lambda_block: CPUArray, x_block: CPUArray, nnt: CPUArray,
    bias: CPUArray, x_block_0: CPUArray,
    step_size: float, mask: CPUArray, *, max_iter: int = 1000,
    optimality_tolerance: float = 1e-5, feasibility_tolerance: float = 1e-5
) -> CPUArray:
    """Accelerate the dense dual with a mean gap and primal feasibility check.

    Keep the incoming assignment and bias fixed throughout this subproblem.
    Reuse its primal reconstruction as the gradient so accumulated roundoff in
    ``P @ X + X_0 == X`` cannot make the update disagree with the stopping test.
    """
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
        raise ValueError('PGD feasibility_tolerance must be positive and finite')
    t = 1
    momentum = 0
    z_block = np.copy(lambda_block)
    previous_gradient = None
    for iteration in range(max_iter + 1):
        candidate = x_block + nnt @ (lambda_block - bias[:, None])
        min_assignment = float(np.min(candidate))
        gap = abs(float(np.sum(lambda_block * candidate, dtype=np.float64))) / x_block.shape[1]
        if gap <= optimality_tolerance and min_assignment >= -feasibility_tolerance:
            return lambda_block
        if iteration == max_iter:
            return lambda_block
        momentum_gradient = candidate if previous_gradient is None else \
            candidate + momentum * (candidate - previous_gradient)
        next_lambda = np.maximum(0, z_block - step_size * momentum_gradient)
        t_acc = cpu_cast_float(0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)))
        momentum = cpu_cast_float((t - 1.0) / t_acc)
        z_block = next_lambda + momentum * (next_lambda - lambda_block)
        previous_gradient = candidate
        lambda_block[:] = next_lambda
        t = t_acc


def do_dual_pgd(
    lambda_block: CPUArray, lambda_sum_block: CPUArray,
    nnt: CPUArray, bias: CPUArray, x_block_0: CPUArray,
    step_size: float, n_iter: int, mask: CPUArray
) -> CPUArray:
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
        lambda_block -= step_size * (
            nnt @ (lambda_block + lambda_sum_block - np.expand_dims(bias, axis=1)) +
            x_block_0
        )
        np.maximum(lambda_block, 0, out=lambda_block, where=~mask)
    return lambda_block


def do_path_based_pgd(
    y_block: CPUArray, y_block_old: CPUArray, alpha_rows: NumbaList, alpha_cols: NumbaList,
    sharing_bias: CPUArray, beta_block: CPUArray, demand_block: CPUArray, num_edges: int, 
    num_paths: int, step_sizes: Union[float, np.ndarray], n_iter: int,
    capacities: Optional[CPUArray] = None
) -> CPUArray:
    for _ in range(n_iter):
        grad_block = \
            path_based_projection_nnz(
                y_block - y_block_old, alpha_rows, alpha_cols,
                num_edges, demand_block, capacities
            ) + \
            path_based_transpose_vector_product_nnz(
                sharing_bias, alpha_rows, alpha_cols,
                num_paths, demand_block, capacities
            )
        y_block = project_onto_probability_simplex(y_block - step_sizes * grad_block, beta_block)
    return y_block


def do_path_based_nesterov_pgd(
    y_block: CPUArray, y_block_old: CPUArray,
    alpha_rows: NumbaList, alpha_cols: NumbaList, sharing_bias: CPUArray,
    beta_block: CPUArray, demand_block: CPUArray, num_edges: int, 
    num_paths: int, step_sizes: Union[float, np.ndarray], n_iter: int,
    capacities: Optional[CPUArray] = None
) -> CPUArray:
    t = 1
    z_block = np.copy(y_block)
    for _ in range(n_iter):
        grad_block = \
            path_based_projection_nnz(
                z_block - y_block_old, alpha_rows, alpha_cols,
                num_edges, demand_block, capacities
            ) + \
            path_based_transpose_vector_product_nnz(
                sharing_bias, alpha_rows, alpha_cols,
                num_paths, demand_block, capacities
            )
        candidate = project_onto_probability_simplex(z_block - step_sizes * grad_block, beta_block)
        t_acc = cpu_cast_float(0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)))
        z_block = candidate + cpu_cast_float((t - 1.0)/t_acc) * (candidate - y_block)
        y_block = candidate
        t = t_acc

    return y_block


def do_path_based_maxflow_pgd(
    y_block: CPUArray, y_block_old: CPUArray,
    alpha_rows: NumbaList, alpha_cols: NumbaList, sharing_bias: CPUArray,
    beta_block: CPUArray, demand_block: CPUArray, num_edges: int, 
    num_paths: int, step_sizes: Union[float, np.ndarray], n_iter: int, eta: float,
    capacities: Optional[CPUArray] = None
) -> CPUArray:
    t = 1
    z_block = np.copy(y_block)
    for _ in range(n_iter):
        grad_block = \
            path_based_projection_nnz(
                z_block - y_block_old, alpha_rows, alpha_cols,
                num_edges, demand_block, capacities
            ) + \
            path_based_transpose_vector_product_nnz(
                sharing_bias, alpha_rows, alpha_cols,
                num_paths, demand_block, capacities
            ) - \
            demand_block / eta
        candidate = project_onto_probability_orthant(z_block - step_sizes * grad_block, beta_block)
        t_acc = cpu_cast_float(0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)))
        z_block = candidate + cpu_cast_float((t - 1.0)/t_acc) * (candidate - y_block)
        y_block = candidate
        t = t_acc

    return y_block
