import cupy as cp
import numpy as np
import te.constants
from typing import Optional, Tuple
from te.algorithms.utils import careful_norm, careful_norm_squared, all_elements_within_threshold
from te.algorithms.gpu_utils import GPUArray


"""Different Projected Gradient Descent (PGD) algorithms for non-negative constraints"""


# def do_plain_pgd(lambda_k: np.ndarray, x_k_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, c: np.ndarray, 
#                  gamma: float, thresh: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
#     _c = x_k_0 + n @ c
#     for i in range(n_iter):
#         lambda_k_old = lambda_k
#         lambda_k = np.clip(lambda_k - gamma * (nnt @ lambda_k + _c), a_min=0, a_max=None)
#         if careful_norm(lambda_k - lambda_k_old) < thresh:
#             break
#     y_k = c + n.T @ lambda_k
#     return lambda_k, y_k


# def do_pgd_with_exact_line_search(lambda_k: np.ndarray, x_k_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, c: np.ndarray, 
#                                   thresh: float, n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
#     # def get_alpha(current_lambda) -> float:
#     #     t1 = careful_norm_squared(n.T @ x_k_0 + c)
#     #     t2 = careful_norm_squared(n.T @ (current_lambda + x_k_0) + c)
#     #     return np.clip(1 - t1 / t2, a_min=0, a_max=None)

#     big_c = x_k_0 + n @ c
#     big_lambda = nnt @ big_c
#     norm_1 = 0.5 * careful_norm_squared(big_c)
#     norm_2 = careful_norm_squared(n.T @ big_c)

#     def get_alpha(current_lambda) -> Optional[float]:
#         norm = careful_norm_squared(n.T @ current_lambda)
#         dot = np.dot(current_lambda, big_lambda)
#         t1 = norm + 1.5 * dot + norm_1
#         t2 = norm + norm_2 + 2 * dot
#         if t1 < te.constants.MINIMUM_NORM or t2 < te.constants.MINIMUM_NORM:
#             return None
#         return t1 / t2
    
#     i = 0
#     while i < n_iter:
#         lambda_k_old = lambda_k
#         grad = nnt @ lambda_k + big_c
#         alpha = get_alpha(lambda_k_old)
#         if alpha is None:
#             break
#         lambda_k = np.clip(lambda_k_old - alpha * grad, a_min=0, a_max=None)
#         if careful_norm(lambda_k - lambda_k_old) < thresh:
#             break
#         i += 1
#     y_k = c + n.T @ lambda_k
#     return lambda_k, y_k


def do_plain_pgd(lambda_k: np.ndarray, x_k_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, c: np.ndarray, 
                 gamma: float, thresh: Optional[float], n_iter: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    The simplest type of PGD.
    Takes a constant step size, takes a Gradient Descent (GD) step and
    then projects the result back into the feasible set by clipping.
    Receives the input column by column.

    You really shouldn't use this, since there would be too much back and
    forth between the worker and the controller (one RTT per column).
    """
    mod = cp.get_array_module(lambda_k)
    _c = x_k_0 + n @ c
    total_iterations = 0
    for i in range(n_iter):
        lambda_k_old = mod.array(lambda_k)
        lambda_k = mod.clip(lambda_k_old - gamma * (nnt @ lambda_k_old + _c), a_min=0, a_max=None)
        total_iterations += 1
        if thresh and careful_norm(lambda_k - lambda_k_old) < thresh:
            break
    y_k = c + n.T @ lambda_k
    return lambda_k, y_k, total_iterations


# @jit(nopython=True)
def do_iterative_plain_pgd(lambda_block: np.ndarray, x_block_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, c_block: np.ndarray, 
                           gamma: float, thresh: Optional[float], n_iter: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Plain PGD, but receives the input in block (i.e. multiple columns per input).
    Drastically cuts down on the time spend communicating between the node and
    the controller.

    The actual implementation however, still sequentially does PGD on each column.
    """
    mod = cp.get_array_module(lambda_block)
    big_c_block = x_block_0 + n @ c_block
    num_blocks = mod.shape(lambda_block)[-1]
    total_iterations = 0
    for k in range(num_blocks):
        for _ in range(n_iter):
            lambda_k_old = mod.array(lambda_block[:, k])
            lambda_block[:, k] = mod.clip(lambda_k_old - gamma * (nnt @ lambda_k_old + big_c_block[:, k]), a_min=0, a_max=None)
            total_iterations += 1
            if thresh and careful_norm(lambda_block[:, k] - lambda_k_old) < thresh:
                break
    y_block = c_block + n.T @ lambda_block
    return lambda_block, y_block, total_iterations


def do_iterative_pgd_with_exact_line_search(lambda_block: np.ndarray, x_block_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, 
                                            c_block: np.ndarray, thresh: Optional[float], n_iter: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    PGD with exact line search over block input.
    The line search here is done on the unconstrained problem. Can be
    a bit dangerous but works pretty well in practice.
    Like `do_iterative_plain_pgd`, the actual operation is still sequential.
    """
    mod = cp.get_array_module(lambda_block)
    total_iterations = 0
    big_c_block = x_block_0 + n @ c_block
    nnt_big_c_block = nnt @ big_c_block
    norm_1 = 0.5 * careful_norm_squared(big_c_block, axis=0)
    norm_2 = careful_norm_squared(n.T @ big_c_block, axis=0)

    def get_alpha(current_lambda: np.ndarray, k: int) -> Optional[float]:
        norm = careful_norm_squared(n.T @ current_lambda)
        dot = mod.dot(current_lambda, nnt_big_c_block[:, k])
        t1 = norm + 1.5 * dot + norm_1[k]
        t2 = norm + norm_2[k] + 2 * dot
        if thresh and t1 < te.constants.MINIMUM_NORM or t2 < te.constants.MINIMUM_NORM:
            return None
        return t1 / t2

    num_blocks = mod.shape(lambda_block)[-1]
    for k in range(num_blocks):
        i = 0
        while i < n_iter:
            lambda_k_old = mod.array(lambda_block[:, k])
            grad = nnt @ lambda_k_old + big_c_block[:, k]
            alpha = get_alpha(lambda_k_old, k)
            total_iterations += 1
            if alpha is None:
                break
            lambda_block[:, k] = mod.clip(lambda_k_old - alpha * grad, a_min=0, a_max=None)
            if careful_norm(lambda_block[:, k] - lambda_k_old) < thresh:
                break
            i += 1
    y_block = c_block + n.T @ lambda_block
    return lambda_block, y_block, total_iterations


def do_block_plain_pgd(lambda_block: np.ndarray, x_block_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, c_block: np.ndarray, 
                       gamma: float, thresh: Optional[float], n_iter: int, check_conv: bool) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Plain PGD with block operations, meaning that each step is done across the entire
    input matrix. Assuming that the matrix is not too massive, this performs much
    faster than the sequential method.
    """
    mod = cp.get_array_module(lambda_block)
    i = 0
    total_iterations = 0
    big_c_block = x_block_0 + n @ c_block
    number_of_converged_columns = 0
    number_of_commodities = lambda_block.shape[-1]
    commodity_indexes = mod.arange(number_of_commodities)
    converged_lambda_block = mod.zeros_like(lambda_block)
    while number_of_converged_columns < number_of_commodities:
        # Do a block descent
        lambda_block_old = mod.array(lambda_block)
        grad_block = nnt @ lambda_block_old + big_c_block
        lambda_block = mod.clip(lambda_block_old - gamma * grad_block, a_min=0, a_max=None)
        total_iterations += lambda_block.shape[-1]
        # Check which columns seem to have converged
        if thresh and check_conv:
            converged_indices = mod.where(careful_norm(lambda_block - lambda_block_old, axis=0) < thresh)[0]
            if len(converged_indices) > 0:
                number_of_converged_columns += len(converged_indices)
                converged_lambda_block[:, commodity_indexes[converged_indices]] = lambda_block[:, converged_indices]
                lambda_block = mod.delete(lambda_block, converged_indices, axis=1)
                big_c_block = mod.delete(big_c_block, converged_indices, axis=1)
                commodity_indexes = mod.delete(commodity_indexes, converged_indices)
            else:
                # We might have hit the feasible set. If we are not moving, time to quit ...
                if careful_norm(lambda_block - lambda_block_old, scaled=True) < thresh:
                    indices = mod.arange(lambda_block.shape[-1])
                    converged_lambda_block[:, commodity_indexes[indices]] = lambda_block[:, indices]
                    break
        i += 1
        if i == n_iter:
            # Maximum iteration reached. Make do with what we have ...
            if thresh and check_conv:
                indices = mod.arange(lambda_block.shape[-1])
                converged_lambda_block[:, commodity_indexes[indices]] = lambda_block[:, indices]
            else:
                converged_lambda_block = lambda_block
            break
    
    y_block = c_block + n.T @ converged_lambda_block
    return converged_lambda_block, y_block, total_iterations


def do_block_pgd_with_exact_line_search(lambda_block: np.ndarray, x_block_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, 
                                        c_block: np.ndarray, thresh: float, n_iter: int, check_conv: bool) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Same as `do_block_plain_pgd`, but with exact line search.
    """
    mod = cp.get_array_module(lambda_block)
    total_iterations = 0
    big_c_block = x_block_0 + n @ c_block
    nnt_big_c_block = nnt @ big_c_block
    norm_1 = 0.5 * careful_norm_squared(big_c_block, axis=0)
    norm_2 = careful_norm_squared(n.T @ big_c_block, axis=0)

    def get_alpha_block(current_lambda_block: np.ndarray) -> np.ndarray:
        norm = careful_norm_squared(n.T @ current_lambda_block, axis=0)
        dot = mod.diagonal(current_lambda_block.T @ nnt_big_c_block)
        t1 = norm + 1.5 * dot + norm_1
        t2 = norm + norm_2 + 2 * dot
        # if all_elements_within_threshold(t1, te.constants.MINIMUM_NORM) or \
        #     all_elements_within_threshold(t2, te.constants.MINIMUM_NORM):
        #     return mod.zeros_like(t1)
        return t1 / t2
    i = 0
    number_of_converged_columns = 0
    number_of_commodities = lambda_block.shape[-1]
    commodity_indexes = mod.arange(number_of_commodities)
    converged_lambda_block = mod.zeros_like(lambda_block)
    while number_of_converged_columns < number_of_commodities:
        # Do a block descent
        lambda_block_old = mod.array(lambda_block)
        grad_block = nnt @ lambda_block_old + big_c_block
        alpha_block = get_alpha_block(lambda_block_old)
        lambda_block = mod.clip(lambda_block_old - alpha_block * grad_block, a_min=0, a_max=None)
        total_iterations += lambda_block.shape[-1]
        # Check which columns seem to have converged
        # if thresh and check_conv:
        #     converged_indices = mod.where(careful_norm(lambda_block - lambda_block_old, axis=0) < thresh)[0]
        #     if len(converged_indices) > 0:
        #         number_of_converged_columns += len(converged_indices)
        #         converged_lambda_block[:, commodity_indexes[converged_indices]] = lambda_block[:, converged_indices]
        #         lambda_block = mod.delete(lambda_block, converged_indices, axis=1)
        #         big_c_block = mod.delete(big_c_block, converged_indices, axis=1)
        #         nnt_big_c_block = mod.delete(nnt_big_c_block, converged_indices, axis=1)
        #         norm_1 = mod.delete(norm_1, converged_indices)
        #         norm_2 = mod.delete(norm_2, converged_indices)
        #         commodity_indexes = mod.delete(commodity_indexes, converged_indices)
        #     else:
        #         # We might have hit the feasible set. If we are not moving, time to quit ...
        #         if careful_norm(lambda_block - lambda_block_old, scaled=True) < thresh:
        #             indices = mod.arange(lambda_block.shape[-1])
        #             converged_lambda_block[:, commodity_indexes[indices]] = lambda_block[:, indices]
        #             break
        i += 1
        if i == n_iter:
            # Maximum iteration reached. Make do with what we have ...
            if thresh and check_conv:
                indices = mod.arange(lambda_block.shape[-1])
                converged_lambda_block[:, commodity_indexes[indices]] = lambda_block[:, indices]
            else:
                converged_lambda_block = lambda_block
            break
    
    y_block = c_block + n.T @ converged_lambda_block
    return converged_lambda_block, y_block, total_iterations


def do_gpu_plain_pgd_with_step_reduction(lambda_block: GPUArray, x_block_0: GPUArray, nnt: GPUArray, 
                                         n: GPUArray, c_block: GPUArray, gamma: float, n_iter: int, 
                                         kappa: float, epoch: int) -> Tuple[GPUArray, GPUArray]:
    """
    A plain block oriented PGD operation, with step size heuristic.
    This will run a GPU, as such, norm-2 and selective operations (like
    checking for converged columns) become extremely slow.
    Matrix operations on the other hand benefit greatly.
    As such, this implementation is meant to be very small fast.

    The step size heuristic is:

        step_size <- (gamma / epoch**kappa)
    
    For 0 <= `kappa` <= 1.
    """
    mod = cp.get_array_module(lambda_block)
    big_c_block = x_block_0 + n @ c_block
    step_size = gamma / (epoch+1) ** kappa
    for _ in range(n_iter):
        grad_block = nnt @ lambda_block + big_c_block
        lambda_block = mod.clip(lambda_block - step_size * grad_block, a_min=0, a_max=None)
    del big_c_block
    del grad_block
    y_block = c_block + n.T @ lambda_block
    return lambda_block, y_block

import torch
from te.algorithms.tensor_utils import Tensor

def do_tensor_plain_pgd_with_step_reduction(lambda_block: Tensor, x_block_0: Tensor, nnt: Tensor, 
                                            n: Tensor, c_block: Tensor, gamma: float, n_iter: int, 
                                            kappa: float, epoch: int) -> Tuple[Tensor, Tensor]:
    """
    A plain block oriented PGD operation, with step size heuristic.
    This will run a GPU, as such, norm-2 and selective operations (like
    checking for converged columns) become extremely slow.
    Matrix operations on the other hand benefit greatly.
    As such, this implementation is meant to be very small fast.

    The step size heuristic is:

        step_size <- (gamma / epoch**kappa)
    
    For 0 <= `kappa` <= 1.
    """
    big_c_block = x_block_0 + n @ c_block
    step_size = gamma / (epoch+1) ** kappa
    for _ in range(n_iter):
        grad_block = nnt @ lambda_block + big_c_block
        lambda_block = lambda_block - step_size * grad_block
        lambda_block = torch.clamp(lambda_block, min=0)
    del big_c_block
    del grad_block
    y_block = c_block + n.T @ lambda_block
    return lambda_block, y_block