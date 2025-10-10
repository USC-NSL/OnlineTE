import numpy as np

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
    from te.algorithms.array_utils.gpu_utils import zip_map
except ModuleNotFoundError:
    import numpy as cp
    def zip_map(*args, **kwargs):
        raise ValueError('CUDA/CuPY is not available. You should never reach here!')

import te.constants
from typing import Optional
from te.algorithms.utils import careful_norm, careful_norm_squared, all_elements_within_threshold
from te.algorithms.sub_algorithms.simplex_projection import column_wise_projection_onto_probability_simplex


"""Different Projected Gradient Descent (PGD) algorithms for non-negative constraints"""


def do_plain_pgd(lambda_k, x_k_0, nnt, n, c, gamma: float, thresh: Optional[float], n_iter: int):
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


def do_iterative_plain_pgd(lambda_block, x_block_0, nnt, n, c_block, gamma: float, thresh: Optional[float], n_iter: int):
    """
    Plain PGD, but receives the input in block (i.e. multiple columns per input).
    Drastically cuts down on the time spent communicating between the node and
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


def do_iterative_pgd_with_exact_line_search(lambda_block, x_block_0, nnt, n, c_block, thresh: Optional[float], n_iter: int):
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


def do_block_plain_pgd(lambda_block, x_block_0, nnt, n, c_block, gamma: float, thresh: Optional[float], 
                       n_iter: int, check_conv: bool):
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


def do_block_pgd_with_exact_line_search(lambda_block, x_block_0, nnt, n, c_block, thresh: float, 
                                        n_iter: int, check_conv: bool):
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
        if all_elements_within_threshold(t1, te.constants.MINIMUM_NORM, mod) or \
            all_elements_within_threshold(t2, te.constants.MINIMUM_NORM, mod):
            return mod.zeros_like(t1)
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


def do_plain_pgd_with_step_reduction(lambda_block, x_block_0, nnt, n, c_block, gamma: float, n_iter: int, 
                                     kappa: float, epoch: int, mask = None):
    """
    A plain block oriented PGD operation, with step size heuristic.
    The step size heuristic is:

        step_size <- (gamma / epoch**kappa)
    
    For 0 <= `kappa` <= 1.
    """
    mod = cp.get_array_module(lambda_block)
    big_c_block = x_block_0 + n @ c_block
    step_size = gamma / ((epoch+1) ** kappa)
    for _ in range(n_iter):
        grad_block = nnt @ lambda_block + big_c_block
        # mod.clip(lambda_block - step_size * grad_block, a_min=0, a_max=None, out=lambda_block)
        assert mask is not None
        lambda_block = lambda_block - step_size * grad_block
        correction = mod.clip(mod.multiply(lambda_block, mask), a_min=None, a_max=0)
        lambda_block = mod.clip(lambda_block, a_min=0, a_max=None) + correction
    del big_c_block
    del grad_block
    y_block = c_block + n.T @ lambda_block
    return lambda_block, y_block


def do_gpu_pgd_with_exact_line_search(lambda_block, x_block_0, nnt, n, c_block, n_iter: int):
    def get_step(d, grad_block):
        term1 = cp.sum(cp.multiply(d, grad_block), axis=0)
        term2 = cp.einsum('...i,...i->...', cp.dot(d.T, nnt), d.T) + cp.ones_like(term1) * cp.float16(1e-3)
        return term1 / term2
    
    big_c_block = x_block_0 + n @ c_block
    for _ in range(n_iter):
        grad_block = nnt @ lambda_block + big_c_block
        d = lambda_block - cp.clip(lambda_block - grad_block, a_min=0, a_max=None)
        lambda_block = cp.clip(lambda_block - get_step(d, grad_block) * grad_block, a_min=0, a_max=None)
    del big_c_block
    del grad_block
    y_block = c_block + n.T @ lambda_block
    return lambda_block, y_block


def do_multi_gpu_plain_pgd_with_step_reduction(lambda_block, x_block_0, nnt, n, c_block, gamma: float, 
                                               n_iter: int, kappa: float, epoch: int):
    """
    A plain block oriented PGD operation, with step size heuristic.
    This will run on a GPU, as such, norm-2 and selective operations (like
    checking for converged columns) become extremely slow.
    Matrix operations on the other hand benefit greatly.

    The step size heuristic is:

        step_size <- (gamma / epoch**kappa)
    
    For 0 <= `kappa` <= 1.
    """
    big_c_block = zip_map([x_block_0, n, c_block], lambda x0, _n, c: x0 + _n @ c)
    step_size = gamma / ((epoch+1) ** kappa)
    for _ in range(n_iter):
        grad_block = zip_map([nnt, lambda_block, big_c_block], lambda nn, l, c: nn @ l + c)
        lambda_block = zip_map([lambda_block, grad_block], lambda l, g: cp.clip(l - step_size * g, a_min=0, a_max=None))
    del big_c_block
    del grad_block
    y_block = zip_map([c_block, n, lambda_block], lambda c, _n, l: c + _n.T @ l)
    return lambda_block, y_block


def do_pgd_with_backtracking(lambda_block, x_block_0, nnt, n, c_block, n_iter: int, beta: float, max_back: int):
    mod = cp.get_array_module(lambda_block)
    big_c_block = x_block_0 + n @ c_block
    num_cols = mod.shape(lambda_block)[1]
    
    def get_f(current_lambda):
        current_lambda_t = current_lambda.T
        return 0.5 * mod.sum((current_lambda_t @ nnt) * current_lambda_t, axis=1) + \
            mod.sum(current_lambda * big_c_block, axis=0)
    
    def get_generalized_grad(current_lambda, step_sizes, grad_block):
        return (current_lambda - mod.clip(current_lambda - step_sizes * grad_block, a_min=0, a_max=None)) / step_sizes
    
    def get_step_sizes(current_lambda, grad_block):
        step_sizes = mod.ones(shape=(num_cols,), dtype=current_lambda.dtype)
        for _ in range(max_back):
            current_generalized_grad = get_generalized_grad(current_lambda, step_sizes, grad_block)
            generalized_f = get_f(current_lambda - step_sizes * current_generalized_grad)
            interpolated_f = get_f(current_lambda) + \
                             step_sizes * mod.linalg.norm(current_generalized_grad, axis=0)**2 / 2 - \
                             step_sizes * mod.sum(current_generalized_grad * grad_block, axis=0)
            step_sizes = step_sizes * (1 - beta * (generalized_f > interpolated_f))
        return step_sizes
    
    for _ in range(n_iter):
        grad_block = nnt @ lambda_block + big_c_block
        step_sizes = get_step_sizes(lambda_block, grad_block)
        mod.clip(lambda_block - step_sizes * grad_block, a_min=0, a_max=None, out=lambda_block)
    y_block = c_block + n.T @ lambda_block
    return lambda_block, y_block


def do_plain_path_based_pgd_with_step_reduction(y_block, scaled_alpha_block, c_block, beta_block,
                                                gamma: float, n_iter: int, 
                                                kappa: float, epoch: int):
    """
    See `do_pgd_with_step_reduction` for the step heuristics.
    This function is specifically for the path-based problem, where we try to
    minimize:

        || alpha_k Y_k - C_k ||_2^2
    
    Such that `Y_k` remains in the pinned probability simplex (pinned meaning that
    `Y_tk` values for `t` indices beyond `beta_k` MUST be zero).

    This operations is slightly more complex than the edge-based version. First, note
    that the matrix shapes are:
    - `y_block`: `T x K`
    - `scaled_alpha_block`: `K x n x T`
    - `c_block`: `n x K`
    - `beta_block`: Vector of length `K`

    Thus, the objective is:

        sum_e (sum_t (alpha_ket Y_tk) - C_ek)^2
    
    And the derivative for `Y_tk` is:

        sum_e sum_t' (alpha_ket alpha_ket' Y_t'k) - 
              sum_e  (C_ek alpha_ket)
    
    This is easiest to express as two Einstein sums. Let `t -> i`, `k -> j`,
    `e -> k` and `t' -> h`; We get:

        sum_k sum_h (alpha_jki alpha_jkh Y_hj) - 
              sum_k (C_kj alpha_jki)
    """
    step_size = gamma / ((epoch+1) ** kappa)
    for _ in range(n_iter):
        sum_1 = np.einsum('jki,jkh,hj->ij', scaled_alpha_block, scaled_alpha_block, y_block)
        sum_2 = np.einsum('kj,jki->ij', c_block, scaled_alpha_block)
        grad_block = sum_1 - sum_2
        y_block = column_wise_projection_onto_probability_simplex(
            y_block - step_size * grad_block, beta_block
        )
    return y_block
