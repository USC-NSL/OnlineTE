"""
The following functions implement matrix multiplication with the path mask
array `alpha` in an efficient manner.
Without these, multiplication would be done with `Numpy` which calls generic
`BLAS` backends, and these backends would be extremely inefficient since:
- They would implicitly case `alpha` to a `float` array during multiplication.
- They would not take advantage of `alpha` being sparse.
These implementations can make efficient use of both of these properties. In
particular:
- Knowing `alpha` is Boolean valued, reduces multiplications to a branch statement. These
  branch statements are efficient since miss-predictions are rare because of `alpha` being
  sparse.
- Branch miss-predictions can be removed entirely by just iterating over non-zero elements
  of `alpha`. Since `alpha` is Boolean valued and indices are 32-bit integers at least, doing
  this can increase memory usage as it effectively use 64 bits of data to address 1 bit, but
  on larger topologies, `alpha` is even more sparse (on `Kdl`, less than 1 percent of the
  entries in `alpha` are `True`).
"""


import numpy as np
from numba import njit, prange
from typing import Optional, List, Tuple
from array_utils.cpu.types import *


@njit(parallel=True)
def get_initial_total_flow_nnz(rows: List[np.ndarray], beta: np.ndarray, shape: Tuple[int, int, int], D_k: np.ndarray,
                               C_e: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns the total flow over each edge, when all commodities are
    routed evenly on all paths.
    """
    K, N, _ = shape
    is_capped = C_e is not None
    output = np.zeros((N,), dtype=D_k.dtype)
    for k in prange(K):
        tmp = np.zeros((N,), dtype=D_k.dtype)
        d_val = D_k[k]/beta[k]
        row = rows[k]
        nnz = len(row)
        if nnz == 0:
            continue
        for i in range(nnz):
            n = row[i]
            if not is_capped:
                tmp[n] += d_val
            else:
                tmp[n] += d_val / C_e[n]
        output += tmp
    return output


@njit(parallel=True)
def path_based_to_edge_based_nnz(Y_tk: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], N: int, D_k: np.ndarray, 
                                 C_e: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Implements `D_k * alpha_k Y_k` for each `k` by iterating over non-zero entries.
    On larger topologies, this implementation greatly outperforms `path_based_to_edge_based`.
    If `C_e` is given, `D_k / C_e` will be used when needed.
    """
    K = len(rows)
    is_capped = C_e is not None
    output = np.zeros((N, K), dtype=Y_tk.dtype)
    
    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        if nnz == 0:
            continue
        for i in range(nnz):
            n = row[i]
            t = col[i]
            if not is_capped:
                output[n, k] += Y_tk[t, k] * d_val 
            else:
                output[n, k] += Y_tk[t, k] * d_val / C_e[n]
    return output


@njit(parallel=True)
def path_based_to_edge_based_mean_nnz(Y_tk: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], N: int, D_k: np.ndarray,
                                      C_e: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Implements `D_k * alpha_k Y_k` averaged over all `k` by only iterating non-zero entries.
    On larger topologies, this implementation greatly outperforms `path_based_to_edge_based_mean`.
    """
    K = len(rows)
    is_capped = C_e is not None
    output = np.zeros((N,), dtype=Y_tk.dtype)

    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        tmp = np.zeros((N,), dtype=Y_tk.dtype)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            if not is_capped:
                tmp[n] += d_val * Y_tk[t, k]
            else:
                tmp[n] += d_val * Y_tk[t, k] / C_e[n]
        output += tmp
    return output / K


@njit(parallel=True)
def path_based_projection_nnz(Y_tk: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], N: int, D_k: np.ndarray,
                              C_e: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Implements `D_k^2 * (alpha_k^T alpha_k) Y_k` for each `k` by only iterating non-zero entries.
    """
    T, K = Y_tk.shape
    is_capped = C_e is not None
    output = np.zeros((T, K), dtype=Y_tk.dtype)

    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        tmp = np.zeros((N,), dtype=Y_tk.dtype)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            tmp[n] += Y_tk[t, k]
        for i in range(nnz):
            n = row[i]
            t = col[i]
            if not is_capped:
                output[t, k] += tmp[n] * d_val ** 2
            else:
                output[t, k] += tmp[n] * (d_val / C_e[n]) ** 2
    return output


@njit
def path_based_transpose_product_nnz(X_ek: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], T: int, D_k: np.ndarray,
                                     C_e: Optional[np.ndarray] = None):
    _, K = X_ek.shape
    is_capped = C_e is not None
    output = np.zeros((K, T), dtype=X_ek.dtype)
    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            if not is_capped:
                output[k, t] += X_ek[n, k] * d_val
            else:
                output[k, t] += X_ek[n, k] * d_val / C_e[n]
    return output.T


@njit
def path_based_transpose_vector_product_nnz(X_e: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], T: int, D_k: np.ndarray,
                                            C_e: Optional[np.ndarray] = None):
    K = D_k.shape[0]
    is_capped = C_e is not None
    output = np.zeros((K, T), dtype=X_e.dtype)
    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            if not is_capped:
                output[k, t] += X_e[n] * d_val
            else:
                output[k, t] += X_e[n] * d_val / C_e[n]
    return output.T


@njit
def path_based_eigen_upper_nnz(cols: List[np.ndarray], T: int):
    K = len(cols)
    output = np.zeros((K,), dtype=np.int32)
    for k in prange(K):
        tmp = np.zeros((T,), dtype=np.int32)
        col = cols[k]
        nnz = len(col)
        for i in range(nnz):
            tmp[col[i]] += 1
        output[k] = tmp.max()
    return output


def path_based_power_method(rows: List[np.ndarray], cols: List[np.ndarray], shape: Tuple[int, int, int],
                            C_e: Optional[np.ndarray] = None, n_iters: int = 20) -> np.ndarray:
    K, N, T = shape
    d = cpu_zeros((K,)) + 1
    # TODO: Don't be lazy ... this should start from a random vector
    v = cpu_zeros((T, K)) + 1
    for _ in range(n_iters):
        res = path_based_projection_nnz(v, rows, cols, N, d, C_e)
        v = res / cpu_array(np.linalg.norm(v, axis=0))[None, :]
    res = path_based_projection_nnz(v, rows, cols, N, d, C_e)
    return cpu_array(np.sum(np.multiply(res, v), axis=0) / np.sum(np.multiply(v, v), axis=0))


def warm_start_jit(rows: List[np.ndarray], cols: List[np.ndarray], shape: Tuple[int, int, int], beta: IntegerCPUArray):
    K, N, T = shape
    C_e = cpu_zeros((N,)) + 1
    D_k = cpu_zeros((K,))
    X_ek = cpu_zeros((N, K))
    Y_tk = cpu_zeros((T, K))
    path_based_eigen_upper_nnz(cols, T)
    get_initial_total_flow_nnz(rows, beta, shape, D_k, C_e)
    path_based_to_edge_based_nnz(Y_tk, rows, cols, N, D_k, C_e)
    path_based_to_edge_based_mean_nnz(Y_tk, rows, cols, N, D_k, C_e)
    path_based_projection_nnz(Y_tk, rows, cols, N, D_k, C_e)
    path_based_transpose_product_nnz(X_ek, rows, cols, T, D_k, C_e)
    path_based_transpose_vector_product_nnz(X_ek[:, 0], rows, cols, T, D_k, C_e)


"""
The following are older implementations with a dense `alpha`.
At some point, they were used for debugging ...
"""
# @njit(paralel=True)
# def get_initial_total_flow(alpha: np.ndarray, D_k: np.ndarray) -> np.ndarray:
#     """
#     Returns the total flow over each edge, when all commodities are
#     routed evenly on all paths.
#     """
#     K, N, T = alpha.shape
#     output = cpu_zeros((N,))
#     for k in prange(K):
#         d_val = D_k[k]/T
#         for n in prange(N):
#             acc = 0.0
#             for t in range(T):
#                 if alpha[k, n, t]:
#                     acc += d_val
#             output[n] += acc
#     return output

# @njit(parallel=True)
# def path_based_to_edge_based(Y_tk: np.ndarray, alpha: np.ndarray, D_k: np.ndarray) -> np.ndarray:
#     """
#     Efficiently implements `D_k * alpha_k Y_k` for each `k`.
#     This quickly translates path-based assignments to edge-based.
#     """
#     K, N, T = alpha.shape
#     output = np.zeros((N, K), dtype=Y_tk.dtype)
    
#     for k in prange(K):
#         d_val = D_k[k]
#         for n in prange(N):
#             acc = 0.0
#             for t in range(T):
#                 if alpha[k, n, t]:
#                     acc += Y_tk[t, k]
#             output[n, k] = acc * d_val
#     return output

# @njit(parallel=True)
# def path_based_to_edge_based_mean(Y_tk: np.ndarray, alpha: np.ndarray, D_k: np.ndarray) -> np.ndarray:
#     """
#     Efficiently implements `D_k * alpha_k Y_k` averaged over all `k`.
#     This quickly returns the mean edge-based assignment from the path-based assignments.
#     """
#     K, N, T = alpha.shape
#     output = np.zeros((N,), dtype=Y_tk.dtype)

#     for n in prange(N):
#         for k in prange(K):
#             d_val = D_k[k]
#             acc = 0.0
#             for t in range(T):
#                 if alpha[k, n, t]:
#                     acc += Y_tk[t, k]
#             output[n] += acc * d_val
#     return output / K

def path_based_to_edge_based_dense(Y_tk: CPUArray, alpha: BooleanCPUArray, D_k: CPUArray) -> CPUArray:
    """
    Given the path-based assignment `Y_tk` over paths described by the `alpha`
    matrix. Remember that `alpha` is a 3D matrix where the first axis indexes 
    over commodities, and the inner two axis index over edge and path index.
    To produce an edge-based assignment, we would do:

        X_ek = sum_t (alpha_ket Y_tk D_k)
    
    This translates succiently into an Einstein sum.
    """
    return np.einsum('kij,jk,k->ik', alpha, Y_tk, D_k)


def path_based_projection_dense(Y_tk: CPUArray, alpha: BooleanCPUArray, D_k: CPUArray) -> CPUArray:
    """
    Evaluates:

        P_tk = sum_e (alpha_ket X_ek D_k)
    
    Where `X_ek` is the edge based evaluation of the current path set.
    """
    return np.einsum('kji,jk,k->ik', alpha, path_based_to_edge_based_dense(Y_tk, alpha, D_k), D_k)