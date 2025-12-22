import numpy as np
from typing import Optional, Tuple
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, cpu_zeros


def soft_thresholding(array: CPUArray, threshold: float) -> CPUArray:
    """
    The Soft-Thresholding operator (also known as the "Shrinkage" operator).
    """
    return np.where(np.abs(array) <= threshold, 0, array - np.sign(array) * threshold)


# _SPARSE_RANGE_LASSO_DUAL: Optional[CPUArray] = None


# def sparse_range_lasso(
#     Y_block: CPUArray, S_block: CPUArray,
#     X_block_0: CPUArray, N: CPUArray, C_block: CPUArray, 
#     gamma: float, epsilon: float, n_iter: int, 
#     mask: CPUArray, NTN_inv: Optional[CPUArray] = None
# ) -> Tuple[CPUArray, CPUArray]:
#     """
#     Trivial ADMM for solving least-squares with L1 norm regularization.
#     """
#     global _SPARSE_RANGE_LASSO_DUAL
#     if _SPARSE_RANGE_LASSO_DUAL is None:
#         _SPARSE_RANGE_LASSO_DUAL = cpu_zeros(S_block.shape)

#     for _ in range(n_iter):
#         if NTN_inv is not None:
#             Y_block = NTN_inv @ (C_block - gamma * N.T @ (X_block_0 + _SPARSE_RANGE_LASSO_DUAL - S_block))
#         else:
#             Y_block = (C_block - gamma * N.T @ (X_block_0 + _SPARSE_RANGE_LASSO_DUAL - S_block)) / (1 + gamma)
#         S_hat = X_block_0 + N @ Y_block
#         S_block = soft_thresholding(S_hat + _SPARSE_RANGE_LASSO_DUAL, epsilon / gamma)
#         S_block = np.clip(S_block - np.multiply(S_block, mask), a_min=0, a_max=None)
#         _SPARSE_RANGE_LASSO_DUAL += (S_hat - S_block)
#     del S_hat
#     return S_block, Y_block


def sparse_range_lasso(
    X_block: CPUArray, Z_block: CPUArray, L_block: CPUArray,
    X_block_0: CPUArray, NNT: CPUArray, C_block: CPUArray, 
    gamma: float, epsilon: float, n_iter: int, mask: BooleanCPUArray
) -> Tuple[CPUArray, CPUArray, CPUArray]:
    for _ in range(n_iter):
        X_block = X_block_0 + NNT @ ((C_block + gamma * (Z_block - L_block)) / (1 + gamma) - X_block_0)
        Z_block = soft_thresholding(X_block + L_block, epsilon / gamma)
        Z_block = np.clip(Z_block - np.multiply(Z_block, mask), a_min=0, a_max=None)
        L_block += (X_block - Z_block)
    return X_block, Z_block, L_block
