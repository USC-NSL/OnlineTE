import numpy as np
from typing import Tuple
from array_utils.cpu.types import *


def soft_thresholding(array: CPUArray, threshold: float) -> CPUArray:
    """
    The Soft-Thresholding operator (also known as the "Shrinkage" operator).
    """
    return np.where(np.abs(array) <= threshold, 0, array - np.sign(array) * threshold)


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
