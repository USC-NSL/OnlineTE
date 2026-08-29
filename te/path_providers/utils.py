import math
import numpy as np
import te.constants
from typing import Optional, List, Tuple


def get_number_of_required_workers(
    number_of_columns: int, 
    max_num_workers: int = te.constants.MAX_NUMBER_OF_SINGLE_HOST_WORKERS, 
    max_column_per_workers: int = te.constants.MAX_NUMBER_OF_COMMODITIES_PER_CORE
) -> int:
    """
    Get number of workers required to handle a 2D matrix with a given number of columns.
    We limit the number of workers to a max value, and also the number of columns that should
    be assigned to each one.
    The latter will be relaxed if the number of columns is too high.
    """
    return min(max_num_workers, math.ceil(number_of_columns / max_column_per_workers))


def get_slice_size(
    number_of_columns: int,
    max_num_workers: int = te.constants.MAX_NUMBER_OF_SINGLE_HOST_WORKERS, 
    max_column_per_workers: int = te.constants.MAX_NUMBER_OF_COMMODITIES_PER_CORE
) -> int:
    """
    Get the appropriate number of columns to chop a 2D matrix with a given
    number of columns into.
    """
    return int(number_of_columns // get_number_of_required_workers(number_of_columns, max_num_workers, max_column_per_workers))


def get_slice_starts_and_exclusive_ends(
    number_of_commodities: int,
    max_num_workers: int = te.constants.MAX_NUMBER_OF_SINGLE_HOST_WORKERS, 
    max_column_per_workers: int = te.constants.MAX_NUMBER_OF_COMMODITIES_PER_CORE
) -> List[Tuple[int, int]]:
    """
    Get (inclusive) begin and (exclusive) end of column-wise slice of an array
    with a given number of columns.
    """
    slice_size = get_slice_size(number_of_commodities, max_num_workers, max_column_per_workers)
    number_of_slices = math.ceil(number_of_commodities / slice_size)
    return [(slice_size * i, min(slice_size * (i+1), number_of_commodities)) for i in range(number_of_slices)]


def get_path_unavailability_mask(beta: np.ndarray, T: int) -> np.ndarray:
    """
    Given `beta_k`, this outputs a mask `mask`, a `T x K` array such that:
        `mask[t, k] = 0` if `t > beta_k[k]`, otherwise it is 1.
    """
    path_indices = np.arange(T)
    return path_indices[:, np.newaxis] < beta[np.newaxis, :]


def random_path_assignment(K: int, T: int, beta: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Completely random assignment for each path.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random
    Y_tk = np.multiply(rng.random(size=(T, K)), get_path_unavailability_mask(beta, T))
    sums = np.sum(Y_tk, axis=0)
    return Y_tk / sums[np.newaxis, :]


