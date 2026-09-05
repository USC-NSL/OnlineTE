import numpy as np
from typing import Optional
from array_utils.cpu.types import cpu_cast_float


def column_wise_project_onto_probability_simplex(x: np.ndarray, upto: Optional[int] = None) -> np.ndarray:
    """
    Given a vector `x`, project it onto the probability simplex.
    Optionally accepts the argument `upto`. Any index above or equal to `upto`
    on the vector will be pinned to zero.

    Source
    ------
    See https://gist.github.com/daien/1272551

    Citations
    ----
    - Wang, Weiran, and Miguel A. Carreira-Perpinán. 
      "Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application." 
      arXiv preprint arXiv:1309.1541 (2013).
    """
    N, = x.shape
    if upto is None:
        upto = N
    if x.sum() == 1 and np.all(x >= 0) and upto == N:
        return x
    u = np.sort(x[:upto])[::-1]
    cssv = np.cumsum(u)
    rho: int = np.nonzero(u * np.arange(1, upto+1) > (cssv - 1))[0][-1]
    theta: float = (cssv[rho] - 1) / cpu_cast_float(rho + 1.0)
    w = np.clip(x - theta, a_min=0, a_max=None)
    w[upto:] = 0
    return w


def project_onto_probability_simplex(x: np.ndarray, pinned: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Given a 2D array `x`, project each column onto the probability simplex.
    Optionally, we can accept a vector `pinned`, with a length that matches
    the number of columns in `x`.
    For each column `i`, the projection will be done such that entries beyond
    `x[i]` are pinned to zero.
    """
    N, M = x.shape
    if pinned is not None:
        mask = np.arange(N)[:, np.newaxis] >= pinned
        x[mask] = -np.inf
    u = np.sort(x, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0)
    mask = u * np.arange(1, N+1)[:, np.newaxis] > (cssv - 1)
    rho = np.sum(mask, axis=0) - 1
    theta = (cssv[rho, np.arange(M)] - 1) / cpu_cast_float(rho + 1.0)
    w = np.clip(x - theta, a_min=0, a_max=None)
    return w


def project_onto_probability_orthant(x: np.ndarray, pinned: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project onto the probability orthant (i.e. the volume of the probability simplex).
    Will leave the point untouched if it is already inside the simplex, or pushes it onto
    the surface if it is not.
    """
    N = x.shape[0]
    if pinned is not None:
        mask = np.arange(N)[:, np.newaxis] >= pinned
        x[mask] = 0
    x = np.clip(x, a_min=0, a_max=None, out=x)
    columns_to_project = (x.sum(axis=0) > 1)
    if not np.any(columns_to_project):
        return x
    x[:, columns_to_project] = project_onto_probability_simplex(
        x[:, columns_to_project], 
        None if pinned is None else pinned[columns_to_project]
    )
    return x


# A witness implementation known to be correct, taken from 
# https://gist.github.com/mblondel/6f3b7aaad90606b98f71
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


if __name__ == '__main__':
    import time
    from array_utils.cpu import *
    set_cpu_float_precision(SINGLE_PRECISION)
    N, M = 16, 5000
    x = np.random.random(size=(N, M))
    upto = np.random.randint(low=1, high=N, size=(M,))
    x_dir = project_onto_probability_simplex(x, upto)
    start = time.perf_counter()
    x_iter = np.array([column_wise_project_onto_probability_simplex(x[:, i], upto[i]) for i in range(M)]).T
    print(f"Iter took: {(time.perf_counter() - start)*1000} ms")
    x_iter_check = np.array([np.pad(column_wise_project_onto_probability_simplex(x[:upto[i], i]), (0, N - upto[i]), mode='constant', constant_values=0) for i in range(M)]).T
    x_iter_witness = np.array([np.pad(projection_simplex_sort(x[:upto[i], i]), (0, N - upto[i]), mode='constant', constant_values=0) for i in range(M)]).T
    assert np.allclose(x_iter - x_iter_check, 0)
    assert np.allclose(x_iter_witness - x_iter_check, 0)
    start = time.perf_counter()
    x_dir = project_onto_probability_simplex(x, upto)
    print(f"Dir took: {(time.perf_counter() - start)*1000} ms")
    assert np.all(x_iter >= 0) and np.all(x_dir >= 0)
    assert np.allclose(np.sum(x_iter, axis=0), 1) and np.allclose(np.sum(x_dir, axis=0), 1)
    assert np.allclose(x_iter, x_dir)
    for i in range(M):
        assert np.allclose(x_dir[upto[i]:, i], 0)
    x_in = np.random.random(size=(N, M))
    x_in = x_in / np.sum(x_in, axis=0)
    x_in_res = project_onto_probability_orthant(x_in, upto)
    x_test = project_onto_probability_orthant(x, upto)
    assert np.allclose(x_in, x_in_res)
