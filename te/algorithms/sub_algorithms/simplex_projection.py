import numpy as np
from typing import Optional
from te.algorithms.array_utils.cpu_utils import cpu_array


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
    theta: float = (cssv[rho] - 1) / cpu_array(rho + 1.0)
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
    theta = (cssv[rho, np.arange(M)] - 1) / cpu_array(rho + 1.0)
    w = np.clip(x - theta, a_min=0, a_max=None)
    return w


if __name__ == '__main__':
    import time
    N, M = 16, 1500
    x = np.random.random(size=(N, M))
    upto = np.random.randint(low=1, high=N, size=(M,))
    x_dir = project_onto_probability_simplex(x, upto)
    start = time.perf_counter()
    x_iter = np.array([column_wise_project_onto_probability_simplex(x[:, i], upto[i]) for i in range(M)]).T
    print(f"Iter took: {(time.perf_counter() - start)*1000} ms")
    start = time.perf_counter()
    x_dir = project_onto_probability_simplex(x, upto)
    print(f"Dir took: {(time.perf_counter() - start)*1000} ms")
    assert np.all(x_iter >= 0) and np.all(x_dir >= 0)
    assert np.allclose(np.sum(x_iter, axis=0), 1) and np.allclose(np.sum(x_dir, axis=0), 1)
    assert np.allclose(x_iter, x_dir)
    for i in range(M):
        assert np.allclose(x_dir[upto[i]:, i], 0)
