import numpy as np


"""
Based on https://gist.github.com/daien/1272551
"""

def project_onto_probability_simplex(x: np.ndarray, upto: int) -> np.ndarray:
    """
    Given a vector `x`, project it onto the probability simplex such that
    indices beyond `upto` are pinned to zero.
    """
    N, = x.shape
    if x.sum() == 1 and np.all(x >= 0) and upto == N:
        return x
    u = np.sort(x[:upto])[::-1]
    cssv = np.cumsum(u)
    rho: int = np.nonzero(u * np.arange(1, upto+1) > (cssv - 1))[0][-1]
    theta: float = (cssv[rho] - 1) / (rho + 1.0)
    w = (x - theta).clip(min=0)
    w[upto:] = 0
    return w


def column_wise_projection_onto_probability_simplex(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    """
    Similar to `project_onto_probability_simplex`, but receives a 2D matrix `x` 
    and projects individual columns. `upto` is now a vector, its length must agree 
    with the number of columns on `x`.

    TODO: Do not be fooled by the fancy Numpy function call, it is still a trivial
          loop over the columns, and hence becomes really slow when `x` has many
          columns (see https://github.com/USC-NSL/DistributedTE/issues/26).
    """
    _, K = x.shape
    assert upto.ndim == 1 and len(upto) == K
    out = np.empty_like(x)
    for k in range(K):
        out[:, k] = project_onto_probability_simplex(x[:, k], upto[k])
    return out


if __name__ == '__main__':
    x1 = (np.arange(10)-5)/10
    x2 = (np.arange(5)-5)/10
    print(project_onto_probability_simplex(x1, 5))
    print(project_onto_probability_simplex(x2, 5))
    x3 = ((np.arange(20) - 5)/10).reshape((2, 10)).T
    print(column_wise_projection_onto_probability_simplex(x3, np.array([5, 10])))
