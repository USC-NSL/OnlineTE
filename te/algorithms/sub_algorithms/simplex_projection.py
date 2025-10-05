import numpy as np


"""
Based on https://gist.github.com/daien/1272551
"""

def project_onto_probability_simplex(x: np.ndarray, upto: int):
    N, = x.shape
    assert upto <= N
    # check if we are already on the simplex
    if x.sum() == 1 and np.all(x >= 0) and upto == N:
        # best projection: itself!
        return x
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    # u = np.sort(x, axis=1)[::-1]
    u = np.sort(x)[::-1]
    # cssv = np.cumsum(u, axis=0)
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho: int = np.nonzero(u * np.arange(1, N+1) > (cssv - 1))[0][-1]
    print(rho)
    # compute the Lagrange multiplier associated to the simplex constraint
    theta: float = (cssv[rho] - 1) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (x - theta).clip(min=0)
    return w


if __name__ == '__main__':
    x = (np.arange(10)-5)/10
    # mask = np.hstack([np.ones(dtype=bool, shape=(5,)), np.zeros(dtype=bool, shape=(5,))])
    # print(np.multiply(x, mask))
    # project_onto_probability_simplex(x, 5)
    print(project_onto_probability_simplex(x, 5))