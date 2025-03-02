import numpy as np
from typing import Tuple


def dykstra_proj(x_0: np.ndarray, A: np.ndarray, x: np.ndarray, feasibility_tol: float, 
                 max_iter: int = -1) -> Tuple[np.ndarray, bool]:
    """
    Use Dykstra' algorithm to project point `x` onto the nearest point `y` that
    satisfies:

        0 \leq x_0 + A @ y
    """
    assert len(np.shape(x)) == 1 or (len(np.shape(x)) == 2 and np.shape(x)[-1] == 1)

    # Should we even do anything?
    if np.all((x_0 + A @ x) > 0):
        return x, True
    
    number_of_sets = np.shape(A)[0]
    dim = np.shape(x)[0]
    
    # TODO: Need to add the robust stopping criterion
    At = A.T
    increments = np.zeros((dim, number_of_sets))
    iterates = np.zeros((dim, number_of_sets+1))
    iterates[:, -1] = x
    iterates[:, 0] = x
    
    while True:
        max_div = 0
        counter = 0
        for p in range(number_of_sets):
            a = At[:, p]
            column, did_nothing = half_space_proj(x_0[p], a, iterates[:, p-1] + increments[:, p], feasibility_tol)
            if did_nothing:
                continue
            increments[:, p] = (iterates[:, p-1] + increments[:, p]) - column
            div = np.linalg.norm(column - iterates[:, p])
            iterates[:, p] = column
            if div > max_div:
                max_div = div
        counter += 1

        if (max_iter > 0 and counter == max_iter) or (max_div <= feasibility_tol):
            out = iterates[:, -2]
            return out, False


def half_space_proj(x_0: float, a: np.ndarray, x: np.ndarray, feasibility_tol: float) -> Tuple[np.ndarray, bool]:
    """
    Return the orthogonal projection of point `x` onto the half space:

        0 \leq x_0 + a.T @ x
    
    If the condition is already satisfied, return `(x, True)`. If not, then it returns
    the `(projection, False)`
    """

    d = x_0 + np.dot(a, x)
    if d >= -feasibility_tol:
        return x, True
    return x - (d / np.linalg.norm(a)**2) * a, False
