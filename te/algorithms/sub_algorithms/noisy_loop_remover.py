import numpy as np
import networkx as nx
from typing import Dict, Tuple
from utils.logging import as_info


def remove_noisy_loops(
        X_EK: np.ndarray, graph: nx.DiGraph, positive_tol: float = 0.0, in_place: bool = True
    ) -> np.ndarray:
    """
    Remove two-edge noisy loops from edge-based assignments.

    For each opposite edge pair (s, t) and (t, s), and for each commodity k:
        f_k = min(X[e1, k], X[e2, k])
    If f_k is positive, subtract it from both entries:
        X[e1, k] -= f_k
        X[e2, k] -= f_k

    This preserves flow conservation while removing opposite-direction residuals.
    This removes many noisy enteries in the final solution that provably serve
    no purpose and can also make the entry more sparse.
    If there are _actual_ transit loops in the final solution (i.e. we didn't
    correctly converge for the original solution), then this really won't help.

    Arguments
    ---------
    X_EK: np.ndarray
        The final edge-based assignment matrix.
    graph: nx.DiGraph
        Topology graph.
    positive_tol: float = 0.0
        Tolerance for considering a value positive.
    in_place: bool = True
        If `True`, modification is done in-place, otherwise a new matrix
        will be returned.
    
    Returns
    -------
    X: np.ndarray
        Processed edge-based assignment, free of noisy loops.
    """
    if len(X_EK.shape) != 2:
        raise ValueError(f'Expected X_EK to be rank-2, got shape {X_EK.shape}')

    N = len(graph.edges())
    if X_EK.shape[0] != N:
        raise ValueError(f'Expected X_EK.shape[0] == |E| ({N}), got {X_EK.shape[0]}')

    X = X_EK if in_place else X_EK.copy()
    edges = list(graph.edges())
    edge_to_index: Dict[Tuple[int, int], int] = {edge: i for i, edge in enumerate(edges)}

    processed = set()
    pair_count = 0
    for e1, (s, t) in enumerate(edges):
        if e1 in processed:
            continue
        opposite = (t, s)
        e2 = edge_to_index.get(opposite, None)
        if e2 is None:
            continue

        processed.add(e1)
        processed.add(e2)
        pair_count += 1

        f = np.minimum(X[e1, :], X[e2, :])
        mask = f > positive_tol
        if np.any(mask):
            X[e1, mask] -= f[mask]
            X[e2, mask] -= f[mask]

    print(as_info(f'Noisy loop remover processed {pair_count} opposite-edge pairs'))
    return X
