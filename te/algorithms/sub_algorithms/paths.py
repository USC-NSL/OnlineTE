import os
import contextlib
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from typing import Optional, Dict, Tuple, List
from itertools import islice
from te import TE_PATH
from te.algorithms.array_utils.cpu_utils import cpu_mmap
from te.algorithms.base import Commodity, TrafficEngineeringLPEvaluationParams
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from te.algorithms.sub_algorithms.utils import (get_slice_starts_and_exclusive_ends, get_number_of_required_workers,
                                                TempHelper, NUM_PROCS)
from utils.logging import as_info, as_warning, as_fail, ShortTQDMEnumerate


PATH_FOLDER = os.path.join(TE_PATH, "paths")
MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
MAX_NUMBER_OF_WORKERS = min(24, NUM_PROCS)
TEMP_FOLDER_NAME = 'K_shortest_paths'
MEMMAP_FILE_NAME_ALPHA = 'ALPHA_KET.npy'
MEMMAP_FILE_NAME_BETA = 'BETA_KET.npy'


class TShortestPaths:
    """
    Helper class that creates that creates/loads/stores the `T` shortest paths
    between any two nodes in a graph.

    Per our formulation, paths between each pair (i.e. paths for each commodity)
    are kept as some `n x T` matrix, where `n` is the number of edges.
    The path matrix can be stored as a boolean array for compression, and then 
    converted to a floating-point type when it is loaded (we need floating-point
    arithmetics later).

    Many algorithms assume that all paths are equally available, thus we need to also
    signal when paths are completely unavailable (i.e. when less than `T` paths are
    available to a commodity).
    For this, we also keep `beta_k` which is the number of available paths for the
    commodity `k`. Assignments beyond `beta_k` need to pinned to 0.
    """
    def __init__(self, T: int, graph: nx.DiGraph):
        self._T: int = T
        self._K: int = graph.number_of_nodes() * (graph.number_of_nodes() - 1)
        self._N: int = graph.number_of_edges()
        self._graph: nx.DiGraph = graph
        self._alpha_k: Optional[np.ndarray] = None
        self._beta_k: Optional[np.ndarray] = None
    
    @property
    def T(self) -> int:
        """Number of paths for each commodity"""
        return self._T

    @property
    def K(self) -> int:
        """Total number of commodities"""
        return self._K

    @property
    def N(self) -> int:
        """Number of edges in graph"""
        return self._N
    
    @property
    def graph(self) -> nx.DiGraph:
        """The `nx.DiGraph` object of the topology"""
        return self._graph
    
    @property
    def alpha(self) -> np.ndarray:
        """Path matrix, a boolean `K x N x T` array"""
        assert self._alpha_k is not None
        return self._alpha_k
    
    @property
    def beta(self) -> np.ndarray:
        """Number of available paths for each commodity, an integer `K` vector"""
        assert self._beta_k is not None
        return self._beta_k

    @staticmethod
    def _get_paths(edge_indexing: Dict[Tuple[int, int], int], 
                   max_paths: int,
                   graph: nx.DiGraph,
                   commodity_slice: List[Tuple[int, int]], 
                   alpha_slice: np.ndarray, 
                   beta_slice: np.ndarray,
                   index: Optional[int] = 0):
        if index == 0:
            enum = ShortTQDMEnumerate(commodity_slice)
        else:
            enum = enumerate(commodity_slice)
        for k, item in enum:
            src, dst = item
            assert src != dst
            for t, path in enumerate(islice(nx.shortest_simple_paths(graph, src, dst), max_paths)):
                for i in range(len(path) - 1):
                    alpha_slice[k, edge_indexing[(path[i], path[i+1])], t] = True
            beta_slice[k] = t+1
    
    def make(self):
        """Creates the path matrix and sets the `alpha` attribute"""
        K = self.K
        T = self.T
        E = self.N
        EDGE_INDEXING = {edge: e for e, edge in enumerate(self.graph.edges(data=False))}
        slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
        commodities = []
        for src in range(self.graph.number_of_nodes()):
            for dst in range(self.graph.number_of_nodes()):
                if src == dst:
                    continue
                commodities.append((src, dst))

        if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
            alpha_k = np.zeros(dtype=bool, shape=(K, self._N, self._T))
            beta_k = np.zeros(dtype=np.int32, shape=(K,))
            TShortestPaths._get_paths(EDGE_INDEXING, T, graph, commodities, alpha_k, beta_k)
        else:
            with contextlib.closing(TempHelper(TEMP_FOLDER_NAME)) as tp:
                alpha_path = tp.get_file_path(MEMMAP_FILE_NAME_ALPHA)
                beta_path = tp.get_file_path(MEMMAP_FILE_NAME_BETA)
                ALPHA_KET = cpu_mmap(alpha_path, (K, E, T), 'w+', bool)
                BETA_K = cpu_mmap(beta_path, (K,), 'w+', np.int32)
                nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
                print(as_info(f'Spawning {nprocs} workers to get path assignments'))
                Parallel(n_jobs=nprocs)\
                    (delayed(TShortestPaths._get_paths)\
                        (EDGE_INDEXING, T, graph, commodities[item[0]:item[1]], 
                         ALPHA_KET[item[0]:item[1], :, :], 
                         BETA_K[item[0]:item[1]], index)
                        for index, item in enumerate(slices))
                del ALPHA_KET
                del BETA_K
                alpha_k = np.load(alpha_path, allow_pickle=False)
                beta_k = np.load(beta_path, allow_pickle=False)
        
        self._alpha_k = alpha_k
        self._beta_k = beta_k

    def save(self, name: str):
        """Save the `alpha` array in `.npy` format"""
        file_name = f'{name}.npy'
        with open(os.path.join(PATH_FOLDER, file_name), 'wb') as f:
            np.save(f, self.alpha)
            np.save(f, self.beta)

    @classmethod
    def load(cls, graph: nx.DiGraph, T: int, topo_name: Optional[str] = None, save_as: Optional[str] = None):
        # TODO: Maybe check the graph hash to make sure these are the same things?
        if topo_name is not None:
            path = os.path.join(PATH_FOLDER, f'{topo_name}.npy')
            try:
                with open(path, 'rb') as f:
                    alpha = np.load(f, allow_pickle=False)
                    beta = np.load(f, allow_pickle=False)
            except OSError:
                raise ValueError(as_fail(f'No path file for {topo_name} exists!'))
            k, n, t = np.shape(alpha)
            assert (k == (graph.number_of_nodes() * (graph.number_of_nodes() - 1))) and \
                   (n == graph.number_of_edges()), 'Topology size mismatch!'
            if t < T:
                raise ValueError(as_fail(f'Given path file does not contain enough paths! ({t} < {T})'))
            elif t > T:
                print(as_warning(f'Will only use the first {T} paths (instead of total {t})'))
            obj = TShortestPaths(T, graph)
            obj._alpha_k = alpha[:, :, :T]
            obj._beta_k = np.clip(beta, a_min=0, a_max=T)
        else:
            obj = TShortestPaths(T, graph)
            obj.make()
        
        if save_as is not None:
            obj.save(save_as)
        
        return obj


def path_based_to_edge_based(Y_tk: np.ndarray, alpha: np.ndarray, D_k: np.ndarray) -> np.ndarray:
    """
    Given the path-based assignment `Y_tk` over paths described by the `alpha`
    matrix. Remember that `alpha` is a 3D matrix where the first axis indexes 
    over commodities, and the inner two axis index over edge and path index.
    To produce an edge-based assignment, we would do:

        X_ek = sum_t (alpha_ket Y_tk D_k)
    
    This translates succiently into an Einstein sum.
    """
    return np.einsum('kij,jk,k->ik', alpha, Y_tk, D_k)


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


if __name__ == '__main__':
    from topologies.utils import load_zoo_topology
    # topo_name = 'Claranet'
    # topo_name = 'Forthnet'
    topo_name = 'Interoute'
    # topo_name = 'Kdl'

    # Create the paths and save them
    T = 16
    SEED = 12345
    graph = load_zoo_topology(topo_name)
    obj = TShortestPaths(T, graph)
    obj.make()
    obj.save(name=topo_name)

    # Load it again
    obj: TShortestPaths = TShortestPaths.load(graph, T, topo_name)

    # Check if they make sense using a random path assignment
    Y = random_path_assignment(obj.K, obj.T, obj.beta, SEED)
    assert np.allclose(np.sum(Y, axis=0), 1)
    DEMANDS = np.ones(shape=(obj.K,))
    commodity_list = []
    for i in range(graph.number_of_nodes()):
        for j in range(graph.number_of_nodes()):
            if i == j:
                continue
            commodity_list.append(Commodity(i, j, 1.0))
    X = path_based_to_edge_based(Y, obj.alpha, DEMANDS)
    eval_params = TrafficEngineeringLPEvaluationParams(
        TopologyName=topo_name, Seed=SEED, FeasibilityTolerance=1e-8
    )
    check_flow_conservation(X, graph, commodity_list, eval_params)
