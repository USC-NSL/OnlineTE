import os
import contextlib
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from typing import Optional, Dict, Tuple, List
from itertools import islice
from te import TE_PATH
from topologies.utils import load_topology
from te.algorithms.array_utils.cpu_utils import cpu_mmap, cpu_bool_zeros, cpu_int_zeros, IntegerCPUArray, BooleanCPUArray
from te.algorithms.sub_algorithms.utils import (get_slice_starts_and_exclusive_ends, get_number_of_required_workers,
                                                NUM_PROCS, TempHelper)
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
    def __init__(self, alpha: BooleanCPUArray, beta: IntegerCPUArray, edge_disjoint: bool = False,
                 name: Optional[str] = None):
        K, N, T = alpha.shape
        assert beta.shape == (K,)
        self._alpha_k = alpha
        self._beta_k = beta
        self._edge_disjoint = edge_disjoint
        self._T: int = T
        self._K: int = K
        self._N: int = N
        self._name = name
    
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
    def alpha(self) -> np.ndarray:
    # def alpha(self) -> COO3D:
        """Path matrix, a boolean `K x N x T` array"""
        assert self._alpha_k is not None
        return self._alpha_k
    
    @property
    def beta(self) -> IntegerCPUArray:
        """Number of available paths for each commodity, an integer `K` vector"""
        assert self._beta_k is not None
        return self._beta_k
    
    @property
    def edge_disjoint(self) -> bool:
        """Whether the path object contanis edge-disjoint shortest paths or not"""
        return self._edge_disjoint

    @property
    def name(self) -> Optional[str]:
        """Name of this path object"""
        return self._name
    
    def set_name(self, name: str):
        assert self._name is None
        self._name = name
    
    @property
    def file_name(self) -> str:
        if self._name is None:
            name = "paths"
        else:
            name = self._name
        return f"{name}.npz" if not self._edge_disjoint else f"{name}_disjoint.npz"

    @staticmethod
    def _get_paths(edge_indexing: Dict[Tuple[int, int], int], 
                   max_paths: int,
                   edge_disjoint: bool,
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
            if not edge_disjoint:
                for t, path in enumerate(islice(nx.shortest_simple_paths(graph, src, dst), max_paths)):
                    for i in range(len(path) - 1):
                        alpha_slice[k, edge_indexing[(path[i], path[i+1])], t] = True
            else:
                for t, path in enumerate(islice(sorted(nx.edge_disjoint_paths(graph, src, dst), key=lambda path: len(path)), max_paths)):
                    for i in range(len(path) - 1):
                        alpha_slice[k, edge_indexing[(path[i], path[i+1])], t] = True
            beta_slice[k] = t+1

    @staticmethod
    def get_expected_file_name_for_topology(topo_name: str, edge_disjoint: bool):
        return f"{topo_name}.npz" if not edge_disjoint else f"{topo_name}_disjoint.npz"

    def get_initial_total_flow(self, demands: np.ndarray) -> np.ndarray:
        """
        Returns the total flow over each edge, when all commodities are
        routed on the first path.
        """
        return np.einsum('ij,i->j', self._alpha_k[:, :, 0], demands)
    
    @classmethod
    def make_from_graph(cls, graph: nx.DiGraph, T: int, edge_disjoint: bool = False):
        """Create a path object from an arbitrary graph"""
        M = graph.number_of_nodes()
        K = M * (M - 1)
        N = graph.number_of_edges()
        EDGE_INDEXING = {edge: e for e, edge in enumerate(graph.edges(data=False))}
        slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
        commodities = []
        for src in range(M):
            for dst in range(M):
                if src == dst:
                    continue
                commodities.append((src, dst))

        if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
            alpha_k = cpu_bool_zeros((K, N, T))
            beta_k = cpu_int_zeros((K,))
            TShortestPaths._get_paths(EDGE_INDEXING, T, edge_disjoint, graph, commodities, alpha_k, beta_k)
        else:
            with contextlib.closing(TempHelper(TEMP_FOLDER_NAME)) as tp:
                alpha_path = tp.get_file_path(MEMMAP_FILE_NAME_ALPHA)
                beta_path = tp.get_file_path(MEMMAP_FILE_NAME_BETA)
                ALPHA_KET = cpu_mmap(alpha_path, (K, N, T), 'w+', bool)
                BETA_K = cpu_mmap(beta_path, (K,), 'w+', np.int32)
                nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
                print(as_info(f'Spawning {nprocs} workers to get path assignments'))
                Parallel(n_jobs=nprocs)\
                    (delayed(TShortestPaths._get_paths)\
                        (EDGE_INDEXING, T, edge_disjoint, graph, commodities[item[0]:item[1]], 
                         ALPHA_KET[item[0]:item[1], :, :], 
                         BETA_K[item[0]:item[1]], index)
                        for index, item in enumerate(slices))
                del ALPHA_KET
                del BETA_K
                alpha_k: BooleanCPUArray = np.load(alpha_path, allow_pickle=False)
                beta_k: IntegerCPUArray = np.load(beta_path, allow_pickle=False)
        return cls(alpha_k, beta_k, edge_disjoint)
    
    @classmethod
    def make_from_topo_name(cls, topo_name: str, T: int, edge_disjoint: bool = False):
        graph, _ = load_topology(topo_name)
        obj = cls.make_from_graph(graph, T, edge_disjoint)
        obj.set_name(topo_name)
        return obj

    def save(self, compressed: bool = False):
        """Save the `alpha` and `beta` arrays in `.npz` format"""
        if not os.path.exists(PATH_FOLDER):
            os.mkdir(PATH_FOLDER)
        save_fn = np.savez if not compressed else np.savez_compressed
        save_fn(
            os.path.join(PATH_FOLDER, self.file_name),
            alpha = self.alpha,
            beta = self.beta,
            edge_disjoint = self.edge_disjoint
        )
    
    @classmethod
    def load_from_file(cls, file_name: str, T: Optional[int] = None):
        if file_name.endswith('.npz'):
            name = file_name[:-4]
        path = os.path.join(PATH_FOLDER, f'{name}.npz')
        try:
            loader = np.load(path)
            alpha = loader['alpha']
            beta = loader['beta']
            edge_disjoint = loader['edge_disjoint']
        except (OSError, FileNotFoundError):
            return None
        _, _, t = np.shape(alpha)
        if T is not None:
            if t < T:
                raise ValueError(as_fail(f'Given path file does not contain enough paths! ({t} < {T})'))
            elif t > T:
                print(as_warning(f'Will only use the first {T} paths (instead of total {t})'))
        else:
            print(as_info(f"File defines {t} paths at most"))
        alpha_k = alpha[:, :, :T]
        beta_k = np.clip(beta, a_min=0, a_max=T)
        return cls(alpha_k, beta_k, edge_disjoint, name)
    
    def scale_down(self, new_T: int):
        """
        Change the maximum number of paths to a smaller value.

        Note
        ----
        This is a very lazy function. It merely changes the view into the
        `alpha` array, it does not reclaim the allocated memory.
        """
        # TODO: Fix the above.
        assert new_T < self.T
        self._alpha_k = self._alpha_k[:, :, :new_T]
        self._beta_k = np.clip(self._beta_k, a_min=0, a_max=new_T)
        self._T = new_T


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


def get_or_make_path_object_for_topology_name(topo_name: str, T: int, edge_disjoint: bool, compress_if_new: bool = False):
    recompute = False
    expected_name = TShortestPaths.get_expected_file_name_for_topology(topo_name, edge_disjoint)
    try:
        obj = TShortestPaths.load_from_file(expected_name, T)
        if obj is None:
            print(as_info(f"Couldn't find a matching file for these parameters. Making path object from scratch."))
            recompute = True
        else:
            if edge_disjoint and not obj.edge_disjoint:
                print(as_warning(f"Current file is not actually edge-disjointed! Will recompute and overwrite."))
                recompute = True
            elif not edge_disjoint and obj.edge_disjoint:
                recompute = True
                print(as_warning(f"Current file is edge-disjointed rather than SPF! Will recompute and overwrite."))
    except ValueError as e:
        print(as_warning(str(e)))
        recompute = True
    finally:
        if recompute:
            obj = TShortestPaths.make_from_topo_name(topo_name, T, edge_disjoint)
            obj.save(compress_if_new)
        else:
            print(as_info(f"Path object {expected_name} was loaded without computation."))
        return obj


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Compute paths for topologies')
    parser.add_argument('topo_name', type=str, help='Topology name (without the postfix of .json, .gml, etc.)')
    parser.add_argument('T', type=int, help='Maximum number of paths per commodity')
    parser.add_argument('--disjoint', action='store_true', help='Use only edge-disjoint paths')
    parser.add_argument('--compress', action='store_true', help='Compress the result in case we compute a new array and need to save it')
    args = parser.parse_args()

    get_or_make_path_object_for_topology_name(args.topo_name, args.T, args.disjoint)
