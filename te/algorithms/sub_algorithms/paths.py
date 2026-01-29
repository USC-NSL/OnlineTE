import os
import pickle
import numpy as np
import networkx as nx
from dataclasses import dataclass
from joblib import Parallel, delayed
from typing import Optional, Dict, Tuple, List
from itertools import islice
from te import TE_PATH
from topologies.utils import load_topology
from numba import njit, prange
from te.algorithms.array_utils.cpu_utils import CPUArray, IntegerCPUArray, BooleanCPUArray, cpu_zeros, cpu_int_zeros, cpu_bool_zeros
from te.algorithms.sub_algorithms.utils import get_slice_starts_and_exclusive_ends, get_number_of_required_workers, NUM_PROCS
from utils.logging import as_info, as_warning, as_fail, ShortTQDMEnumerate


PATH_FOLDER = os.path.join(TE_PATH, "paths")
MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
MAX_NUMBER_OF_WORKERS = min(24, NUM_PROCS)


@dataclass
class PathMask:
    shape: Tuple[int, int, int]
    rows: List[IntegerCPUArray]
    cols: List[IntegerCPUArray]

    def __post_init__(self):
        K = self.shape[0]
        assert len(self.rows) == K
        assert len(self.cols) == K
        # Check the first element at least to be kind of sure ....
        assert self.rows[0].dtype == np.int32
        assert self.cols[0].dtype == np.int32
    
    def reduce_max_path(self, new_T: int):
        for i in range(len(self.rows)):
            mask = self.cols[i] < new_T
            self.cols[i] = self.cols[i][mask]
            self.rows[i] = self.rows[i][mask]
        self.shape = (self.shape[0], self.shape[1], new_T)
    
    def as_array(self, k_start: int = 0, k_end: Optional[int] = None) -> BooleanCPUArray:
        """
        Return the `alpha` matrix as a gigantic Boolean array.
        
        Warning
        -------
        This one *WILL* eat at your memory. Take care when and how it is called.
        """
        _, N, T = self.shape
        if k_end is None:
            K = self.shape[0]
        else:
            assert k_end > k_start
            K = k_end - k_start
        out = cpu_bool_zeros((K, N, T))
        row_slice = self.rows[k_start:k_start + K]
        col_slice = self.cols[k_start:k_start + K]
        for k in range(K):
            row = row_slice[k]
            col = col_slice[k]
            nnz = len(row)
            for i in range(nnz):
                n = row[i]
                t = col[i]
                out[k, n, t] = True
        return out


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
    def __init__(self, alpha: PathMask, beta: IntegerCPUArray, edge_disjoint: bool = False,
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
    def alpha(self) -> PathMask:
        """Path matrix, a boolean `K x N x T` array (although stored as NNZ indices)"""
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
    
    def downscale(self, new_T: int):
        assert new_T < self.T
        self.alpha.reduce_max_path(new_T)
        np.clip(self.beta, a_min=0, a_max=new_T, out=self.beta)
        self._T = new_T
    
    @property
    def file_name(self) -> str:
        if self._name is None:
            name = "paths"
        else:
            name = self._name
        return f"{name}.pkl" if not self._edge_disjoint else f"{name}_disjoint.pkl"

    @staticmethod
    def _get_paths(edge_indexing: Dict[Tuple[int, int], int], 
                   max_paths: int,
                   edge_disjoint: bool,
                   graph: nx.DiGraph,
                   commodity_slice: List[Tuple[int, int]], 
                   index: Optional[int] = 0) -> Tuple[List[IntegerCPUArray], List[IntegerCPUArray], IntegerCPUArray]:
        alpha_rows_slice = []
        alpha_cols_slice = []
        beta_slice = cpu_int_zeros((len(commodity_slice),))
        if index == 0:
            enum = ShortTQDMEnumerate(commodity_slice)
        else:
            enum = enumerate(commodity_slice)
        for k, item in enum:
            src, dst = item
            assert src != dst
            rows = []
            cols = []
            path_enum = enumerate(islice(nx.shortest_simple_paths(graph, src, dst), max_paths)) if not edge_disjoint else \
                        enumerate(islice(sorted(nx.edge_disjoint_paths(graph, src, dst), key=lambda path: len(path)), max_paths))
            for t, path in path_enum:
                for i in range(len(path) - 1):
                    rows.append(edge_indexing[(path[i], path[i+1])])
                    cols.append(t)
            alpha_rows_slice.append(np.array(rows, dtype=np.int32))
            alpha_cols_slice.append(np.array(cols, dtype=np.int32))
            beta_slice[k] = t+1
        return alpha_rows_slice, alpha_cols_slice, beta_slice

    @staticmethod
    def get_expected_file_name_for_topology(topo_name: str, edge_disjoint: bool):
        return f"{topo_name}.pkl" if not edge_disjoint else f"{topo_name}_disjoint.pkl"
    
    @classmethod
    def make_from_graph(cls, graph: nx.DiGraph, T: int, edge_disjoint: bool = False):
        """Create a path object from an arbitrary graph"""
        M = graph.number_of_nodes()
        N = graph.number_of_edges()
        K = M * (M - 1)
        EDGE_INDEXING = {edge: e for e, edge in enumerate(graph.edges(data=False))}
        slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
        commodities = []
        for src in range(M):
            for dst in range(M):
                if src == dst:
                    continue
                commodities.append((src, dst))

        if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
            rows, cols, beta_k = TShortestPaths._get_paths(EDGE_INDEXING, T, edge_disjoint, graph, commodities)
            betas = [beta_k]
        else:
            nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
            print(as_info(f'Spawning {nprocs} workers to get path assignments'))
            ls = Parallel(n_jobs=nprocs)(delayed(TShortestPaths._get_paths)\
                (EDGE_INDEXING, T, edge_disjoint, graph, commodities[item[0]:item[1]], index)
                for index, item in enumerate(slices))
            rows, cols, betas = [], [], []
            for row, col, beta in ls:
                rows.extend(row)
                cols.extend(col)
                betas.extend(beta)
        alpha_k = PathMask((K, N, T), rows, cols)
        beta_k = np.hstack(betas)
        return cls(alpha_k, beta_k, edge_disjoint)
    
    @classmethod
    def make_from_topo_name(cls, topo_name: str, T: int, edge_disjoint: bool = False):
        graph, _ = load_topology(topo_name)
        obj = cls.make_from_graph(graph, T, edge_disjoint)
        obj.set_name(topo_name)
        return obj

    def save(self):
        if not os.path.exists(PATH_FOLDER):
            os.mkdir(PATH_FOLDER)
        with open(os.path.join(PATH_FOLDER, self.file_name), 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_from_file(file_name: str, T: Optional[int] = None):
        if file_name.endswith('.pkl'):
            file_name = file_name[:-4]
        path = os.path.join(PATH_FOLDER, f'{file_name}.pkl')
        try:
            with open(path, 'rb') as f:
                obj: TShortestPaths = pickle.load(f)
        except (OSError, FileNotFoundError):
            return None
        t = obj.T
        if T is not None:
            if t < T:
                raise ValueError(as_fail(f'Given path file does not contain enough paths! ({t} < {T})'))
            elif t > T:
                print(as_warning(f'Will only use the first {T} paths (instead of total {t})'))
                obj.downscale(T)
        else:
            print(as_info(f"File defines {t} paths at most"))
        return obj


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

def get_or_make_path_object_for_topology_name(topo_name: str, T: int, edge_disjoint: bool):
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
            obj.save()
        else:
            print(as_info(f"Path object {expected_name} was loaded without computation."))
        return obj


"""
The following functions implement matrix multiplication with the path mask
array `alpha` in an efficient manner.
Without these, multiplication would be done with `Numpy` which calls generic
`BLAS` backends, and these backends would be extremely inefficient since:
- They would implicitly case `alpha` to a `float` array during multiplication.
- They would not take advantage of `alpha` being sparse.
These implementations can make efficient use of both of these properties. In
particular:
- Knowing `alpha` is Boolean valued, reduces multiplications to a branch statement. These
  branch statements are efficient since miss-predictions are rare because of `alpha` being
  sparse.
- Branch miss-predictions can be removed entirely by just iterating over non-zero elements
  of `alpha`. Since `alpha` is Boolean valued and indices are 32-bit integers at least, doing
  this can increase memory usage as it effectively use 64 bits of data to address 1 bit, but
  on larger topologies, `alpha` is even more sparse (on `Kdl`, less than 1 percent of the
  entries in `alpha` are `True`).
"""


@njit(parallel=True)
def get_initial_total_flow_nnz(rows: List[np.ndarray], beta: np.ndarray, shape: Tuple[int, int, int], D_k: np.ndarray) -> np.ndarray:
    """
    Returns the total flow over each edge, when all commodities are
    routed evenly on all paths.
    """
    K, N, _ = shape
    output = np.zeros((N,), dtype=D_k.dtype)
    for k in prange(K):
        tmp = np.zeros((N,), dtype=D_k.dtype)
        d_val = D_k[k]/beta[k]
        row = rows[k]
        nnz = len(row)
        if nnz == 0:
            continue
        for i in range(nnz):
            n = row[i]
            tmp[n] += d_val
        output += tmp
    return output


@njit(parallel=True)
def path_based_to_edge_based_nnz(Y_tk: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], N: int, D_k: np.ndarray) -> np.ndarray:
    """
    Implements `D_k * alpha_k Y_k` for each `k` by iterating over non-zero entries.
    On larger topologies, this implementation greatly outperforms `path_based_to_edge_based`.
    """
    K = len(rows)
    output = np.zeros((N, K), dtype=Y_tk.dtype)
    
    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        if nnz == 0:
            continue
        for i in range(nnz):
            n = row[i]
            t = col[i]
            output[n, k] += Y_tk[t, k] * d_val 
    return output


@njit(parallel=True)
def path_based_to_edge_based_mean_nnz(Y_tk: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], N: int, D_k: np.ndarray) -> np.ndarray:
    """
    Implements `D_k * alpha_k Y_k` averaged over all `k` by only iterating non-zero entries.
    On larger topologies, this implementation greatly outperforms `path_based_to_edge_based_mean`.
    """
    K = len(rows)
    output = np.zeros((N,), dtype=Y_tk.dtype)

    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        tmp = np.zeros((N,), dtype=Y_tk.dtype)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            tmp[n] += d_val * Y_tk[t, k]
        output += tmp
    return output / K


@njit(parallel=True)
def path_based_projection_nnz(Y_tk: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], N: int, D_k: np.ndarray) -> np.ndarray:
    """
    Implements `D_k^2 * (alpha_k^T alpha_k) Y_k` for each `k` by only iterating non-zero entries.
    """
    T, K = Y_tk.shape
    output = np.zeros((T, K), dtype=Y_tk.dtype)

    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        tmp = np.zeros((N,), dtype=Y_tk.dtype)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            tmp[n] += Y_tk[t, k]
        for i in range(nnz):
            n = row[i]
            t = col[i]
            output[t, k] += tmp[n] * d_val ** 2
    return output


@njit
def path_based_transpose_product_nnz(X_ek: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], T: int, D_k: np.ndarray):
    _, K = X_ek.shape
    output = np.zeros((K, T), dtype=X_ek.dtype)
    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            output[k, t] += X_ek[n, k] * d_val
    return output.T


@njit
def path_based_transpose_vector_product_nnz(X_e: np.ndarray, rows: List[np.ndarray], cols: List[np.ndarray], T: int, D_k: np.ndarray):
    K = D_k.shape[0]
    output = np.zeros((K, T), dtype=X_e.dtype)
    for k in prange(K):
        d_val = D_k[k]
        row = rows[k]
        col = cols[k]
        nnz = len(row)
        for i in range(nnz):
            n = row[i]
            t = col[i]
            output[k, t] += X_e[n] * d_val
    return output.T


@njit
def path_based_eigen_upper_nnz(cols: List[np.ndarray], T: int):
    K = len(cols)
    output = np.zeros((K,), dtype=np.int32)
    for k in prange(K):
        tmp = np.zeros((T,), dtype=np.int32)
        col = cols[k]
        nnz = len(col)
        for i in range(nnz):
            tmp[col[i]] += 1
        output[k] = tmp.max()
    return output


def warm_start_jit(rows: List[np.ndarray], cols: List[np.ndarray], shape: Tuple[int, int, int], beta: IntegerCPUArray):
    K, N, T = shape
    D_k = cpu_zeros((K,))
    X_ek = cpu_zeros((N, K))
    Y_tk = cpu_zeros((T, K))
    path_based_eigen_upper_nnz(cols, T)
    get_initial_total_flow_nnz(rows, beta, shape, D_k)
    path_based_to_edge_based_nnz(Y_tk, rows, cols, N, D_k)
    path_based_to_edge_based_mean_nnz(Y_tk, rows, cols, N, D_k)
    path_based_projection_nnz(Y_tk, rows, cols, N, D_k)
    path_based_transpose_product_nnz(X_ek, rows, cols, T, D_k)
    path_based_transpose_vector_product_nnz(X_ek[:, 0], rows, cols, T, D_k)


"""
The following are older implementations with a dense `alpha`.
At some point, they were used for debugging ...
"""
# @njit(paralel=True)
# def get_initial_total_flow(alpha: np.ndarray, D_k: np.ndarray) -> np.ndarray:
#     """
#     Returns the total flow over each edge, when all commodities are
#     routed evenly on all paths.
#     """
#     K, N, T = alpha.shape
#     output = cpu_zeros((N,))
#     for k in prange(K):
#         d_val = D_k[k]/T
#         for n in prange(N):
#             acc = 0.0
#             for t in range(T):
#                 if alpha[k, n, t]:
#                     acc += d_val
#             output[n] += acc
#     return output

# @njit(parallel=True)
# def path_based_to_edge_based(Y_tk: np.ndarray, alpha: np.ndarray, D_k: np.ndarray) -> np.ndarray:
#     """
#     Efficiently implements `D_k * alpha_k Y_k` for each `k`.
#     This quickly translates path-based assignments to edge-based.
#     """
#     K, N, T = alpha.shape
#     output = np.zeros((N, K), dtype=Y_tk.dtype)
    
#     for k in prange(K):
#         d_val = D_k[k]
#         for n in prange(N):
#             acc = 0.0
#             for t in range(T):
#                 if alpha[k, n, t]:
#                     acc += Y_tk[t, k]
#             output[n, k] = acc * d_val
#     return output

# @njit(parallel=True)
# def path_based_to_edge_based_mean(Y_tk: np.ndarray, alpha: np.ndarray, D_k: np.ndarray) -> np.ndarray:
#     """
#     Efficiently implements `D_k * alpha_k Y_k` averaged over all `k`.
#     This quickly returns the mean edge-based assignment from the path-based assignments.
#     """
#     K, N, T = alpha.shape
#     output = np.zeros((N,), dtype=Y_tk.dtype)

#     for n in prange(N):
#         for k in prange(K):
#             d_val = D_k[k]
#             acc = 0.0
#             for t in range(T):
#                 if alpha[k, n, t]:
#                     acc += Y_tk[t, k]
#             output[n] += acc * d_val
#     return output / K

def path_based_to_edge_based_dense(Y_tk: CPUArray, alpha: BooleanCPUArray, D_k: CPUArray) -> CPUArray:
    """
    Given the path-based assignment `Y_tk` over paths described by the `alpha`
    matrix. Remember that `alpha` is a 3D matrix where the first axis indexes 
    over commodities, and the inner two axis index over edge and path index.
    To produce an edge-based assignment, we would do:

        X_ek = sum_t (alpha_ket Y_tk D_k)
    
    This translates succiently into an Einstein sum.
    """
    return np.einsum('kij,jk,k->ik', alpha, Y_tk, D_k)


def path_based_projection_dense(Y_tk: CPUArray, alpha: BooleanCPUArray, D_k: CPUArray) -> CPUArray:
    """
    Evaluates:

        P_tk = sum_e (alpha_ket X_ek D_k)
    
    Where `X_ek` is the edge based evaluation of the current path set.
    """
    return np.einsum('kji,jk,k->ik', alpha, path_based_to_edge_based_dense(Y_tk, alpha, D_k), D_k)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Compute paths for topologies')
    parser.add_argument('topo_name', type=str, help='Topology name (without the postfix of .json, .gml, etc.)')
    parser.add_argument('T', type=int, help='Maximum number of paths per commodity')
    parser.add_argument('--disjoint', action='store_true', help='Use only edge-disjoint paths')
    args = parser.parse_args()

    get_or_make_path_object_for_topology_name(args.topo_name, args.T, args.disjoint)
