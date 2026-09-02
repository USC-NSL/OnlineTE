from __future__ import annotations

import pickle
import numpy as np
import networkx as nx
from dataclasses import dataclass
from joblib import Parallel, delayed
from typing import Optional, Dict, Tuple, List, Callable, Iterator
from itertools import islice, repeat, pairwise
from te.traffic_models.base import commodity_od_iterator
from array_utils.cpu.types import *
from array_utils.cpu.wrapper import *
from .utils import get_slice_starts_and_exclusive_ends, get_number_of_required_workers
from utils.logging import as_info, ShortTQDMEnumerate


PerCommodityProvider = Callable[[nx.DiGraph, int, int], Iterator[List[int]]]
"""
A function that receives a graph and a source and destination ID
and returns an iterator that gives hop-by-hop paths between the
source and destination nodes.
"""


def hopbyhop_to_indexed_path(hopbyhop: List[int], edge_indexing: Dict[Tuple[int, int], int]) -> List[int]:
    """
    Convert a hop-by-hop path into a edge-indexed path.

    Example
    -------
    Take a chain graph `2 -> 3 -> 4` with edges indexed 0 and 1.
    This takes `[2, 3, 4]`, internally looks at edge-by-edge form
    which is `[(2, 3), (3, 4)]` and then outputs `[0, 1]`.
    """
    return [edge_indexing[edge] for edge in pairwise(hopbyhop)]


def indexed_path_to_hopbyhop(indexed: List[int], edges: List[Tuple[int, int]]) -> List[int]:
    """Convert indexed path into a hopbyhop one (mostly for debugging)"""
    return [edges[index][1] for index in indexed]


@dataclass
class PathProvider:
    """
    Implements the path tensor and availability in a sparse format.
    The tensor `alpha_{ket}` is defined to be one when for commodity
    index `k`, the `t`-th path crosses edge index `e` and is zero
    otherwise.
    It is Boolean-valued and very sparse, so we store it in COO format
    here.
    The main parameter is how many paths (`T`) we want to store.

    Note
    ----
    This format is simple and general. It can represent any selection of
    paths and can be fed directly to our path-based algorithms.
    """
    shape: Tuple[int, int, int]
    rows: List[IntegerCPUArray]
    cols: List[IntegerCPUArray]
    beta: IntegerCPUArray

    def __post_init__(self):
        K = self.shape[0]
        assert len(self.rows) == K
        assert len(self.cols) == K
        assert len(self.beta) == K and self.beta.ndim == 1
        # Check the first element at least to be kind of sure ....
        assert self.rows[0].dtype == np.int32
        assert self.cols[0].dtype == np.int32

    @property
    def T(self) -> int:
        return self.shape[-1]
    
    def reduce_max_path(self, new_T: int):
        """Shrink the mask in case we have too many paths"""
        assert new_T < self.T
        for i in range(len(self.rows)):
            mask = self.cols[i] < new_T
            self.cols[i] = self.cols[i][mask]
            self.rows[i] = self.rows[i][mask]
        self.shape = (self.shape[0], self.shape[1], new_T)
        np.clip(self.beta, a_min=0, a_max=new_T, out=self.beta)

    def drop_path(self, k: int, t: int):
        """Make path index `t` in-accessible to commodity `k`"""
        assert self.beta[k] > 1
        cols = self.cols[k]
        rows = self.rows[k]
        mask = (cols != t)
        cols = cols[mask]
        rows = rows[mask]
        self.beta[k] -= 1

    def update_path(self, k: int, t: int, path: List[int]):
        """Changes path index `t` for commodity `k` to be `path`"""
        cols = self.cols[k]
        rows = self.rows[k]
        mask = (cols != t)
        cols = cols[mask]
        rows = rows[mask]
        new_cols = cpu_int_fill((len(path),), t)
        new_rows = cpu_int_array(path)
        cols = np.hstack([cols, new_cols])
        rows = np.hstack([rows, new_rows])

    def add_path(self, k: int, path: List[int]):
        """
        Add a new path to commodity `k`.

        Note
        ----
        - Currently, this doesn't handle going above `T` paths and will
          raise an error if it detects that.
        - We _DO NOT_ check if this is a duplicate path. If it is a duplicate
          path, you will probably destroy most of our algorithms!
        """
        assert self.beta[k] < self.T
        cols = self.cols[k]
        rows = self.rows[k]
        new_cols = cpu_int_fill((len(path),), self.beta[k]+1)
        new_rows = cpu_int_array(path)
        cols = np.hstack([cols, new_cols])
        rows = np.hstack([rows, new_rows])
        self.beta[k] += 1
    
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

    def save(self, path: str):
        """Pickle this object and save it in `path`"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> PathProvider:
        """Try to load a stored object from `path`"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def _get_paths(
    edge_indexing: Dict[Tuple[int, int], int], 
    max_paths: int,
    graph: nx.DiGraph,
    commodity_slice: List[Tuple[int, int]], 
    per_commodity_provider: PerCommodityProvider,
    index: Optional[int] = 0
) -> Tuple[List[IntegerCPUArray], List[IntegerCPUArray], IntegerCPUArray]:
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
        hopbyhop_path_enum = enumerate(islice(per_commodity_provider(graph, src, dst), max_paths))
        for t, hopbyhop_path in hopbyhop_path_enum:
            indexed_path = hopbyhop_to_indexed_path(hopbyhop_path, edge_indexing)
            rows.extend(indexed_path)
            cols.extend(repeat(t, len(indexed_path)))
        alpha_rows_slice.append(np.array(rows, dtype=np.int32))
        alpha_cols_slice.append(np.array(cols, dtype=np.int32))
        beta_slice[k] = t+1
    return alpha_rows_slice, alpha_cols_slice, beta_slice


def build_provider( 
    T: int,
    graph: nx.DiGraph,
    per_commodity_provider: PerCommodityProvider,
    edge_indexing: Dict[Tuple[int, int], int]
) -> PathProvider:
    """
    Build a provider based on the given per-commodity provider.
    This is usually quite slow, so may have to be done in parallel.
    """
    M = graph.number_of_nodes()
    N = graph.number_of_edges()
    K = M * (M - 1)
    slices = get_slice_starts_and_exclusive_ends(K)
    commodities = list(commodity_od_iterator(M))
    nprocs = get_number_of_required_workers(K)

    if nprocs == 1:
        rows, cols, beta_k = _get_paths(
            edge_indexing=edge_indexing,
            max_paths=T,
            graph=graph,
            commodity_slice=commodities,
            per_commodity_provider=per_commodity_provider
        )
        betas = [beta_k]
    else:
        print(as_info(f'Spawning {nprocs} workers to build path provider'))
        ls = Parallel(n_jobs=nprocs)(delayed(_get_paths)\
            (edge_indexing, T, graph, commodities[item[0]:item[1]], 
             per_commodity_provider, index)
            for index, item in enumerate(slices))
        rows, cols, betas = [], [], []
        for row, col, beta in ls:
            rows.extend(row)
            cols.extend(col)
            betas.extend(beta)
    
    return PathProvider(
        shape=(K, N, T),
        rows=rows,
        cols=cols,
        beta=np.hstack(betas)
    )


__all__ = ['PathProvider', 'build_provider']