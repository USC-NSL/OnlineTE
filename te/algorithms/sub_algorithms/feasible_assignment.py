import networkx as nx
from joblib import Parallel, delayed
from typing import List, Dict, Tuple, Union
from topologies.utils import get_edge_indexing, Commodity
from te.algorithms.utils import as_info
from te.algorithms.array_utils.cpu_utils import cpu_coo_array, cpu_coo_to_csr, cpu_coo_to_csc, CPUCSRArray, CPUCSCArray, CPUArray
from te.algorithms.sub_algorithms.utils import (get_slice_starts_and_exclusive_ends, get_number_of_required_workers,
                                                NUM_PROCS)


MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
MAX_NUMBER_OF_WORKERS = min(24, NUM_PROCS)


def _get_feasible_flow_assignment(edge_indexing: Dict[Tuple[int, int], int], graph: nx.DiGraph,
                                  commodity_slice: List[Commodity], shift: int) -> Tuple[List[int], List[int], List[float]]:
    rows, cols, data = [], [], []
    for k, commodity in enumerate(commodity_slice):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand
        path = nx.shortest_path(graph, SOURCE, DESTINATION)
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            rows.append(edge_indexing[edge])
            cols.append(k + shift)
            data.append(DEMAND)
    return rows, cols, data


def get_feasible_flow_assignment(
    graph: nx.DiGraph,
    commodities: List[Commodity],
    csc: bool = False
) -> Union[CPUCSRArray, CPUCSCArray, CPUArray]:
    """
    Create an assignment that satisfies all demands by routing everything on
    the first shortest path between source and destination.

    This is usually a pretty sparse matrix, and as such we return CSC/CSR by
    default.

    Note
    ----
    `scipy.sparse` currently, does not support `float16` data types.
    However, many times we do benefit from using `float16` when running
    things on a GPU.
    Currently, we do not wish to port this to Pytorch, and as such we just
    take the memory hit and return a dense array when `float16` is used.
    """
    N = len(graph.edges())
    K = len(commodities)
    EDGE_INDEXING = get_edge_indexing(graph)
    slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)

    if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
        rows, cols, data = _get_feasible_flow_assignment(EDGE_INDEXING, graph, commodities, 0)
    else:
        nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
        print(as_info(f'Spawning {nprocs} workers to create initial feasible assignment'))
        ls = Parallel(n_jobs=nprocs)\
            (delayed(_get_feasible_flow_assignment)\
                (EDGE_INDEXING, graph, commodities[begin:end], begin)
                for begin, end in slices)
        rows = []
        cols = []
        data = []
        for _row, _col, _data in ls:
            rows.extend(_row)
            cols.extend(_col)
            data.extend(_data)

    if csc:
        return cpu_coo_to_csc(cpu_coo_array(rows, cols, data, (N, K)))
    return cpu_coo_to_csr(cpu_coo_array(rows, cols, data, (N, K)))
