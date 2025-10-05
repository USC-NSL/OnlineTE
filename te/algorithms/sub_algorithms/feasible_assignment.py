import contextlib
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from typing import List, Dict, Tuple
from topologies.utils import get_edge_indexing, Commodity
from te.algorithms.utils import as_info
from te.algorithms.array_utils.cpu_utils import cpu_mmap, cpu_zeros
from te.algorithms.sub_algorithms.utils import (get_slice_starts_and_exclusive_ends, get_number_of_required_workers,
                                                TempHelper, NUM_PROCS)


MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
MAX_NUMBER_OF_WORKERS = min(24, NUM_PROCS)
TEMP_FOLDER_NAME = 'feasible_assignment'
MEMMAP_FILE_NAME = 'X_KE.npy'


def _get_feasible_flow_assignment(edge_indexing: Dict[Tuple[int, int], int], graph: nx.DiGraph,
                                  commodity_slice: List[Commodity], array_slice: np.ndarray):
    for k, commodity in enumerate(commodity_slice):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand
        path = nx.shortest_path(graph, SOURCE, DESTINATION)
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            array_slice[edge_indexing[edge], k] = DEMAND


def get_feasible_flow_assignment(graph: nx.DiGraph, commodities: List[Commodity]):
    N = len(graph.edges())
    K = len(commodities)
    EDGE_INDEXING = get_edge_indexing(graph)
    slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)

    if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
        X_KE = cpu_zeros((N, K))
        _get_feasible_flow_assignment(EDGE_INDEXING, graph, commodities, X_KE)
        return X_KE
    else:
        with contextlib.closing(TempHelper(TEMP_FOLDER_NAME)) as tp:
            # MEMMAP the array to allow for concurrent writing
            output_path = tp.get_file_path(MEMMAP_FILE_NAME)
            X_KE = cpu_mmap(output_path, (N, K), 'w+')
            nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
            print(as_info(f'Spawning {nprocs} workers to create initial feasible assignment'))
            Parallel(n_jobs=nprocs)\
                (delayed(_get_feasible_flow_assignment)\
                    (EDGE_INDEXING, graph, commodities[begin:end], X_KE[:, begin:end])
                    for begin, end in slices)
            del X_KE
            return np.load(output_path, allow_pickle=True)
