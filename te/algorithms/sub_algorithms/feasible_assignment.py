import os
import shutil
import contextlib
import numpy as np
import networkx as nx
from joblib import Parallel, delayed
from typing import List, Dict, Tuple
from topologies.utils import get_edge_indexing, Commodity
from te.algorithms.utils import as_info, create_temp_folder
from te.algorithms.gpu_utils import cpu_memmap


MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
MAX_NUMBER_OF_WORKERS = 12
TEMP_FOLDER_NAME = 'feasible_assignment'
MEMMAP_FILE_NAME = 'X_KE'


class TempHelper:
    def __init__(self, temp_folder: str):
        self.temp_folder = temp_folder
        self.temp_path = create_temp_folder(temp_folder)
    
    def get_file_path(self, name: str) -> str:
        return os.path.join(self.temp_path, name)

    def close(self):
        try:
            shutil.rmtree(self.temp_path)
        except: # noqa
            pass


get_number_of_required_workers = lambda number_of_commodities: \
    int(min(MAX_NUMBER_OF_WORKERS, np.ceil(number_of_commodities / MAX_NUMBER_OF_COMMODITIES_PER_CORE)))


get_slice_size = lambda number_of_commodities: \
    int(number_of_commodities // get_number_of_required_workers(number_of_commodities))


def get_slice_starts_and_exclusive_ends(number_of_commodities) -> List[int]:
    slice_size = get_slice_size(number_of_commodities)
    number_of_slices = int(np.ceil(number_of_commodities / slice_size))
    return [(slice_size * i, min(slice_size * (i+1), number_of_commodities)) for i in range(number_of_slices)]


def _get_feasible_flow_assignment(edge_indexing: Dict[Tuple[int, int], int], graph: nx.DiGraph,
                                  commodity_slice: List[Commodity], array_slice: np.ndarray) -> np.ndarray:
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
    slices = get_slice_starts_and_exclusive_ends(K)

    if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
        X_KE = np.zeros(shape=(N, K))
        return _get_feasible_flow_assignment(EDGE_INDEXING, graph, commodities, X_KE)
    else:
        with contextlib.closing(TempHelper(TEMP_FOLDER_NAME)) as tp:
            # MEMMAP the array to allow for concurrent writing
            output_path = tp.get_file_path(MEMMAP_FILE_NAME)
            X_KE = cpu_memmap(output_path, (N, K), 'w+')
            nprocs = get_number_of_required_workers(K)
            print(as_info(f'Spawning {nprocs} workers to create initial feasible assignment'))
            Parallel(n_jobs=nprocs)\
                (delayed(_get_feasible_flow_assignment)\
                    (EDGE_INDEXING, graph, commodities[begin:end], X_KE[:, begin:end])
                    for begin, end in slices)
            return X_KE
