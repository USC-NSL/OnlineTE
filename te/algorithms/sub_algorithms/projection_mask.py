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
TEMP_FOLDER_NAME = 'projection_mask'
MEMMAP_FILE_NAME = 'X_MASK_KE.npy'


def _get_in_out_mask(edge_indexing: Dict[Tuple[int, int], int], graph: nx.DiGraph,
                     commodity_slice: List[Commodity], array_slice: np.ndarray) -> np.ndarray:
    for k, commodity in enumerate(commodity_slice):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        source_in_edges = graph.in_edges(SOURCE)
        destination_out_edges = graph.out_edges(DESTINATION)

        """
        The array is allocated to zero. Thus we first set the entries to -1 and
        then add 1 back to get 0 for places we want to mask and 1 otherwise.
        We do this since there is no `mmap` equivalent of `np.ones`, just zeros.
        """

        for edge in source_in_edges:
            array_slice[edge_indexing[edge], k] = -1
        for edge in destination_out_edges:
            array_slice[edge_indexing[edge], k] = -1
    array_slice += 1


def get_in_out_mask(graph: nx.DiGraph, commodities: List[Commodity]):
    N = len(graph.edges())
    K = len(commodities)
    EDGE_INDEXING = get_edge_indexing(graph)
    slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)

    if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
        MASK = cpu_zeros((N, K))
        _get_in_out_mask(EDGE_INDEXING, graph, commodities, MASK)
        return MASK
    else:
        with contextlib.closing(TempHelper(TEMP_FOLDER_NAME)) as tp:
            # MEMMAP the array to allow for concurrent writing
            output_path = tp.get_file_path(MEMMAP_FILE_NAME)
            MASK = cpu_mmap(output_path, (N, K), 'w+')
            nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
            print(as_info(f'Spawning {nprocs} workers to create in-out mask'))
            Parallel(n_jobs=nprocs)\
                (delayed(_get_in_out_mask)\
                    (EDGE_INDEXING, graph, commodities[begin:end], MASK[:, begin:end])
                    for begin, end in slices)
            del MASK
            return np.load(output_path, allow_pickle=True)
