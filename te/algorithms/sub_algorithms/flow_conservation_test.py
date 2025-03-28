import contextlib
import numpy as np
import networkx as nx
from joblib import Parallel, delayed, dump
from typing import Optional, List, Tuple
from collections import defaultdict
from te.traffic_models.base import Commodity
from te.algorithms.array_utils.cpu_utils import cpu_memmap
from te.algorithms.utils import str_round, is_satisfied, is_negligible, as_fail, as_info
from te.algorithms.sub_algorithms.utils import (get_slice_starts_and_exclusive_ends, get_number_of_required_workers,
                                                TempHelper, NUM_PROCS)


MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
MAX_NUMBER_OF_WORKERS = min(24, NUM_PROCS)
TEMP_FOLDER_NAME = 'flow_conservation_check'
MEMMAP_FILE_NAME = 'X_KE'


VIOLATION_OUTFLOW = 0
VIOLATION_LOOP = 1
VIOLATION_LEAK = 2
VIOLATION_INFLOW = 3
VIOLATION_TRANSIT = 4


def _check_centralized_flow_conservation(
        shift: int, flows: np.ndarray, graph: nx.DiGraph, 
        commodities: List[Commodity], feasibility_tol: Optional[float],
        feasibility_ratio: Optional[float] = None
    ) -> List[Tuple[int, int, int, float, float]]:

    violations = []
    print(f'MAX = {np.max(flows)}')
    for k, commodity in enumerate(commodities):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand
        
        flow_out = defaultdict(list)
        flow_in = defaultdict(list)
        for e, edge in enumerate(graph.edges()):
            flow_out[edge[0]].append(flows[e, k])
            flow_in[edge[1]].append(flows[e, k])

        for v in graph.nodes():
            fout = sum(flow_out[v])
            fin  = sum(flow_in[v])

            if (abs(fin) > 1e6):
                print(f'BIG FIN: {fin}')
            if (abs(fout) > 1e6):
                print(f'BIG FOUT: {fout}')

            if v == SOURCE:
                if not is_satisfied(fout, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_OUTFLOW, k+shift, v, fout, DEMAND))
                if not is_negligible(fin, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_LOOP, k+shift, v, fin, 0))
            elif v == DESTINATION:
                if not is_negligible(fout, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_LEAK, k+shift, v, fout, 0))
                if not is_satisfied(fin, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_INFLOW, k+shift, v, fin, DEMAND))
            else:
                if not is_satisfied(fout, fin, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_INFLOW, k+shift, v, fin, fout))
    return violations


def report_violations(violations: List[Tuple[int, int, int, float, float]]):
    for violation_type, k, v, actual, should_be in violations:
        actual_str = str_round(actual, 4)
        should_be_str = str_round(should_be, 4)
        if violation_type == VIOLATION_OUTFLOW:
            print(as_fail(f"Commodity {k}: Node {v} --> Demand outflow does not hold at source: {actual_str} vs {should_be_str}"))
        elif violation_type == VIOLATION_LOOP:
            print(as_fail(f"Commodity {k}: Node {v} --> Source receives its own demand! {actual_str}"))
        elif violation_type == VIOLATION_LEAK:
            print(as_fail(f"Commodity {k}: Node {v} --> Destination is leaking demand! {actual_str}"))
        elif violation_type == VIOLATION_INFLOW:
            print(as_fail(f"Commodity {k}: Node {v} --> Demand inflow does not hold at destination: {actual_str} vs {should_be_str}"))
        elif violation_type == VIOLATION_TRANSIT:
            print(as_fail(f"Commodity {k}: Node {v} --> Transit demand conservation does not hold: {actual_str} --> {should_be_str}"))
        else:
            raise ValueError(f'Unknown violation type: {violation_type}')


def check_flow_conservation(
        flows: np.ndarray, graph: nx.DiGraph, 
        commodities: List[Commodity], feasibility_tol: Optional[float],
        feasibility_ratio: Optional[float] = None
    ):
    """
    (PARALLEL VERSION)
    Check if solution satisfies all of the following constraints:
        - Transit nodes conserve flows                                    ( flow conservation )
        - A demand destined to a node, never flows out from that node     (  no demand leaks  )
        - A demand sourced from a node, never flows back into that node   (      no loops     )
    """
    N = len(graph.edges())
    K = len(commodities)
    slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)

    if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
        violations = _check_centralized_flow_conservation(0, flows, graph, commodities, feasibility_tol, feasibility_ratio)
    else:
        with contextlib.closing(TempHelper(TEMP_FOLDER_NAME)) as tp:
            # MEMMAP the array to allow for concurrent writing
            input_path = tp.get_file_path(MEMMAP_FILE_NAME)
            dump(flows, input_path)
            X_KE = cpu_memmap(input_path, (N, K), 'r')
            nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
            print(as_info(f'Spawning {nprocs} workers to check flow conservation'))
            violations_it = Parallel(n_jobs=nprocs, return_as='generator')\
                (delayed(_check_centralized_flow_conservation)\
                    (begin, X_KE[:, begin:end], graph, commodities[begin:end], feasibility_tol, feasibility_ratio)
                    for begin, end in slices)
            violations = []
            for item in violations_it:
                violations.extend(item)
    # report_violations(violations)
