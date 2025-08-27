import math
import contextlib
import numpy as np
import te.constants
import networkx as nx
from itertools import groupby
from collections import Counter
from joblib import Parallel, delayed
from typing import Optional, List, Tuple, NewType, Set
from collections import defaultdict
from te.traffic_models.base import Commodity
from te.algorithms.array_utils.cpu_utils import cpu_mmap, cpu_dump
from te.algorithms.base import TrafficEngineeringLPEvaluationParams
from te.algorithms.utils import str_round
from utils.logging import as_fail, as_warning, as_info, as_success
from te.algorithms.sub_algorithms.utils import (get_slice_starts_and_exclusive_ends, get_number_of_required_workers,
                                                TempHelper, NUM_PROCS)


ViolationType = NewType('ViolationType', int)
Violation = Tuple[ViolationType, int, int, float, float]
ViolationSeverity = NewType('ViolationSeverity', int)


MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
MAX_NUMBER_OF_WORKERS = min(24, NUM_PROCS)
TEMP_FOLDER_NAME = 'flow_conservation_check'
MEMMAP_FILE_NAME = 'X_KE.npy'


NEGLIGIBLE_DEMAND_ABS_TOL = 1e-3
SEVERE_VIOLATION_REL_TOL = 5e-2

VIOLATION_OUTFLOW = ViolationType(0)
"""The output of source node is not its demand"""
VIOLATION_LOOP = ViolationType(1)
"""The source node is receiving its own demand"""
VIOLATION_LEAK = ViolationType(2)
"""The destination node does not completely consume the demand"""
VIOLATION_INFLOW = ViolationType(3)
"""The input of the destination node is not its demand"""
VIOLATION_TRANSIT = ViolationType(4)
"""Flow conservation does not hold on a transit node"""
LIST_OF_VIOLATION_TYPES = [VIOLATION_OUTFLOW, VIOLATION_LOOP, VIOLATION_LEAK, VIOLATION_INFLOW, VIOLATION_TRANSIT]


LEVEL_NEGLIGIBLE = ViolationSeverity(0)
LEVEL_MILD = ViolationSeverity(1)
LEVEL_SEVERE = ViolationSeverity(2)


def is_satisfied(optim, actual, feasibility_tol: Optional[float], feasibility_ratio: Optional[float]):
    """
    Check if `actual` is close to `optim` assignment.
    The test can either use absolute or relative tolerance (if both are present, only
    absolute tolerance is considered).
    """
    if feasibility_tol is not None:
        return math.isclose(optim, actual, abs_tol=feasibility_tol)
    if abs(optim) < te.constants.FLOAT_RES:
        return math.isclose(actual, 0, abs_tol=te.constants.FLOAT_RES)
    return math.isclose(optim, actual, rel_tol=feasibility_ratio)


def is_negligible(actual, baseline, feasibility_tol: Optional[float], feasibility_ratio: Optional[float]):
    """
    If `feasibility_tol` is given, the it checks if absolute value of `actual` is
    within `min(feasibility_tol, baseline)`.
    If `feasibility_ratio` is present, it checks if the absolute value is within
    `baseline * feasibility_ratio` tolerance.
    """
    if abs(actual) < te.constants.FLOAT_RES:
        return True
    if feasibility_tol is not None:
        return abs(actual) < min(baseline, feasibility_tol)
    return abs(actual) < abs(baseline * feasibility_ratio)


def get_violation_severity(violation: Violation) -> ViolationSeverity:
    violation_type, _, _, actual, demand = violation
    if abs(demand) < NEGLIGIBLE_DEMAND_ABS_TOL:
        """
        The commodity has negligible demand by default.
        Thus, the violation should also be negligible. Anything above that we consider
        a severe violation (i.e. there is no `mild` case).
        """
        if abs(actual) < NEGLIGIBLE_DEMAND_ABS_TOL:
            return LEVEL_NEGLIGIBLE
        return LEVEL_SEVERE   
    
    if violation_type in {VIOLATION_INFLOW, VIOLATION_OUTFLOW}:
        """
        The commodity has non-negligible demand, but inflow/outflow
        for destination/source are not zero.
        Relative error will be checked here.
        """
        if abs(actual) < abs(demand * SEVERE_VIOLATION_REL_TOL):
            return LEVEL_MILD
        return LEVEL_SEVERE
    elif violation_type == VIOLATION_LEAK:
        """
        The commodity has non-negligible demand and destination seems
        to not consume all of the traffic completely.
        Relative error will be checked here.
        """
        if abs(actual) < abs(demand * SEVERE_VIOLATION_REL_TOL):
            return LEVEL_MILD
        return LEVEL_SEVERE
    elif violation_type == VIOLATION_LOOP:
        """
        The commodity has non-negligible demand and source seems
        to receive some of its own demand.
        Loops can be devistating, thus this needs to be negligible.
        """
        if abs(actual) < NEGLIGIBLE_DEMAND_ABS_TOL:
            return LEVEL_NEGLIGIBLE
        return LEVEL_SEVERE  
    elif violation_type == VIOLATION_TRANSIT:
        """
        The commodity has non-negligible demand and we are considering
        transit node flow conservation.
        Relative error here is enough. If it starts to build up, then
        we will catch it when looking at the source/destination nodes.
        """
        if is_satisfied(actual, demand, feasibility_ratio=SEVERE_VIOLATION_REL_TOL):
            return LEVEL_MILD
        return LEVEL_SEVERE
    else:
        raise ValueError(f'Unknown violation type: {violation_type}')


def show_violation_severity(violations: List[Violation], number_of_commodities: int, number_of_nodes: int):
    level_key = lambda v: get_violation_severity(v)
    type_key = lambda v: v[0]

    levels = {level: [] for level in (LEVEL_NEGLIGIBLE, LEVEL_MILD, LEVEL_SEVERE)}
    for level, violations in groupby(sorted(violations, key=level_key), key=level_key):
        levels[level] = list(violations)
    neg = levels[LEVEL_NEGLIGIBLE]
    mid = levels[LEVEL_MILD]
    sev = levels[LEVEL_SEVERE]

    constraint_numbers = {
        VIOLATION_OUTFLOW: number_of_commodities,
        VIOLATION_INFLOW: number_of_commodities,
        VIOLATION_LEAK: number_of_commodities,
        VIOLATION_LOOP: number_of_commodities,
        VIOLATION_TRANSIT: number_of_commodities * (number_of_nodes - 2)
    }
    constraint_names = {
        VIOLATION_OUTFLOW: 'Source Outflow',
        VIOLATION_INFLOW: 'Destination Inflow',
        VIOLATION_LEAK: 'Destination Consume',
        VIOLATION_LOOP: 'Source Produce',
        VIOLATION_TRANSIT: 'Transit Conservation'
    }

    if len(neg) > 0:
        print(as_info(f'Negligible Violations:'))
        groups = Counter(type_key(item) for item in neg)
        for violation_type in LIST_OF_VIOLATION_TYPES:
            if groups[violation_type] == 0:
                continue
            print(as_info(f'\t{constraint_names[violation_type]}: {groups[violation_type]}' 
                          f'({str(round(groups[violation_type]/constraint_numbers[violation_type]*100, 1))}% of constraints)'))
    if len(mid) > 0:
        print(as_warning(f'Mild Violations:'))
        groups = Counter(type_key(item) for item in mid)
        for violation_type in LIST_OF_VIOLATION_TYPES:
            if groups[violation_type] == 0:
                continue
            print(as_warning(f'\t{constraint_names[violation_type]}: {groups[violation_type]}' 
                             f'({str(round(groups[violation_type]/constraint_numbers[violation_type]*100, 1))}% of constraints)'))
    if len(sev) > 0:
        print(as_fail(f'Severe Violations:'))
        groups = Counter(type_key(item) for item in sev)
        for violation_type in LIST_OF_VIOLATION_TYPES:
            if groups[violation_type] == 0:
                continue
            print(as_fail(f'\t{constraint_names[violation_type]}: {groups[violation_type]}' 
                          f'({str(round(groups[violation_type]/constraint_numbers[violation_type]*100, 1))}% of constraints)'))


def _check_centralized_flow_conservation(
        shift: int, flows: np.ndarray, graph: nx.DiGraph, 
        commodities: List[Commodity], feasibility_tol: Optional[float],
        feasibility_ratio: Optional[float] = None
    ) -> Tuple[List[Violation], Set[int]]:

    violations = []
    unsats = set()
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

            if v == SOURCE:
                if not is_satisfied(fout, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_OUTFLOW, k+shift, v, fout, DEMAND))
                    unsats.add(k)
                if not is_negligible(fin, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_LOOP, k+shift, v, fin, DEMAND))
            elif v == DESTINATION:
                if not is_negligible(fout, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_LEAK, k+shift, v, fout, DEMAND))
                if not is_satisfied(fin, DEMAND, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_INFLOW, k+shift, v, fin, DEMAND))
                    unsats.add(k)
            else:
                if not is_satisfied(fout, fin, feasibility_tol, feasibility_ratio):
                    violations.append((VIOLATION_INFLOW, k+shift, v, fin, fout))
    return violations, unsats


def report_violations(violations: List[Violation]):
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
        commodities: List[Commodity], 
        eval_params: TrafficEngineeringLPEvaluationParams
    ) -> Tuple[float, Set[int]]:
    """
    (PARALLEL VERSION)
    Check if solution satisfies all of the following constraints:
        - Transit nodes conserve flows                                    ( flow conservation )
        - A demand destined to a node, never flows out from that node     (  no demand leaks  )
        - A demand sourced from a node, never flows back into that node   (      no loops     )
    Returns the ratio of unsatisfied demands as well as the particular commodity indices
    """
    N = len(graph.edges())
    K = len(commodities)
    slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)

    # We'll accumulate the set of unsatisfied commodity indices
    unsatisfied_commodities: Set[int] = set()

    if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
        violations, unsatisfied_commodities = _check_centralized_flow_conservation(
            0, flows, graph, commodities, 
            feasibility_tol=eval_params.FeasibilityTolerance,
            feasibility_ratio=eval_params.FeasibilityRatio)
    else:
        with contextlib.closing(TempHelper(TEMP_FOLDER_NAME)) as tp:
            # MEMMAP the array to allow for concurrent writing
            input_path = tp.get_file_path(MEMMAP_FILE_NAME)
            cpu_dump(input_path, flows)
            X_KE = cpu_mmap(input_path, (N, K), 'r')
            nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
            print(as_info(f'Spawning {nprocs} workers to check flow conservation'))
            violations_it = Parallel(n_jobs=nprocs, return_as='generator')\
                (delayed(_check_centralized_flow_conservation)\
                    (begin, X_KE[:, begin:end], graph, commodities[begin:end],
                     eval_params.FeasibilityTolerance, eval_params.FeasibilityRatio)
                    for begin, end in slices)
            violations = []
            for item in violations_it:
                violations.extend(item[0])
                unsatisfied_commodities = unsatisfied_commodities.union(item[1])
            del X_KE
    if len(violations) == 0:
        print(as_success("No flow assignment violations were found."))
    else:
        print(as_warning("Flow assignment violations exist."))
        if eval_params.PrintReports:
            report_violations(violations)
        else:
            show_violation_severity(violations, K, graph.number_of_nodes())
    return len(unsatisfied_commodities)/K, unsatisfied_commodities
