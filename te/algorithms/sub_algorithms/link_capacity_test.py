import gurobipy
import numpy as np
import networkx as nx
from typing import List, Union, Optional, Set, Tuple
from utils.logging import as_fail, as_info, str_round
from te.algorithms.base import TrafficEngineeringLPEvaluationParams
from te.traffic_models.base import Commodity


def is_congested(capacity, demand, feasibility_tol: Optional[float]):
    """Check if `demand` is lower than `capacity` within a given tolerance"""
    if abs(demand - capacity) < feasibility_tol:
        return False
    return demand > capacity


def check_capacity_constraint(
        flows: Union[gurobipy.tupledict, np.ndarray], graph: nx.DiGraph, commodities: List[Commodity], 
        eval_params: TrafficEngineeringLPEvaluationParams
    ) -> Tuple[float, Set[int]]:
    """Check if solution honors link capacity constraints"""
    feasibility_tol = eval_params.FeasibilityTolerance
    if feasibility_tol is None:
        feasibility_tol = eval_params.FloatResolution
    print(as_info(f"Checking capacity constraints with absolute tolerance of {feasibility_tol}"))

    K = len(commodities)
    N = len(graph.edges)

    if isinstance(flows, gurobipy.tupledict):
        X_EK = np.zeros((N, K))
        for e in range(N):
            for k in range(K):
                X_EK[e, k] = flows[e, k].X
    else:
        X_EK = flows
    
    X_O_E = np.sum(X_EK, axis=1)
    congested_edges: Set[int] = set()
    for e, (s, d, c_e) in enumerate(graph.edges(data='capacity')):
        demand = X_O_E[e]
        if is_congested(c_e, demand, feasibility_tol=feasibility_tol):
            if eval_params.PrintReports:
                cap_str = str_round(c_e, 4)
                demand_str = str_round(demand, 4)
                print(as_fail(f"Link {s} --> {d} Is Congested: {demand_str} > {cap_str}"))
            congested_edges.add(e)
    return len(congested_edges)/N, congested_edges