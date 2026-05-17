import gurobipy
import numpy as np
import networkx as nx
from typing import List, Union
from utils.logging import as_info, as_fail, ShortTQDM
from te.algorithms.base import TrafficEngineeringLPEvaluationParams
from te.traffic_models.base import Commodity


def check_loop_free_assignment(
        flows: Union[gurobipy.tupledict, np.ndarray], graph: nx.DiGraph, commodities: List[Commodity],
        eval_params: TrafficEngineeringLPEvaluationParams
    ) -> bool:
    """
    Check if each commodity assignment is loop-free.

    A commodity assignment is loop-free iff the directed subgraph induced by all
    positive flow assignments for that commodity is acyclic.

    Argumnets
    ---------
    flows: Union[gurobipy.tupledict, np.ndarray]
        Edge-based assignment to check for loops.
    graph: nx.DiGraph
        Topology graph.
    commodities: List[Commodity]
        List of commodities.
    eval_params: TrafficEngineeringLPEvaluationParams
        Evaluation parameters.
    
    Returns
    -------
    loop_free: bool
        If `False`, the solution contains at least one loop. It is hard to think
        of a scenario where this is acceptable.
    """
    feasibility_tol = eval_params.FeasibilityTolerance
    if feasibility_tol is None:
        feasibility_tol = eval_params.FloatResolution
    print(as_info(f"Checking loop-free assignments with positive-flow threshold {feasibility_tol}"))

    K = len(commodities)
    N = len(graph.edges)

    if isinstance(flows, gurobipy.tupledict):
        X_EK = np.zeros((N, K))
        for e in range(N):
            for k in range(K):
                X_EK[e, k] = flows[e, k].X
    else:
        X_EK = flows

    loop_free = True
    edges = list(graph.edges())
    for k in ShortTQDM(range(K)):
        commodity_graph = nx.DiGraph()
        commodity_graph.add_nodes_from(graph.nodes())
        commodity_edges = [
            edges[e] for e in range(N)
            if X_EK[e, k] > feasibility_tol
        ]
        commodity_graph.add_edges_from(commodity_edges)

        if not nx.is_directed_acyclic_graph(commodity_graph):
            loop_free = False
            commodity = commodities[k]
            if eval_params.PrintReports:
                print(as_fail(
                    f"Commodity {k} ({commodity.source} -> {commodity.destination}) "
                    f"contains a positive-flow cycle"
                ))
            else:
                break
    return loop_free
