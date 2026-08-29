import math
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple, Dict
from array_utils.cpu.types import *
from te.traffic_models.base import Commodity, commodity_od_iterator
from topologies.utils import get_sparse_commodity_satisfaction_mask


def is_leq(upper_bound: float, value: float, feasibility_tol: float):
    """Check if `value` is less than or equal to `upper_bound` within a given tolerance"""
    if abs(value - upper_bound) < feasibility_tol:
        return False
    return value > upper_bound


def is_eq(target: float, actual: float, feasibility_tol: float):
    """Check if `actual` is close to `target` assignment"""
    return math.isclose(target, actual, abs_tol=feasibility_tol)


def check_capacity_constraint(
    edge_based_assignment: np.ndarray,
    graph: nx.DiGraph,
    feasibility_tolerance: float
) -> List[Tuple[int, int, int, float, float]]:
    """
    Check if solution honors link capacity constraints.
    Returns a list of violators as 5-tuple:

        edge_index, source_index, destination_index, demand, capacity
    """
    N, _ = edge_based_assignment.shape
    X_O_E = np.sum(edge_based_assignment, axis=1)
    congested_edges: List[Tuple[int, int, int, float, float]] = []
    for e, (s, d, c_e) in enumerate(graph.edges(data='capacity')):
        demand = X_O_E[e]
        if is_leq(c_e, demand, feasibility_tol=feasibility_tolerance):
            congested_edges.append((e, s, d, demand, c_e))
    return len(congested_edges)/N, congested_edges


def vector_consensus_test(
    vec1: np.ndarray,
    vec2: np.ndarray,
    feasibility_tol: float
) -> List[Tuple[int, float, float]]:
    """
    Check if two vectors are close enough given tolerances.
    Returns a list of violators as 3-tuples:

        axis, value_1, value_2
    """
    assert vec1.shape == vec2.shape
    assert vec1.ndim == 1
    n = len(vec1)
    
    violations = []
    for e in range(n):
        v1_e = vec1[e]
        v2_e = vec2[e]
        if not is_eq(v1_e, v2_e, feasibility_tol):
            violations.append((e, v1_e, v2_e))

    return violations


def check_loop_free_assignment(
        edge_based_assignment: np.ndarray,
        graph: nx.DiGraph,
        loop_tolerance: float
    ) -> Optional[Tuple[int, int, int]]:
    """
    Check if each commodity assignment is loop-free.
    Returns a witness commodity that has a loop, otherwise None.
    """
    M = graph.number_of_nodes()
    K = M * (M - 1)
    assert edge_based_assignment.shape[-1] == K

    witness = None
    edges = np.array(graph.edges(data=False))
    for k, od_pair in enumerate(commodity_od_iterator(M)):
        source, destination = od_pair
        commodity_graph = nx.DiGraph()
        commodity_graph.add_nodes_from(graph.nodes())
        commodity_graph.add_edges_from(edges[(edge_based_assignment[:, k] > loop_tolerance)])

        if not nx.is_directed_acyclic_graph(commodity_graph):
            witness = (k, source, destination)
            break
    return witness


def check_flow_leaks(
    edge_based_assignment: np.ndarray,
    graph: nx.DiGraph, 
    commodities: List[Commodity],
    feasibility_tolerance: float,
    edge_indexing: Dict[Tuple[int, int], int]
) -> List[Tuple[int, int, int, float]]:
    """
    Checks for demand leaks at the source by reporting:

        commodity_id, source, destination, total_flow_in
    
    Applies to all problems.
    """
    _, source_in = get_sparse_commodity_satisfaction_mask(graph, edge_indexing)
    source_in_demand = np.squeeze(
        cpu_array(source_in.multiply(edge_based_assignment).sum(axis=0)),
        axis=0
    )
    demands = [commodity.demand for commodity in commodities]
    loop_violation_indices = np.where(
        ~np.isclose(
            source_in_demand,
            0,
            atol=feasibility_tolerance
        )
    )[0]
    
    violations = []
    for i in loop_violation_indices:
        if not is_eq(
            source_in_demand[i], demands[i],
            feasibility_tolerance
        ):
            violations.append((
                i, commodities[i].source,
                commodities[i].destination,
                source_in_demand[i]
            ))
    return violations


def check_flow_satisfaction(
    edge_based_assignment: np.ndarray,
    graph: nx.DiGraph, 
    commodities: List[Commodity],
    feasibility_tolerance: float,
    edge_indexing: Dict[Tuple[int, int], int]
) -> List[Tuple[int, int, int, float, float]]:
    """
    Checks for unsatisfied demand in the source by reporting:

        commodity_id, source, destination, total_flow_out, demand
    
    Applies to MLU only.
    """
    source_out, _ = get_sparse_commodity_satisfaction_mask(graph, edge_indexing)
    source_out_demand = np.squeeze(
        cpu_array(source_out.multiply(edge_based_assignment).sum(axis=0)), 
        axis=0
    )
    demands = [commodity.demand for commodity in commodities]

    out_flow_violation_indices = np.where(
        ~np.isclose(
            source_out_demand,
            demands,
            atol=feasibility_tolerance
        )
    )[0]

    violations = []
    for i in out_flow_violation_indices:
        if not is_eq(
            demands[i], source_out_demand[i],
            feasibility_tolerance
        ):
            violations.append((
                i, commodities[i].source, commodities[i].destination,
                source_out_demand[i], demands[i]
            ))
    return violations


__all__ = [
    'check_capacity_constraint', 'vector_consensus_test', 'check_loop_free_assignment',
    'check_flow_leaks', 'check_flow_satisfaction'
]