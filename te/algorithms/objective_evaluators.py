import numpy as np
import networkx as nx


def get_maximum_link_utilization(edge_based_assignments: np.ndarray, graph: nx.DiGraph) -> float:
    """
    Returns the MLU from the assignment and capacities.
    Just take `sum_k X_{ek} / C_{e}` then report the maximum value.
    """
    capacities = np.array([edge['capacity'] for edge in graph.edges(data='capacity')])
    total_flow = np.sum(edge_based_assignments, axis=1)
    return np.max(total_flow / capacities)


def get_total_routed_flow(edge_based_assignments: np.ndarray, adjacency_matrix: np.ndarray) -> float:
    """
    Returns the total routed flow from the assignment and adjacency matrix.
    Essentially it first evaluates `F = MX`. If flow-conservation holds (
    which if it doesn't something is extremely wrong with `X`), we have
    
        F_{sk} = - F_{dk} = Total Routed Flow Of Commodity `k`
    
    And all other entries must be zero. So we just take the abs of `F`,
    sum it all up, then divide by 2.
    """
    flows = adjacency_matrix @ edge_based_assignments
    return np.sum(np.abs(flows, out=flows)) / 2


__all__ = [
    'get_maximum_link_utilization',
    'get_total_routed_flow'
]