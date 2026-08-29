import enum
import networkx as nx
from typing import Optional, List, Iterator, Union, Callable, Dict


class PathSchemes(str, enum.Enum):
    SHORTEST_PATH = "SHORTEST-PATH"
    EDGE_DISJOINT = "EDGE-DISJOINT"


def default_weight(u: int, v: int, attributes: Dict) -> float:
    """
    Default shortest path weights is the inverse of edge capacity.
    """
    return 1 / attributes['capacity']


def shortest_path_per_commodity_provider(
    graph: nx.DiGraph, source: int, destination: int,
    weight: Optional[Union[str, Callable[[int, int, Dict], float]]] = default_weight
) -> Iterator[List[int]]:
    """
    Simple shortest path iterator based on Yen's algorithm.

    Optionally accepts a `weight` which should be an attribute on
    the graph (e.g. latency).
    """
    return nx.shortest_simple_paths(graph, source, destination, weight=weight)


def edge_disjoint_path_per_commodity_provider(
    graph: nx.DiGraph, source: int, destination: int
) -> Iterator[List[int]]:
    """Edge-disjoint paths iterator"""
    return nx.edge_disjoint_paths(graph, source, destination)


def get_scheme(scheme: Optional[PathSchemes] = None):
    if scheme is None or scheme == PathSchemes.SHORTEST_PATH:
        return shortest_path_per_commodity_provider
    elif scheme == PathSchemes.EDGE_DISJOINT:
        return edge_disjoint_path_per_commodity_provider
    else:
        raise ValueError


__all__ = ['PathSchemes', 'get_scheme']