import os
import json
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Union, List
from topologies import (
    TOPOLOGIES_PATH, TOPOLOGY_ZOO_DIR_NAME, 
    TOPOLOGY_ZOO_INDEX_FILE_NAME
)
from te.traffic_models.base import TrafficMatrixBase


TOPOLGOY_ZOO_PATH = os.path.join(TOPOLOGIES_PATH, TOPOLOGY_ZOO_DIR_NAME)
TOPOLOGY_ZOO_INDEX_PATH = os.path.join(TOPOLOGIES_PATH, TOPOLOGY_ZOO_INDEX_FILE_NAME)


"""
IMPORTANT NOTE:
    If using anything below Python 3.7, this code would not be
    safe, since it returns a `nx.Graph` instance.
    Older python `dict` objects are not ordered, thus order of
    edges are not preserved, which breaks our solver.
    If you are using older python, wrap the result of loading
    from GraphML in `nx.ordered.OrderedGraph`.
"""


def load_zoo_topology(name: str) -> nx.DiGraph:
    """Load the topology zoo model as an instance of `nx.DiGraph`"""
    
    gml_path = os.path.join(TOPOLGOY_ZOO_PATH, f"{name}.graphml")
    assert os.path.exists(gml_path)

    g: nx.Graph = nx.read_graphml(gml_path, node_type=int)
    # Remove self loops (some topologies do have them, like `Interroute`)
    self_loops = list(nx.selfloop_edges(g))
    if len(self_loops) > 0:
        g.remove_edges_from(self_loops)
        print(f"Removing {len(self_loops)} self-loop edges")
        
    # Remove isolated nodes (some topologies have it, like "US Signal")
    isolateds = list(nx.isolates(g))
    if len(isolateds) > 0:
        g.remove_nodes_from(isolateds)
        g = nx.relabel_nodes(g, {n: i for i, n in enumerate(g.nodes())})
        print(f"Removing {len(isolateds)} isolated nodes")

    return g.to_directed()


def get_edge_indexing(graph: nx.DiGraph) -> Dict[Tuple[int, int], int]:
    """Return a dict mapping edge to index for a graph."""

    assert isinstance(graph, nx.DiGraph)
    return {edge: index for index, edge in enumerate(graph.edges(data=False))}


def get_node_and_out_edge_index_mapping(graph: nx.DiGraph) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    This returns a dict object that maps pairs of (node_index,
    out_edge_index) to (edge_index, out_node_index).
    """

    assert isinstance(graph, nx.DiGraph)
    d = dict()
    counter = {node: 0 for node in graph.nodes(data=False)}
    for edge_index, (src, dst) in enumerate(graph.edges(data=False)):
        d[(src, counter[src])] = (edge_index, dst)
        counter[src] += 1
    return d


def get_in_edge_mapping(graph: nx.DiGraph):
    """
    This creates a mapping from node index to a list of
    tuples of (pred_node_index, pred_edge_out_index).
    """

    m = len(graph.nodes)
    mapping: Dict[int, List[Tuple[int, int]]] = dict()
    
    for v in range(m):
        mapping[v] = list()
        for pred_v in graph.predecessors(v):
            for i, (_, dst) in enumerate(graph.out_edges(pred_v, data=False)):
                if v == dst:
                    mapping[v].append([pred_v, i])

    return mapping


def index_zoo():
    """Make a map from topology name to its number of nodes"""
    
    assert os.path.exists(TOPOLGOY_ZOO_PATH)
    if os.path.exists(TOPOLOGY_ZOO_INDEX_PATH):
        return

    gmls = filter(lambda name: name.endswith('.gml'), os.listdir(TOPOLGOY_ZOO_PATH))
    index = dict()
    
    for gml_name in gmls:
        name = gml_name.replace('.gml', '')
        with open(os.path.join(TOPOLGOY_ZOO_PATH, gml_name)) as gml:
            index[name] = gml.read().count("node")
    
    # Save the index, we'll use it later
    with open(TOPOLOGY_ZOO_INDEX_PATH, 'w') as findex:
        json.dump(index, findex, indent=2)


def load_index() -> Dict[str, int]:
    """Load the index of zoo topologies, or make it from scratch"""

    assert os.path.exists(TOPOLGOY_ZOO_PATH)
    if not os.path.exists(TOPOLOGY_ZOO_INDEX_PATH):
        index_zoo()
    
    with open(TOPOLOGY_ZOO_INDEX_PATH, 'r') as findex:
        return json.load(findex)


def get_zoo_topology_at_least_as_large_as(n: int, m: int = -1) -> nx.DiGraph:
    """Pick a random topology that has at least `n` nodes and at most `m`, if given."""
    
    assert os.path.exists(TOPOLGOY_ZOO_PATH)
    index = load_index()

    if m <= 0:
        choices = {name: size for name, size in index.items() if size >= n}
    else:
        choices = {name: size for name, size in index.items() if (size >= n and size <= m)}

    if len(choices) == 0:
        if m <= 0:
            print(f"No topology of size at least {n} exists in the zoo.")
        else:
            print(f"No topology of size at least {n} and at most {m} exists in the zoo.")
        return None
    
    chosen = np.random.choice(list(choices.keys()))
    print(f"Chose topology {chosen} of size {index[chosen]}")
    
    return load_zoo_topology(chosen)


def set_edge_capacity_to(graph: nx.DiGraph, capacity: float, edge: Union[Tuple[int, int], List[Tuple[int, int]]] = None):
    """Set capacity of edge(s) to a particular value"""

    if edge is None:
        nx.set_edge_attributes(graph, capacity, 'capacity')
    elif isinstance(edge, tuple):
        nx.set_edge_attributes(graph, {edge: capacity}, 'capacity')
    elif isinstance(edge, list):
        nx.set_edge_attributes(graph, {e: capacity for e in edge}, 'capacity')
    else:
        raise ValueError


def set_edge_capacity_randomly(graph: nx.DiGraph, low: float, high: float, seed: int = None):
    """Set capacity on each edge by randomly picking from [low, high)"""

    assert low < high
    rng = np.random.default_rng(seed)
    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: (high - low) * rng.random() + low})


def get_capacity_lower_bound(graph: nx.DiGraph, traffic: TrafficMatrixBase) -> float:
    """
    This returns the `lower bound` for capacity needed for the problem to not be
    `trivially` infeasible.
    By `trivially` infeasible, we mean that the sum of demands that any node
    sends and receives, must be less than the total capacity of all edges
    that are connected to it.
    This sets a low bar for the capacity to be assigned to the problem.

    Note: This assumes all edges have the same capacity ...
    """

    cap = 0
    degrees_in = graph.in_degree()
    degrees_out = graph.out_degree()
    flow_outs = np.sum(traffic.tm, axis=1)
    flow_ins = np.sum(traffic.tm, axis=0)

    assert len(flow_outs) == len(flow_ins)
    for i, send_recv in enumerate(zip(flow_outs, flow_ins)):
        sending, receiving = send_recv
        cap = max(
            cap, 
            sending / degrees_out[i],
            receiving / degrees_in[i]
        )
    
    return cap


def make_graph_from_dict(graph_n: int, graph_dict: Dict[Tuple[int, int], float]) -> nx.DiGraph:
    """
    Create a graph out of a dictionary object that maps edges
    to float numbers that represent capacity.
    The dictionary is undirected, we automatically convert the final
    graph to directed manually (so in and out capacities must be the
    same!).
    """
    
    dict_of_dicts = {k: {} for k in range(graph_n)}
    for k, v in graph_dict.items():
        src, dst = k
        dict_of_dicts[src][dst] = {}
        dict_of_dicts[src][dst]['capacity'] = v
    
    return nx.Graph(dict_of_dicts).to_directed()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # g = get_zoo_topology_at_least_as_large_as(100, 112)
    g = get_zoo_topology_at_least_as_large_as(60, 70)
    if g:
        nx.draw(g)
        plt.show()
