import os
import json
import numpy as np
import sympy as sp
import scipy.sparse as sparse
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp
    cp.get_array_module = lambda x: np
import networkx as nx
from scipy.linalg import null_space
from typing import Dict, Tuple, Union, List, Optional
from topologies import TOPOLOGIES_PATH, TOPOLOGY_ZOO_DIR_NAME, TOPOLOGY_ZOO_INDEX_FILE_NAME
from te.traffic_models.base import Commodity, commodity_od_iterator
from utils.logging import as_warning
from networkx.readwrite import json_graph


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

    # First, check for a GraphML file
    graphml_path = os.path.join(TOPOLGOY_ZOO_PATH, f"{name}.graphml")
    if os.path.exists(graphml_path):
        g: nx.Graph = nx.read_graphml(graphml_path, node_type=int)
    else:
        # Fallback to see if GML file exists instead
        gml_path = os.path.join(TOPOLGOY_ZOO_PATH, f"{name}.gml")
        if os.path.exists(gml_path):
            g: nx.Graph = nx.read_gml(gml_path, label='id', destringizer=int)
        else:
            # We don't have this topology :/ ...
            raise FileNotFoundError(f"No GML/GraphML file associated with topology {name} exists!")

    # Remove self loops (some topologies do have them, like `Interroute`)
    self_loops = list(nx.selfloop_edges(g))
    if len(self_loops) > 0:
        g.remove_edges_from(self_loops)
        print(as_warning(f"Removing {len(self_loops)} self-loop edges"))
        
    # Remove isolated nodes (some topologies have it, like "US Signal")
    isolateds = list(nx.isolates(g))
    if len(isolateds) > 0:
        g.remove_nodes_from(isolateds)
        g = nx.relabel_nodes(g, {n: i for i, n in enumerate(g.nodes())})
        print(as_warning(f"Removing {len(isolateds)} isolated nodes"))

    new_g = nx.Graph(g)
    
    if new_g.number_of_edges() != g.number_of_edges():
        print(as_warning(f"Removing {g.number_of_edges() - new_g.number_of_edges()} parallel edges"))

    return new_g.to_directed()


def get_edge_indexing(graph: nx.DiGraph) -> Dict[Tuple[int, int], int]:
    """Return a dict mapping edge to index for a graph."""

    assert isinstance(graph, nx.DiGraph)
    return {edge: index for index, edge in enumerate(graph.edges(data=False))}


def get_edge_to_out_index_mapping(graph: nx.DiGraph) -> Dict[Tuple[int, int], int]:
    """
    This returns a dict object that maps pairs of an edge to its
    out index for its source node.
    """

    assert isinstance(graph, nx.DiGraph)
    d = dict()
    counter = {node: 0 for node in graph.nodes(data=False)}
    for src, dst in graph.edges(data=False):
        d[(src, dst)] = counter[src]
        counter[src] += 1
    return d


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


def get_node_out_array(graph: nx.DiGraph) -> Dict[int, np.ndarray]:
    """Returns a mapping from node index to an array of out-going edge indices"""
    mapping: Dict[int, np.ndarray] = dict()
    indexing = get_edge_indexing(graph)
    
    for v in range(graph.number_of_nodes()):
        mapping[v] = np.zeros(shape=(graph.out_degree(v),), dtype=np.int32)
        for i, anc_v in enumerate(graph.successors(v)):
            mapping[v][i] = indexing[(v, anc_v)]
    return mapping


def get_node_in_array(graph: nx.DiGraph) -> Dict[int, np.ndarray]:
    """Returns a mapping from node index to an array of incoming edge indices"""
    mapping: Dict[int, np.ndarray] = dict()
    indexing = get_edge_indexing(graph)
    
    for v in range(graph.number_of_nodes()):
        mapping[v] = np.zeros(shape=(graph.in_degree(v),), dtype=np.int32)
        for i, pred_v in enumerate(graph.predecessors(v)):
            mapping[v][i] = indexing[(pred_v, v)]
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


def get_zoo_topology_at_least_as_large_as(n: int, m: int = -1, seed: Optional[int] = None) -> nx.DiGraph:
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
    
    if seed:
        chosen = np.random.default_rng(seed).choice(list(choices.keys()))
    else:
        chosen = np.random.choice(list(choices.keys()))
    print(f"Chose topology {chosen} of size {index[chosen]}")
    
    return load_zoo_topology(chosen)


# def get_capacity_lower_bound(graph: nx.DiGraph, traffic: np.ndarray) -> float:
#     """
#     This returns the `lower bound` for capacity needed for the problem to not be
#     `trivially` infeasible.
#     By `trivially` infeasible, we mean that the sum of demands that any node
#     sends and receives, must be less than the total capacity of all edges
#     that are connected to it.
#     This sets a low bar for the capacity to be assigned to the problem.

#     Note: This assumes all edges have the same capacity ...
#     """

#     cap = 0
#     degrees_in = graph.in_degree()
#     degrees_out = graph.out_degree()
#     flow_outs = np.sum(traffic, axis=1)
#     flow_ins = np.sum(traffic, axis=0)

#     assert len(flow_outs) == len(flow_ins)
#     for i, send_recv in enumerate(zip(flow_outs, flow_ins)):
#         sending, receiving = send_recv
#         cap = max(
#             cap, 
#             sending / degrees_out[i],
#             receiving / degrees_in[i]
#         )
    
#     return float(cap)


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


def set_random_capacities(graph: nx.DiGraph, topo_seed: Optional[int] = None):
    """
    Randomly sets capacities on all links by choosing randomly from
    the interval:

        [delta_min * m, delta_max * m]
    
    Where `delta_min` and `delta_max` are minimum and maximum node
    degrees and `m` is the number of nodes.

    Note
    ----
    This _always_ makes sure that edges connected to the same nodes
    have the same capacity.
    """
    degs = list([d for _, d in graph.in_degree()])
    M = graph.number_of_nodes()
    N = graph.number_of_edges()
    l = min(degs)
    u = max(degs)
    rng = np.random.default_rng(topo_seed)
    assert N % 2 == 0
    caps = (rng.random(size=(N // 2,)) * (u - l) + l) * M
    edges = set()
    for edge in graph.edges(data=False):
        if edge not in edges:
            u, v = edge
            graph[u][v]['capacity'] = caps[len(edges) // 2]
            graph[v][u]['capacity'] = caps[len(edges) // 2]
            edges.add((u, v))
            edges.add((v, u))


def get_graph_M_matrix(graph: nx.DiGraph) -> np.ndarray:
    assert isinstance(graph, nx.DiGraph)

    m = len(graph.nodes())
    n = len(graph.edges())
    M = np.zeros(shape=(m, n))
    
    for i, (s, d) in enumerate(graph.edges(data=False)):
        M[s, i] = +1
        M[d, i] = -1
    
    return M


def get_adjacency_null_space(M_matrix: np.ndarray) -> np.ndarray:
    assert len(M_matrix.shape) == 2

    return null_space(M_matrix)


def get_sparse_null_space(M_matrix: np.ndarray) -> np.ndarray:
    """
    Get the sparse nullspace basis for the `M` matrix.
    Note that this basis while very sparse, no longer has orthogonal columns!
    """
    symbolic_M = sp.Matrix(M_matrix)
    basis = symbolic_M.nullspace()
    np_basis = np.hstack([np.array(base.tolist(), dtype=np.float64) for base in basis])
    return np_basis / np.linalg.norm(np_basis, axis=0)


def get_commodity_in_out_mask(
    graph: nx.DiGraph,
    edge_indexing: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """
    Returns a Boolean valued mask of size `n x k` where entry `ek` is `True`
    for eny edge leaving the destination of commodity `k` or flowing into the
    source of commodity `k`.
    Entries that are masked with this must be _very_ close to zero for any
    acceptable solution, since otherwise it means that the final assignment
    may have created loops between the source and the destination.
    """
    M = graph.number_of_nodes()
    mask = np.zeros(dtype=bool, shape=(graph.number_of_edges(), M*(M-1)))
    for k, od_pair in enumerate(commodity_od_iterator(M)):
        source, destination = od_pair
        for edge in graph.out_edges(nbunch=destination, data=False):
            mask[edge_indexing[edge], k] = True
        for edge in graph.in_edges(nbunch=source, data=False):
            mask[edge_indexing[edge], k] = True
    return mask


def get_sparse_commodity_satisfaction_mask(
    graph: nx.DiGraph,
    edge_indexing: Dict[Tuple[int, int], int]
) -> Tuple[sparse.csc_matrix, sparse.csc_matrix]:
    M = graph.number_of_nodes()
    mask_source_out = sparse.lil_matrix((graph.number_of_edges(), M*(M-1)), dtype=bool)
    mask_source_in = sparse.lil_matrix((graph.number_of_edges(), M*(M-1)), dtype=bool)
    for k, od_pair in enumerate(commodity_od_iterator(M)):
        source, _ = od_pair
        for edge in graph.out_edges(nbunch=source, data=False):
            mask_source_out[edge_indexing[edge], k] = True
        for edge in graph.in_edges(nbunch=source, data=False):
            mask_source_in[edge_indexing[edge], k] = True
    return mask_source_out.tocsc(), mask_source_in.tocsc()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    g = get_zoo_topology_at_least_as_large_as(100, 200)
    # g = get_zoo_topology_at_least_as_large_as(60, 70)
    # if g:
    #     nx.draw(g)
    #     plt.show()
    # g = load_zoo_topology('Interoute')
    # print(f'Nodes: {len(g.nodes)}')
    # print(f'Edges: {len(g.edges)}')
    # g = load_zoo_topology('Claranet')
    # g = get_artificial_topology(300, seed=12345)
    # nx.draw(nx.to_undirected(g), pos=nx.kamada_kawai_layout(g, scale=3), with_labels=True)
    # plt.show()
