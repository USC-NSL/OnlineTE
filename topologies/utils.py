import os
import json
import numpy as np
import sympy as sp
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp
    cp.get_array_module = lambda x: np
import networkx as nx
from scipy.linalg import null_space
from typing import Dict, Tuple, Union, List, Optional
from topologies import (
    TOPOLOGIES_PATH, TOPOLOGY_ZOO_DIR_NAME, 
    TOPOLOGY_ZOO_INDEX_FILE_NAME
)
from te.traffic_models.base import TrafficMatrixBase, Commodity
from te.traffic_models.models import UniformTrafficMatrix, UniformTrafficMatrixParams
from utils.logging import as_warning


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


is_artificial = lambda name: name.lower().startswith('artificial-')
get_artificial_size = lambda name: int(name.lower().split('artificial-')[-1])


def load_zoo_topology(name: str, seed: Optional[int] = None) -> nx.DiGraph:
    """Load the topology zoo model as an instance of `nx.DiGraph`"""

    if is_artificial(name):
        return get_artificial_topology(get_artificial_size(name), seed=seed)

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
            raise ValueError(f"No GML/GraphML file associated with topology {name} exists!")

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


def _get_graph_list_to_join(n_nodes: int, num: int = 1, seed: Optional[int] = None) -> List[nx.DiGraph]:
    """
    This tries to find `num` zoo topologies such that their total number of nodes is
    at least `n_nodes`.
    To do this, we first attempt to find a graph of size `n_nodes / num`. If found, we
    subtract its size from `n_nodes` and repeat with `num = 1`. This is done to encourage
    variety in size of chosen graphs (we would prefer a small graph and a big one, instead
    of many small ones ...).
    If such a graph is not found, we increase `num` by 1 and repeat.
    """
    if num == 10 or n_nodes == num or n_nodes == 0:
        # Give up on the 10th iteration ....
        return []
    upper_bound_tol = int(np.sqrt(n_nodes))
    g = get_zoo_topology_at_least_as_large_as(n=n_nodes//num, m=(n_nodes + upper_bound_tol)//num)
    if g:
        return [g] + _get_graph_list_to_join(max(n_nodes, (n_nodes + upper_bound_tol)//num) - g.number_of_nodes(), seed=seed)
    else:
        return _get_graph_list_to_join(n_nodes, num=num+1, seed=seed)


def _compose_undirected_graphs(graphs: List[nx.Graph]) -> Tuple[nx.Graph, List[int]]:
    sizes = [0]
    for g in graphs:
        sizes.append(g.number_of_nodes() + sizes[-1])
    sizes.pop(-1)
    composite = nx.Graph()
    for i, g in enumerate(graphs):
        size_until_now = sizes[i]
        composite.add_nodes_from(range(size_until_now, size_until_now + g.number_of_nodes()))
        edges = [(s+size_until_now, d+size_until_now) for s, d in g.edges(data=False)]
        composite.add_edges_from(edges)
    return composite, sizes


def get_artificial_topology(n_nodes: int, seed: Optional[int] = None) -> nx.DiGraph:
    """
    Creates a network by randomly picking some networks from the zoo and
    Frakensteining them together.
    The result will contain _at least_ `n_nodes` nodes. There are no
    guarantees about the number of edges though.

    To join `n` graphs together, we choose `sqrt(n_nodes)/n` nodes
    from each graph, and then randomly connect them to eachother
    in a ring.
    """

    graph_list = _get_graph_list_to_join(n_nodes, seed=seed)
    if len(graph_list) == 1:
        return graph_list[0]
    undirs = [g.to_undirected() for g in graph_list]
    composite_graph, sizes = _compose_undirected_graphs(undirs)
    n_connections = int(np.sqrt(n_nodes) / len(undirs))
    rng = np.random.default_rng(seed=seed)
    chosen_nodes = [sizes[i] + rng.choice(g.number_of_nodes(), size=n_connections) for i, g in enumerate(undirs)]
    for i in range(len(undirs)):
        for s, d in zip(chosen_nodes[i], chosen_nodes[(i+1) % len(undirs)]):
            composite_graph.add_edge(s, d)
    return composite_graph.to_directed()


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
    
    return float(cap)


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


def load_zoo_topology_with_capacity_heuristic(name: str, tm: TrafficMatrixBase, scale_factor: float = 10) -> Tuple[float, nx.DiGraph]:
    """
    Load a topology from the zoo and assign a single capacity value to each edge.
    The capacity is chosen heuristically. Its value is set to be `scale_factor * c`,
    where `c` is the return value of `get_capacity_lower_bound`.
    """
    
    graph = load_zoo_topology(name)
    c_min = get_capacity_lower_bound(graph, tm)
    c = c_min * scale_factor
    set_edge_capacity_to(graph, c)
    
    return c, graph


def get_uniform_tm_problem_with_capacity_heuristic(
        topo_name: str, tm_seed: int, scale_factor: float = 10
    ) -> Tuple[float, nx.DiGraph, TrafficMatrixBase]:
    """
    Given a topology name, create the input for a TE problem.
    Returns an heuristically assigned capacity value, a graph and a uniform traffic matrix.
    """
    
    graph = load_zoo_topology(topo_name)
    tm_params = UniformTrafficMatrixParams(n = len(graph.nodes), min = 0.0, max = 1.0)
    tm = UniformTrafficMatrix(seed=tm_seed, params=tm_params)
    c_min = get_capacity_lower_bound(graph, tm)
    c = c_min * scale_factor
    set_edge_capacity_to(graph, c)
    
    return c, graph, tm


def get_graph_M_matrix(graph: nx.DiGraph) -> np.ndarray:
    assert isinstance(graph, nx.DiGraph)

    m = len(graph.nodes())
    n = len(graph.edges())
    M = np.zeros(shape=(m, n))
    
    for i, (s, d) in enumerate(graph.edges(data=False)):
        M[s, i] = +1
        M[d, i] = -1
    
    return M


def get_symbolic_graph_M_matrix(graph: nx.DiGraph) -> sp.Matrix:
    assert isinstance(graph, nx.DiGraph)

    m = len(graph.nodes())
    n = len(graph.edges())
    M = [[0 for _ in range(n)] for __ in range(m)]
    
    for i, (s, d) in enumerate(graph.edges(data=False)):
        M[s][i] = +1
        M[d][i] = -1
    
    return sp.Matrix(M)


def get_adjacency_null_space(M_matrix: np.ndarray) -> np.ndarray:
    assert len(M_matrix.shape) == 2

    return null_space(M_matrix)


def get_sparse_null_space(symbolic_M_matrix: sp.Matrix) -> np.ndarray:
    basis = symbolic_M_matrix.rref(pivots=False).nullspace()
    orthonormal_basis = sp.GramSchmidt(basis, orthonormal=True)
    return np.hstack([np.array(base.tolist(), dtype=np.float64) for base in orthonormal_basis])


def get_feasible_flow_assignment(graph: nx.DiGraph, commodities: List[Commodity]):
    N = len(graph.edges())
    K = len(commodities)
    X_KE = np.zeros(shape=(N, K))
    EDGE_INDEXING = get_edge_indexing(graph)
    
    for k, commodity in enumerate(commodities):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand
        path = nx.shortest_path(graph, SOURCE, DESTINATION)
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            X_KE[EDGE_INDEXING[edge], k] = DEMAND
    return X_KE


# def get_feasible_flow_assignment_gpu(graph: nx.DiGraph, commodities: List[Commodity]):
#     N = len(graph.edges())
#     K = len(commodities)
#     X_KE = cp.zeros(shape=(N, K))
#     EDGE_INDEXING = get_edge_indexing(graph)
    
#     for k, commodity in enumerate(commodities):
#         SOURCE = commodity.source
#         DESTINATION = commodity.destination
#         DEMAND = commodity.demand
#         path = nx.shortest_path(graph, SOURCE, DESTINATION)
#         for i in range(len(path) - 1):
#             edge = (path[i], path[i+1])
#             X_KE[EDGE_INDEXING[edge], k] = DEMAND
#     return X_KE


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # g = get_zoo_topology_at_least_as_large_as(150, 700)
    # g = get_zoo_topology_at_least_as_large_as(60, 70)
    # if g:
    #     nx.draw(g)
    #     plt.show()
    # g = load_zoo_topology('Interoute')
    # print(f'Nodes: {len(g.nodes)}')
    # print(f'Edges: {len(g.edges)}')
    # g = load_zoo_topology('Claranet')
    g = get_artificial_topology(300, seed=12345)
    nx.draw(nx.to_undirected(g), pos=nx.kamada_kawai_layout(g, scale=3), with_labels=True)
    plt.show()
