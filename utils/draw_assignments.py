import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from te.traffic_models.base import traffic_to_commodity
from te.algorithms.solution import GurobiEdgeBasedMinimizeMaximumUtilitySolution


CommodityDict = Dict[Tuple[int, int], Tuple[int, float]]


within_tolerance = lambda x: 0 if abs(x) < 1e-1 else x


def load_solution_info(solution_name: str, initiate_weights: bool = True) -> Tuple[nx.DiGraph, np.ndarray, float, CommodityDict]:
    base_solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution = GurobiEdgeBasedMinimizeMaximumUtilitySolution.load(name=solution_name)
    graph, tm = base_solution.regenerate()
    assignments, u = base_solution.get_vars()
    capacity = base_solution.capacity * u
    commodity_list = traffic_to_commodity(tm)
    if initiate_weights:
        graph = set_graph_weights_to_zero(graph)
    return graph, assignments, capacity, {(item.source, item.destination): (index, item.demand) for index, item in enumerate(commodity_list)}


def set_graph_weights_to_zero(g: nx.DiGraph):
    for e in g.edges():
        g[e[0]][e[1]]['weight'] = 0
        g[e[0]][e[1]]['has_arrow'] = False
    return g


def draw_directed_weighted_graph(graph: nx.DiGraph):
    # g: nx.Graph = nx.to_undirected(graph)
    g = graph
    arrowed_edges = []
    arrowed_edge_widths = dict()
    no_arrow_edges = []
    
    for s, d in g.edges():
        if g[s][d]['has_arrow']:
            arrowed_edges.append((s, d))
            arrowed_edge_widths[(s, d)] = within_tolerance(g[s][d]['weight'])
        else:
            no_arrow_edges.append((s, d))
    
    pos = nx.kamada_kawai_layout(g, weight=None)
    nx.draw_networkx_edges(
        G=g,
        pos=pos,
        edgelist=no_arrow_edges,
        edge_color='black',
        arrows=False,
        alpha=0.6)
    nx.draw_networkx_edges(
        G=g,
        pos=pos,
        edgelist=arrowed_edge_widths.keys(),
        width=list(arrowed_edge_widths.values()),
        edge_color='red',
        arrows=True,
        arrowstyle='->',
        arrowsize=20,
        alpha=0.8)
    nx.draw_networkx_nodes(
        G=g,
        pos=pos,
        nodelist=g.nodes(),
        node_size=300,
        node_color='black',
        alpha=0.7)
    nx.draw_networkx_labels(
        G=g,
        pos=pos,
        labels=dict(zip(g.nodes(), g.nodes())),
        font_color='white')


def get_assignments_for_particular_commodity(source: int, destination: int, graph: nx.DiGraph, capacity: float,
                                             assignments: np.ndarray, commodity_dict: CommodityDict) -> nx.DiGraph:
    commodity_index, _ = commodity_dict[(source, destination)]
    commodity_assignment = assignments[:, commodity_index]
    for i, e in enumerate(graph.edges()):
        requested_assignment = within_tolerance(commodity_assignment[i]) / capacity * 100
        graph[e[0]][e[1]]['weight'] += requested_assignment
        if requested_assignment > 0:
            graph[e[0]][e[1]]['has_arrow'] = True
    return graph


def load_and_draw_assignment_for_source_and_destination(solution_name: str, source: int, destination: int):
    graph, assignments, capacity, commodity_dict = load_solution_info(solution_name)
    graph = get_assignments_for_particular_commodity(source, destination, graph, capacity, assignments, commodity_dict)
    draw_directed_weighted_graph(graph)


if __name__ == '__main__':
    topology_name = 'Claranet'
    base_seed = 12345
    tm_model = 'Uniform'

    plt.figure(figsize=(15, 5), dpi=100)
    plt.subplot(1, 3, 1)
    load_and_draw_assignment_for_source_and_destination(f'{topology_name}_{base_seed}_{tm_model}.tesol', 1, 8)
    plt.title('Simplex')
    plt.subplot(1, 3, 2)
    load_and_draw_assignment_for_source_and_destination(f'{topology_name}_{base_seed}_{tm_model}_barrier.tesol', 1, 8)
    plt.title('Barrier')
    plt.subplot(1, 3, 3)
    load_and_draw_assignment_for_source_and_destination(f'{topology_name}_{base_seed}_{tm_model}_barrier_crossed.tesol', 1, 8)
    plt.title('Barrier + Crossover')
    plt.show()

    # solution_name = f'{topology_name}_{base_seed}_{tm_model}.tesol'
    # solution_name = f'{topology_name}_{base_seed}_{tm_model}_barrier.tesol'
    # solution_name = f'{topology_name}_{base_seed}_{tm_model}_barrier_crossed.tesol'
