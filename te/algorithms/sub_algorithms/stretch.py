import te.constants
import networkx as nx
from typing import List, Dict
from te.algorithms.base import Commodity
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_array


def get_average_stretch(commodity_list: List[Commodity], edge_based_assignment: CPUArray, 
                        graph: nx.DiGraph) -> CPUArray:
    stretch = []
    aggregate = edge_based_assignment.sum(axis=0)
    shortest_path_lens: Dict[int, Dict[int, int]] = dict(nx.shortest_path_length(graph))
    for k, commodity in enumerate(commodity_list):
        if commodity.demand <= te.constants.FLOAT_RES:
            continue
        average_delay = aggregate[k] / (commodity.demand + te.constants.FLOAT_RES)
        stretch.append(average_delay / shortest_path_lens[commodity.source][commodity.destination])
    return cpu_array(stretch)


def get_utilizations(edge_based_assignment: CPUArray, graph: nx.DiGraph) -> CPUArray:
    aggregate = edge_based_assignment.sum(axis=1)
    c = cpu_array([c_e for _, _, c_e in graph.edges(data='capacity')])
    return aggregate / c
