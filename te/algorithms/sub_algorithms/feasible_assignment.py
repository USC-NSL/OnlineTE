import numpy as np
import networkx as nx
from typing import List, Union
from topologies.utils import Commodity, get_graph_M_matrix
from array_utils.cpu.types import *
from array_utils.cpu.sparse.types import *
from array_utils.cpu.sparse.wrapper import *


def _get_feasible_pseudoinv_flow_assignment(
    commodity_list: List[Commodity], pinv: CPUArray
) -> CPUArray:
    rows, cols, data = [], [], []
    _, M = pinv.shape
    for k, commodity in enumerate(commodity_list):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand
        rows.append(SOURCE)
        cols.append(k)
        data.append(DEMAND)
        rows.append(DESTINATION)
        cols.append(k)
        data.append(-DEMAND)
    B = cpu_coo_to_csc(cpu_coo_array(rows, cols, data, (M, len(commodity_list))))
    return pinv @ B


def get_feasible_flow_assignment(
    graph: nx.DiGraph,
    commodities: List[Commodity]
) -> Union[CPUCSRArray, CPUCSCArray, CPUArray]:
    """
    Create an assignment that satisfies all demands by taking the least
    squares solution to the demand constraints.
    """
    pinv = cpu_array(np.linalg.pinv(get_graph_M_matrix(graph=graph)))
    return _get_feasible_pseudoinv_flow_assignment(commodities, pinv)
