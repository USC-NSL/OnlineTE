import numpy as np
import networkx as nx
from typing import Any, Tuple
from topologies.utils import load_zoo_topology, set_edge_capacity_to
from te.traffic_models import get_traffic_model
from te.traffic_models.base import TrafficMatrixBase
from te.algorithms.base import TrafficEngineeringLPSolution


class EdgeBasedMinimizeMaximumUtilitySolution(TrafficEngineeringLPSolution):
    def __init__(self, seed: int, topology_name: str, capacity: float, tm_model_name: str, tm_model_params: Any, assignments: np.ndarray):
        self.seed = seed
        self.topology_name = topology_name
        self.capacity = capacity
        self.tm_model_name = tm_model_name
        self.tm_model_params = tm_model_params
        self.assignments = assignments
    
    def regenerate(self) -> Tuple[nx.DiGraph, TrafficMatrixBase]:
        graph = load_zoo_topology(name=self.topology_name)
        set_edge_capacity_to(graph=graph, capacity=self.capacity)
        tm = get_traffic_model(self.tm_model_name)(seed=self.seed, params=self.tm_model_params)
        return (graph, tm)
