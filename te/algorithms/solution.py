import os
import json
import dataclasses
import numpy as np
import gurobipy as gp
import networkx as nx
from typing import Any, Tuple, Optional
from te.algorithms import SOLUTION_DIR
from topologies.utils import load_zoo_topology, set_edge_capacity_to
from te.traffic_models import get_traffic_model, get_traffic_model_params
from te.traffic_models.base import TrafficMatrixBase
from te.algorithms.base import TrafficEngineeringLPSolution


class GurobiEdgeBasedMinimizeMaximumUtilitySolution(TrafficEngineeringLPSolution):
    """
    This class handles solution outputs from Gurobi.
    We assume the problem at hand is an LP, and the main purpose here is to
    provide enough information for a warm start.
    As such, we assume one of the Simplex methods (in particular, Dual Simplex) is
    used and the model is prepared to accept initial solutions.

    We try to be very generous to Gurobi, and as such, we output basis files (.bas)
    so that Gurobi can very quickly initialize itself.

    NOTE: There is benefit in making the output human-readable here. As such, this
          class dumps/loads solutions as JSON instead of pickle.
          The Gurobi `.bas` file is already human-readable.
    """
    def __init__(self, seed: int, topology_name: str, capacity: float, tm_model_name: str, tm_model_params: Any, 
                 gurobi_sol_path: Optional[str] = None, gurobi_sol_name: Optional[str] = None, runtime: Optional[float] = None):
        assert dataclasses.is_dataclass(tm_model_params)
        if gurobi_sol_path is None:
            assert gurobi_sol_name is not None

        self.seed = seed
        self.topology_name = topology_name
        self.capacity = capacity
        self.tm_model_name = tm_model_name
        self.tm_model_params = tm_model_params
        self.gurobi_sol_path = gurobi_sol_path \
            if gurobi_sol_path is not None \
            else os.path.join(SOLUTION_DIR, f'{gurobi_sol_name}.gurobi.bas')
        self.runtime = runtime
    
    def initiate_model(self, model: gp.Model):
        model.reset()
        model.read(self.gurobi_sol_path)

    def dump(self, model: gp.Model, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        model.write(self.gurobi_sol_path)
        with open(path, 'wb') as f:
            d = self.__dict__
            d.update({
                'tm_model_params': dataclasses.asdict(self.tm_model_params)
            })
            f.write(json.dumps(d, indent=4).encode())
    
    @classmethod
    def load(cls, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        with open(path, 'rb') as f:
            d = json.loads(f.read().decode())
            d.update({
                'tm_model_params': get_traffic_model_params(d['tm_model_name'])(**d['tm_model_params'])
            })
            return cls(**d)

    def regenerate(self) -> Tuple[nx.DiGraph, TrafficMatrixBase]:
        graph = load_zoo_topology(name=self.topology_name)
        set_edge_capacity_to(graph=graph, capacity=self.capacity)
        tm = get_traffic_model(self.tm_model_name)(seed=self.seed, params=self.tm_model_params)
        return (graph, tm)


class EdgeBasedMinimizeMaximumUtilitySolution(TrafficEngineeringLPSolution):
    def __init__(self, seed: int, topology_name: str, capacity: float, tm_model_name: str, tm_model_params: Any, 
                 assignments: np.ndarray, runtime: Optional[float]):
        self.seed = seed
        self.topology_name = topology_name
        self.capacity = capacity
        self.tm_model_name = tm_model_name
        self.tm_model_params = tm_model_params
        self.assignments = assignments
        self.runtime = runtime
    
    def regenerate(self) -> Tuple[nx.DiGraph, TrafficMatrixBase]:
        graph = load_zoo_topology(name=self.topology_name)
        set_edge_capacity_to(graph=graph, capacity=self.capacity)
        tm = get_traffic_model(self.tm_model_name)(seed=self.seed, params=self.tm_model_params)
        return (graph, tm)
