import os
import json
import dataclasses
import numpy as np
import gurobipy as gp
import networkx as nx
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List
from te.algorithms import SOLUTION_DIR
from topologies.utils import load_zoo_topology, set_edge_capacity_to
from te.algorithms.utils import as_warning
from te.traffic_models import get_traffic_model, get_traffic_model_params, get_traffic_converter, get_traffic_converter_params
from te.traffic_models.base import TrafficMatrixBase, TrafficMatrixConverterBase
from te.algorithms.base import TrafficEngineeringLPSolution


key_to_str = lambda key: ','.join([str(item) for item in key])


class GurobiSolutionString(ABC):
    @property
    @abstractmethod
    def value(self) -> str:
        """The value of this variable as a whole"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this variable"""
    
    @classmethod
    @abstractmethod
    def type(self) -> str:
        """Type of this variable"""
    
    def __str__(self) -> str:
        return f'{self.name}@{self.type}:\n{self.value}\n'


@dataclass
class GurobiVarString(GurobiSolutionString):
    _name: str
    _value: str

    @property
    def value(self) -> str:
        return f'{self._value}\n'

    @property
    def name(self) -> str:
        return self._name
    
    @classmethod
    @abstractmethod
    def type(self):
        return 'Var'


@dataclass
class GurobiTupleDictString(GurobiSolutionString):
    _name: str
    _keys: List[str]
    _values: List[str]

    def __post_init__(self):
        assert len(self._keys) == len(self._values)

    @property
    def value(self) -> str:
        return '\n'.join([f'[{k}]={v}' for k, v in zip(self._keys, self._values)]) + '\n'

    @property
    def name(self) -> str:
        return self._name
    
    @classmethod
    @abstractmethod
    def type(self):
        return 'TupleDict'


def var_str_from_sol(sol: gp.Var, name: str) -> GurobiVarString:
    return GurobiVarString(name, str(sol.X))


def tupledict_str_from_sol(sol: gp.tupledict, name: str) -> GurobiTupleDictString:
    keys = []
    values = []
    for k in sol.keys():
        keys.append(key_to_str(k))
        values.append(str(sol[k].X))
    return GurobiTupleDictString(name, keys, values)


class GurobiEdgeBasedMinimizeMaximumUtilitySolution(TrafficEngineeringLPSolution):
    """
    This class handles solution outputs from Gurobi.
    We assume the problem at hand is an LP, and the main purpose here is to
    provide enough information for a warm start.
    As such, we assume one of the Simplex methods (in particular, Dual Simplex) is
    used and the model is prepared to accept initial solutions.

    We try to be very generous to Gurobi, and as such, we output basis files (.bas)
    so that Gurobi can very quickly initialize itself.
    For the actual solution data itself, we use the JSON solution output from
    Gurobi, since it is pretty complete in terms of data.

    NOTE: There is benefit in making the output human-readable here. As such, this
          class dumps/loads solutions as JSON instead of pickle.
          The Gurobi `.bas` and JSON files are already human-readable.
    """
    def __init__(self, seed: int, topology_name: str, capacity: float, tm_model_name: str, tm_model_params: Any, 
                 gurobi_sol_path: Optional[str] = None, gurobi_sol_name: Optional[str] = None, runtime: Optional[float] = None):
        assert dataclasses.is_dataclass(tm_model_params)
        if gurobi_sol_path is None:
            assert gurobi_sol_name is not None

        self.seed = seed
        self.topology_name = topology_name
        self.capacity = float(capacity)
        self.tm_model_name = tm_model_name
        self.tm_model_params = tm_model_params
        self.gurobi_sol_path = gurobi_sol_path \
            if gurobi_sol_path is not None \
            else f'$$SOLDIR/{gurobi_sol_name}'
        self.runtime = runtime
    
    def initiate_model(self, model: gp.Model):
        model.reset()
        model.read(self.bas_path)
    
    @property
    def bas_path(self):
        if self.gurobi_sol_path.startswith('$$SOLDIR/'):
            rest = self.gurobi_sol_path.replace('$$SOLDIR/', '')
            return os.path.join(SOLUTION_DIR, f'{rest}.gurobi.bas')
        return f'{self.gurobi_sol_path}.gurobi.bas'
    @property
    def sol_path(self):
        if self.gurobi_sol_path.startswith('$$SOLDIR/'):
            rest = self.gurobi_sol_path.replace('$$SOLDIR/', '')
            return os.path.join(SOLUTION_DIR, f'{rest}.gurobi.json')
        return f'{self.gurobi_sol_path}.gurobi.json'

    def dump(self, model: gp.Model, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        try:
            model.write(self.bas_path)
        except gp.GurobiError:
            print(as_warning("Model did not use Simplex. Cannot write a basis file!"))
        model.write(self.sol_path)
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
    
    def get_vars(self) -> Tuple[np.ndarray, float]:
        with open(self.sol_path) as f:
            d = json.loads(f.read())['Vars']
            graph = load_zoo_topology(name=self.topology_name)
            num_edges = len(graph.edges)
            num_nodes = len(graph.nodes)
            num_commodities = num_nodes * (num_nodes - 1)
            u = None
            assignments = np.zeros((num_edges, num_commodities))
            for item in d:
                name: str = item['VarName']
                if name.startswith('X'):
                    indices_str = name.split('X')[-1].replace('[', '').replace(']', '').split(',')
                    indices = (int(indices_str[0]), int(indices_str[1]))
                    assignments[indices] = item["X"]
                else:
                    assert name == 'U'
                    u = item['X']
        return assignments, u

    def regenerate(self) -> Tuple[nx.DiGraph, TrafficMatrixBase]:
        graph = load_zoo_topology(name=self.topology_name)
        set_edge_capacity_to(graph=graph, capacity=self.capacity)
        tm = get_traffic_model(self.tm_model_name)(seed=self.seed, params=self.tm_model_params)
        return (graph, tm)


class GurobiEdgeBasedMinimizeMaximumUtilityShiftedSolution(GurobiEdgeBasedMinimizeMaximumUtilitySolution):
    def __init__(self, seed, topology_name, capacity, tm_model_name, tm_model_params, 
                 tm_converter_name: str, tm_converter_params: TrafficMatrixConverterBase, 
                 converter_seed: int, iteration: int,
                 gurobi_sol_path = None, gurobi_sol_name = None, runtime = None):
        super().__init__(seed, topology_name, capacity, tm_model_name, tm_model_params, gurobi_sol_path, gurobi_sol_name, runtime)
        self.tm_converter_name = tm_converter_name
        self.tm_converter_params = tm_converter_params
        self.converter_seed = converter_seed
        self.iteration = iteration
    
    def dump(self, model: gp.Model, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        model.write(self.bas_path)
        model.write(self.sol_path)
        with open(path, 'wb') as f:
            d = self.__dict__
            d.update({
                'tm_model_params': dataclasses.asdict(self.tm_model_params),
                'tm_converter_params': dataclasses.asdict(self.tm_converter_params)
            })
            f.write(json.dumps(d, indent=4).encode())

    @classmethod
    def load(cls, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        with open(path, 'rb') as f:
            d = json.loads(f.read().decode())
            d.update({
                'tm_model_params': get_traffic_model_params(d['tm_model_name'])(**d['tm_model_params']),
                'tm_converter_params': get_traffic_converter_params(d['tm_converter_name'])(**d['tm_converter_params'])
            })
            return cls(**d)

    def regenerate(self) -> Tuple[nx.DiGraph, TrafficMatrixBase, TrafficMatrixConverterBase]:
        graph = load_zoo_topology(name=self.topology_name)
        set_edge_capacity_to(graph=graph, capacity=self.capacity)
        tm = get_traffic_model(self.tm_model_name)(seed=self.seed, params=self.tm_model_params)
        converter = get_traffic_converter(self.tm_converter_name)(seed=self.converter_seed, params=self.tm_converter_params)
        for _ in range(self.iteration + 1):
            tm = converter.convert(tm)
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


if __name__ == '__main__':
    from te.traffic_models.models import UniformTrafficMatrixParams
    SEED = 12345
    TOPOLOGY_NAME = 'Interoute'
    TM_MODEL = 'Uniform'
    SOLUTION_NAME = f'{TOPOLOGY_NAME}_{SEED}_{TM_MODEL}.tesol'
    solution = GurobiEdgeBasedMinimizeMaximumUtilitySolution.load(name=SOLUTION_NAME)
    assignments, u = solution.get_vars()
