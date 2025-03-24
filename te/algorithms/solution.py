import os
import json
import dataclasses
import numpy as np
import gurobipy as gp
import networkx as nx
from functools import singledispatch
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Type, Union
from te.algorithms import SOLUTION_DIR
from topologies.utils import load_zoo_topology, set_edge_capacity_to
from utils.logging import as_warning, as_info
from te.traffic_models import get_traffic_model, get_traffic_model_params, get_traffic_converter, get_traffic_converter_params
from te.traffic_models.base import TrafficMatrixBase, TrafficMatrixConverterBase, TrafficMatrixParamsBase
from te.algorithms.base import (
    as_te_solution_name, as_json_solution_name, as_solution_elements_name, as_simplex_basis_name, 
    TrafficEngineeringLPSolution, SolutionElementBase)


END_TOKEN = '----'


def get_name_type_value(string: str) -> Tuple[str, str, str]:
    name, rest = string.split('@', maxsplit=1)
    type, value = rest.split(':\n', maxsplit=1)
    return name, type, value


def str_to_tuple_of_ints(string: str) -> Tuple:
    number_strings = string.replace('[', '').replace(']', '').split(',')
    return tuple([int(num) for num in number_strings])


def line_to_key_value(line: str) -> Tuple[Tuple, float]:
    key_val_string = line.strip().split('=', maxsplit=1)
    key = str_to_tuple_of_ints(key_val_string[0])
    value = float(key_val_string[1])
    return key, value


"""To quickly get solution elements based on the type name"""
_ELEMS: Dict[str, SolutionElementBase] = dict()


def solution_element(cls: Type[SolutionElementBase]) -> SolutionElementBase:
    """Decorator that registers a solution element"""
    global _ELEMS

    assert issubclass(cls, SolutionElementBase)
    # Turn the class into a frozen dataclass
    frozen_cls = dataclass(cls, frozen=True)
    tpe = cls.type()
    assert tpe not in _ELEMS
    _ELEMS[tpe] = frozen_cls

    return frozen_cls


@solution_element
class FloatVarSolutionElement(SolutionElementBase):
    value: float

    @classmethod
    def type(self):
        return 'FloatVar'
    
    @property
    def str_value(self) -> str:
        return str(self.value)
    
    @classmethod
    def parse(cls, string: str):
        name, type, value = get_name_type_value(string)
        assert type == cls.type()
        return cls(name, float(value))


@solution_element
class ArrayVarSolutionElement(SolutionElementBase):
    value: np.ndarray

    @classmethod
    def type(self):
        return 'ArrayVar'
    
    @property
    def str_value(self) -> str:
        return '\n'.join(
            [f'[{",".join([str(item) for item in self.value.shape])}]'] +
            [f'[{",".join([str(item) for item in index])}]={v}' for index, v in np.ndenumerate(self.value)]
        )

    @classmethod
    def parse(cls, string: str):
        name, type, value = get_name_type_value(string)
        assert type == cls.type()
        line_iter = iter(value.split('\n'))
        shape = str_to_tuple_of_ints(next(line_iter).strip())
        array = np.zeros(shape)
        for line in line_iter:
            index, value = line_to_key_value(line)
            array[index] = value
        return cls(name, array)


@solution_element
class GurobiVarSolutionElement(SolutionElementBase):
    value: float

    @classmethod
    def type(self):
        return 'GurobiVar'
    
    @property
    def str_value(self) -> str:
        return str(self.value)
    
    @classmethod
    def parse(cls, string: str):
        name, type, value = get_name_type_value(string)
        assert type == cls.type()
        return cls(name, float(value))


@solution_element
class GurobiTupleDictSolutionElement(SolutionElementBase):
    value: Dict[Tuple, float]
    
    @classmethod
    def type(self):
        return 'GurobiTupleDict'
    
    @property
    def str_value(self) -> str:
        return '\n'.join([f'[{",".join([str(item) for item in k])}]={v}' for k, v in self.value.items()])
    
    @classmethod
    def parse(cls, string: str):
        name, type, value = get_name_type_value(string)
        assert type == cls.type()
        return cls(name, dict([line_to_key_value(line) for line in value.split('\n')]))


@solution_element
class GurobiDualVariableSolutionElement(SolutionElementBase):
    value: float
    
    @classmethod
    def type(self):
        return 'GurobiDualVar'

    @property
    def str_value(self) -> str:
        return str(self.value)

    @classmethod
    def parse(cls, string: str):
        name, type, value = get_name_type_value(string)
        assert type == cls.type()
        return cls(name, float(value))


@solution_element
class GurobiDualVariableListSolutionElement(SolutionElementBase):
    value: List[float]
    
    @classmethod
    def type(self):
        return 'GurobiDualVarList'

    @property
    def str_value(self) -> str:
        return '\n'.join([str(item) for item in self.value])

    @classmethod
    def parse(cls, string: str):
        name, type, value = get_name_type_value(string)
        assert type == cls.type()
        return cls(name, [float(num_str) for num_str in value.split('\n')])


@solution_element
class GurobiDualVariableTupleDictSolutionElement(SolutionElementBase):
    value: Dict[Tuple, float]
    
    @classmethod
    def type(self):
        return 'GurobiDualVarTupleDict'

    @property
    def str_value(self) -> str:
        return '\n'.join([f'[{",".join([str(item) for item in k])}]={v}' for k, v in self.value.items()])

    @classmethod
    def parse(cls, string: str):
        name, type, value = get_name_type_value(string)
        assert type == cls.type()
        return cls(name, dict([line_to_key_value(line) for line in value.split('\n')]))


@singledispatch
def from_sol(sol, _: str) -> SolutionElementBase:
    raise ValueError(f'Unkown solution type: {type(sol)}')


@from_sol.register
def _(sol: gp.Var, name: str) -> GurobiVarSolutionElement:
    return GurobiVarSolutionElement(name, sol.X)


@from_sol.register
def _(sol: float, name: str) -> FloatVarSolutionElement:
    return FloatVarSolutionElement(name, sol)


@from_sol.register
def _(sol: np.ndarray, name: str) -> ArrayVarSolutionElement:
    return ArrayVarSolutionElement(name, sol)


@from_sol.register
def _(sol: gp.tupledict, name: str) -> GurobiTupleDictSolutionElement:
    return GurobiTupleDictSolutionElement(name, {k: sol[k].X for k in sol.keys()})


@from_sol.register
def _(sol: gp.Constr, name: str) -> GurobiDualVariableSolutionElement:
    return GurobiDualVariableSolutionElement(name, sol.Pi)


@from_sol.register(list)
def _(sol: List[gp.Constr], name: str) -> GurobiDualVariableListSolutionElement:
    return GurobiDualVariableListSolutionElement(name, [c.Pi for c in sol])


@from_sol.register(dict)
def _(sol: Dict[Tuple, gp.Constr], name: str) -> GurobiDualVariableTupleDictSolutionElement:
    return GurobiDualVariableTupleDictSolutionElement(name, {k: v.Pi for k, v in sol.items()})


@dataclass
class EdgeBasedMinimizeMaximumUtilitySolutionParams:
    seed: int
    topology_name: str
    capacity: float
    tm_model_name: str
    tm_model_params: TrafficMatrixParamsBase
    path: Optional[str] = None
    sol_name: Optional[str] = None
    runtime: Optional[float] = None


class EdgeBasedMinimizeMaximumUtilitySolution(TrafficEngineeringLPSolution):
    """
    This class handles solution outputs, with some special functions catering especially
    to outputs for Gurobi so that it can be warm started efficiently (to this end, 
    we mainly assume one of the Simplex methods, in particular Dual Simplex, was
    used to generate the solution).

    We try to be very generous to Gurobi, and as such, we output basis files (.bas)
    so that Gurobi can very quickly initialize itself.
    For the actual solution data itself, we use the JSON solution output from
    Gurobi, since it is pretty complete in terms of data.

    NOTE: There is benefit in making the output human-readable here. As such, this
          class dumps/loads solutions as JSON instead of pickle.
          The Gurobi `.bas` and JSON files are already human-readable.
    
    Solution from non-Gurobi sources (like our algorithm) are instead formatted as a
    `.elems` file, where each line holds information about a designated solution value.
    """
    def __init__(self, params: EdgeBasedMinimizeMaximumUtilitySolutionParams):
        assert dataclasses.is_dataclass(params.tm_model_params)
        if params.path is None:
            assert params.sol_name is not None

        self.seed = params.seed
        self.topology_name = params.topology_name
        self.capacity = float(params.capacity)
        self.tm_model_name = params.tm_model_name
        self.tm_model_params = params.tm_model_params
        self.path = params.path \
            if params.path is not None \
            else f'$$SOLDIR/{params.sol_name}'
        self.runtime = params.runtime
        self.solution_elements: List[SolutionElementBase] = []
    
    def add_solution_element(self, element, name: str):
        self.solution_elements.append(from_sol(element, name))
    
    @property
    def bas_path(self):
        if self.path.startswith('$$SOLDIR/'):
            rest = self.path.replace('$$SOLDIR/', '')
            name = os.path.join(SOLUTION_DIR, rest)
        else:
            name = self.path
        return as_simplex_basis_name(name)
    @property
    def sol_path(self):
        if self.path.startswith('$$SOLDIR/'):
            rest = self.path.replace('$$SOLDIR/', '')
            name = os.path.join(SOLUTION_DIR, rest)
        else:
            name = self.path
        return as_json_solution_name(name)
    @property
    def elem_path(self):
        if self.path.startswith('$$SOLDIR/'):
            rest = self.path.replace('$$SOLDIR/', '')
            name = os.path.join(SOLUTION_DIR, rest)
        else:
            name = self.path
        return as_solution_elements_name(name)

    def dump(self, name: str, path: str = None):
        path = as_te_solution_name(path if path is not None else os.path.join(SOLUTION_DIR, name))
        with open(path, 'wb') as f:
            d = self.__dict__
            d.pop('solution_elements')
            d.update({
                'tm_model_params': dataclasses.asdict(self.tm_model_params)
            })
            f.write(json.dumps(d, indent=4).encode())
    
    def dump_basis(self, model: gp.Model):
        try:
            model.write(self.bas_path)
            print(as_info("Wrote out Simplex basis"))
        except gp.GurobiError:
            print(as_warning("Model did not use Simplex. Cannot write a basis file!"))
    
    def dump_json(self, model: gp.Model):
        model.write(self.sol_path)
        print(as_info("Wrote out JSON solution"))
    
    def dump_elements(self):
        if len(self.solution_elements) == 0:
            print(as_warning("No solution elements were provided!"))
        else:
            with open(self.elem_path, 'w') as f:
                sep = '\n' + END_TOKEN + '\n'
                f.write(sep.join([str(elem) for elem in self.solution_elements]) + sep)
            print(as_info("Wrote out solution elements"))
    
    @classmethod
    def load(cls, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        with open(path, 'rb') as f:
            d: Dict = json.loads(f.read().decode())
            d.update({
                'tm_model_params': get_traffic_model_params(d['tm_model_name'])(**d['tm_model_params'])
            })
            return cls(EdgeBasedMinimizeMaximumUtilitySolutionParams(**d))

    def initiate_model_from_basis(self, model: gp.Model):
        model.reset()
        model.read(self.bas_path)
    
    def get_vars_from_json(self) -> Tuple[np.ndarray, float]:
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
    
    def _load_element_strings(self) -> List[str]:
        holder = ''
        element_strings = []
        with open(self.elem_path) as f:
            for line in f:
                if END_TOKEN in line:
                    element_strings.append(holder.strip())
                    holder = ''
                else:
                    holder += line
        return element_strings
    
    @staticmethod
    def _parse_element_strings(element_strings: List[str]) -> List[SolutionElementBase]:
        out = []
        for elem_string in element_strings:
            _, type, _ = get_name_type_value(elem_string)
            cls = _ELEMS[type]
            out.append(cls.parse(elem_string))
        return out

    def load_solution_elements(self):
        self.solution_elements = self._parse_element_strings(self._load_element_strings())

    def regenerate(self) -> Tuple[nx.DiGraph, TrafficMatrixBase]:
        graph = load_zoo_topology(name=self.topology_name)
        set_edge_capacity_to(graph=graph, capacity=self.capacity)
        tm = get_traffic_model(self.tm_model_name)(seed=self.seed, params=self.tm_model_params)
        return (graph, tm)

    def get_solution_element_by_name(self, name: str) -> SolutionElementBase:
        if len(self.solution_elements) == 0:
            self.load_solution_elements()
        for elem in self.solution_elements:
            if elem.name == name:
                return elem
        raise ValueError


@dataclass
class EdgeBasedMinimizeMaximumUtilityShiftedSolutionParams:
    seed: int
    topology_name: str
    capacity: float
    tm_model_name: str
    tm_model_params: TrafficMatrixParamsBase
    tm_converter_name: str
    tm_converter_params: TrafficMatrixConverterBase
    converter_seed: int
    iteration: int
    path: Optional[str] = None
    sol_name: Optional[str] = None
    runtime: Optional[float] = None


class EdgeBasedMinimizeMaximumUtilityShiftedSolution(EdgeBasedMinimizeMaximumUtilitySolution):
    def __init__(self, params: EdgeBasedMinimizeMaximumUtilityShiftedSolutionParams):
        super().__init__(EdgeBasedMinimizeMaximumUtilitySolutionParams(
            params.seed, params.topology_name, params.capacity, params.tm_model_name,
            params.tm_model_params, params.path, params.sol_name, params.runtime
        ))
        self.tm_converter_name = params.tm_converter_name
        self.tm_converter_params = params.tm_converter_params
        self.converter_seed = params.converter_seed
        self.iteration = params.iteration
    
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
    
    def get_solution_element_by_name(self, name):
        if len(self.solution_elements) == 0:
            self.load_solution_elements()
        for elem in self.solution_elements:
            if elem.name == name:
                return elem
        raise ValueError


def tuple_dict_to_np_array(element: Union[GurobiTupleDictSolutionElement, GurobiDualVariableTupleDictSolutionElement]) -> np.ndarray:
    tpdict = element.value
    keys_iter = iter(tpdict)
    shape = next(keys_iter)
    while True:
        try:
            key = next(keys_iter)
            shape = tuple(max(a, b) for a, b in zip(shape, key))
        except StopIteration:
            shape = tuple(item+1 for item in shape)
            break
    out = np.zeros(shape=shape)
    for k, v in tpdict.items():
        out[k] = v
    return out


def default_solution_name(topology_name: str, rng_seed: int, tm_type: str, postfix: Optional[str] = None, **gurobi_kwargs) -> str:
    items = [topology_name, str(rng_seed), tm_type]
    method = gurobi_kwargs.get('method')
    if method is not None:
        if isinstance(method, int):
            if method == gp.GRB.METHOD_PRIMAL or method == gp.GRB.METHOD_DUAL:
                items.append('Simplex')
            elif method == gp.GRB.METHOD_BARRIER:
                items.append('Barrier')
            else:
                raise ValueError(f'Unexpected method: {method}')
        elif isinstance(method, str):
            method = method.lower()
            if method == 'primal' or method == 'dual' or method == 'simplex':
                items.append('Simplex')
            elif method == 'barrier':
                items.append('Barrier')
            else:
                raise ValueError(f'Unexpected method: {method}')
        else:
            raise ValueError(f'Unexpected method: {method}')
    crossover = gurobi_kwargs.get('crossover')
    if crossover is not None:
        if crossover is True and method == gp.GRB.METHOD_BARRIER:
            items.append('Crossover')
    if postfix is not None:
        items.append(postfix)
    return '_'.join(items)



if __name__ == '__main__':
    SEED = 12345
    TOPOLOGY_NAME = 'Claranet'
    TM_MODEL = 'Uniform'
    SOLUTION_NAME = f'{TOPOLOGY_NAME}_{SEED}_{TM_MODEL}.tesol'
    solution = EdgeBasedMinimizeMaximumUtilitySolution.load(name=SOLUTION_NAME)
    assignments, u = solution.get_vars_from_json()
    solution.load_solution_elements()
    print(solution.solution_elements)
