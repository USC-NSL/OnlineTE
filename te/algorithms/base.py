import os
import enum
import pickle
import inspect
import argparse
import numpy as np
import networkx as nx
import te.constants
import dataclasses
import jsonargparse
import matplotlib.pyplot as plt
from itertools import count
from typing import List, Optional, Tuple, Dict, Union, Any, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from te.algorithms import SOLUTION_DIR
from utils.logging import LINE_SEPARATOR_LENGTH, as_success, as_fail, as_warning
from te.traffic_models.base import TrafficMatrixBase, TrafficMatrixConverterBase, TrafficMatrixConverterParamsBase, Commodity


TE_SOLUTION_POSTFIX = '.tesol'
SOLUTION_ELEMENTS_POSTFIX = '.elems'
JSON_SOLUTION_POSTFIX = '.json'
SIMPLEX_BASIS_POSTFIX = '.bas'


def with_postfix(name: str, postfix: str) -> str:
    if not name.endswith(postfix):
        return name + postfix
    return name

as_te_solution_name = lambda name: with_postfix(name, TE_SOLUTION_POSTFIX)
as_solution_elements_name = lambda name: with_postfix(name, SOLUTION_ELEMENTS_POSTFIX)
as_json_solution_name = lambda name: with_postfix(name, JSON_SOLUTION_POSTFIX)
as_simplex_basis_name = lambda name: with_postfix(name, SIMPLEX_BASIS_POSTFIX)


class TEObjective(str, enum.Enum):
    """Describes the objective that we want to solve for"""
    MLU = "MLU"
    MAX_FLOW = "Max-Flow"
    MAX_CONCURRENT_FLOW = "Max-Concurrent-Flow"


class SolverParams(ABC):
    """
    ABC for any set of solver/evaluation parameters that we want to bundle
    together and make known to the user.
    By default, it supports a pretty-print such that a table is shown for
    a given object when stringified.

    Every inheritence of this class, adds an extra `depth` to it.
    Depth 0 is always the parameters introduced by the latest inheritence, and
    inner depths go into the fields inherited by the parents.

    For examples:
    ```
    @dataclass
    class A(SolverParams):
        field1: str
    
    @dataclass
    class B(SolverParams):
        field2: str
    
    B_obj = B('a', 'b')
    print(B_obj.child_fields)               # returns `{'field2': 'a'}
    print(B_obj.get_fields_up_to_level(1))  # returns `{'field2': 'a', 'field1': 'b'}
    ```

    TODO: Force this to always check if we are implementing a `dataclass`.
    """
    PRINT_FORMAT = "| {:^{left_padding}} | {:^{right_padding}} |"

    @property
    def left_column_share(self) -> float:
        return self._left_column_share
    @left_column_share.setter
    def left_column_share(self, value: float):
        assert (value > 0) and (value < 1)
        self._left_column_share = value
    @property
    def left_column_padding(self) -> int:
        return int(self.left_column_share * (LINE_SEPARATOR_LENGTH - 5))
    @property
    def right_column_padding(self) -> int:
        return LINE_SEPARATOR_LENGTH - 7 - self.left_column_padding
    @property
    def line_padding(self) -> int:
        return LINE_SEPARATOR_LENGTH - 2
    @property
    def line(self) -> str:
        return "+" + "-"*self.line_padding + "+"

    @classmethod
    def field_names(cls) -> List[str]:
        if cls.__base__ == ABC:
            return []
        return [item.name for item in dataclasses.fields(cls)]
    
    @property
    def child_fields(self) -> Dict[str, Any]:
        return self.get_fields_up_to_level(0)
    
    def get_fields_up_to_level(self, level: int):
        ancestor_class = self.__class__
        if level < 0:
            it = count()
        else:
            it = range(level+1)
        for i in it:
            ancestor_class = ancestor_class.__base__
            if ancestor_class == ABC:
                ancestor_fields = []
                break
            else:
                assert issubclass(ancestor_class, SolverParams)
            if i == level:
                ancestor_fields = ancestor_class.field_names()
        child_dict = self.__dict__.copy()
        for key in ancestor_fields:
            child_dict.pop(key)
        keys = list(child_dict.keys())
        for key in keys:
            if key.startswith('_'):
                child_dict.pop(key, None)
        return child_dict
    
    @classmethod
    def _list_or_tuple_to_str(cls, items: Union[List, Tuple]) -> Union[str, List[str]]:
        if len(items) == 0:
            if isinstance(items, list):
                return '[]'
            else:
                return '()'
        example = items[0]
        if isinstance(example, (int, float, bool, str)):
            # Multiple things on a single line ...
            return ', '.join([str(item) for item in items])
        elif isinstance(example, tuple):
            # Sequence of tuples ...
            return [f'({", ".join([str(e) for e in item])})' for item in items]
        else:
            raise ValueError(f'Sequence element type unexpected: {example}')
    
    @classmethod
    def _param_to_str(cls, value) -> Union[str, List[str]]:
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f'{value:.2e}'
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            return value
        elif isinstance(value, (tuple, list)):
            return cls._list_or_tuple_to_str(value)
        elif value is None:
            return "None"
        elif dataclasses.is_dataclass(value):
            return f"<{value.__class__.__name__}>"
        elif inspect.isclass(value):
            return f"[{value.__class__.__name__}]"
        elif isinstance(value, enum.Enum):
            return str(value)
        raise ValueError(f'Unexpected instance: {type(value)}')
    
    def _field_to_string(self, key: str, value: Any):
        # print(f"Looking at key: {key} with value {value} and type {type(value)}")
        value_str = self._param_to_str(value)
        if isinstance(value_str, str):
            return self.PRINT_FORMAT.format(
                key, value_str,
                left_padding=self.left_column_padding,
                right_padding=self.right_column_padding
            )
        elif isinstance(value_str, list):
            result = '\n'.join(
                [self.PRINT_FORMAT.format(
                    key, value_str[0],
                    left_padding=self.left_column_padding,
                    right_padding=self.right_column_padding)] + 
                [self.PRINT_FORMAT.format(
                    '.', value_line,
                    left_padding=self.left_column_padding,
                    right_padding=self.right_column_padding
                ) for value_line in value_str[1:]]
            )
            return result

    def __str__(self) -> str:
        return self.stringify_up_to_level(0)
    
    def stringify_up_to_level(self, level: int) -> str:
        return '\n'.join(
            [self.line] +
            [self._field_to_string(key, value)
                for key, value in self.get_fields_up_to_level(level).items()] +
            [self.line]
        )
    
    def str_all(self) -> str:
        return self.stringify_up_to_level(-1)
    
    @classmethod
    def make_from_args(cls, namespace: Union[argparse.Namespace, jsonargparse.Namespace]):
        params = dict()
        for name in cls.field_names():
            if name in namespace:
                params[name] = namespace[name]
        return cls(**params)


@dataclass(frozen=True)
class SolutionElementBase(ABC):
    """Generic contrainer for solution elements"""
    name: str      # Variable(s) name
    value: Any     # Variable(s) value

    @classmethod
    @abstractmethod
    def type(self) -> str:
        """Type of this variable"""
    
    @property
    @abstractmethod
    def str_value(self) -> str:
        """Variable value to a string"""

    @classmethod
    @abstractmethod
    def parse(cls, string: str):
        """Parse a string into an instance of this class"""
    
    def __str__(self) -> str:
        return f'{self.name}@{self.type()}:\n{self.str_value}'


@dataclass
class TrafficEngineeringLPEvaluationParams(SolverParams):
    """
    All generic evaluation parameters go into this class.

    Attributes
    ---------
    TopologyName: str
        Name of the topology that we are solving on.
        The name is just for logging.
    Seed: int
        Any RNG will be initialized to this seed to make the
        evaluations reproducible.
    Objective: TEObjective
        The particular TE objective we want to solve for.
        Defaults to MLU.
    ScaleFactor: float
        When a problem can become infeasible because of capacity
        constraints, this value can be used to inflate capacity
        values by a set amount to prevent that.
        By default, a value of `10.0` is used, but needs to be
        adjusted for each topology.
    FloatResolution: float
        A pair of floating point numbers that are less than this 
        value apart are treated as the same. May also be used as
        an `epsilon` tolerance when numerical problems may arise
        (e.g. when a divide by zero is likely).
    FeasibilityTolerance: float
        Absolute contraint violation tolerance. Will override any
        other values if needed (including the one from Gurobi).
    FeasibilityRatio: float
        Relative objective _AND_ constraint violation. 
    PrintReports: bool = `False`
        Essentially a `verbose` flag. Prints the detailed report of
        what we are doing and detailed description of constraint or
        objective violations.
    ShowPLT: bool = `False`
        Pop up a window for `PLT` plots in the end.
    SavePLT: bool = `True`
        Save any `PLT` plots generated.
    SaveSol: bool = `False`
        Save the final solution by calling the `save_sol` method
        of the problem.
        (Note that the output can be very large).
    TraceOutputPath: Optional[str] = `'res.txt'`
        A simple text file, containing the objective value trace, if the
        problem class actually implemented and provided it.
    PLTOutputPath: Optional[str] = `'res.png'`
        Path to the output file for all `PLT` plots.
    
    Note
    ----
    For now, exactly one of `FeasibilityTolerance` or `FeasibilityRatio` must
    be given.
    """
    TopologyName: str
    Seed: int
    Objective: TEObjective = TEObjective.MLU
    ScaleFactor: float = 10.0
    FloatResolution: float = te.constants.FLOAT_RES
    FeasibilityTolerance: Optional[float] = None
    FeasibilityRatio: Optional[float] = None
    PrintReports: bool = False
    ShowPLT: bool = False
    SavePLT: bool = True
    SaveSol: bool = False
    TraceOutputPath: Optional[str] = 'res.txt'
    PLTOutputPath: Optional[str] = 'res.png'

    def __post_init__(self):
        # UPDATE: We will prevent both tolerances being given, it makes reasoning about things difficult ...
        assert (self.FeasibilityRatio is None) ^ (self.FeasibilityTolerance is None), "Exactly one of `FeasibilityTolerance` or `FeasibilityRatio` MUST be given"
        self.left_column_share = 0.5
    
    def __call__(self, **kwargs):
        copy = dataclasses.replace(self)
        for k, v in kwargs.items:
            setattr(copy, k, v)
    
    @property
    def is_mlu(self) -> bool:
        return self.Objective == TEObjective.MLU


@dataclass
class TrafficEngineeringLPWarmStartParams(SolverParams):
    """
    A dataclass for keeping data about TM converters for warm-starting.

    Attributes
    ----------
    ConverterSeed: int
        The RNG seed that _may_ be used by the TM converter
    WarmIters: int
        Number of warm-start iterations. A warm-start iteration includes a
        converstion of the current TM and solving the problem again
    ConverterParams: type[TrafficMatrixConverterParamsBase]
        Parameters to pass to the TM converter
    """
    ConverterSeed: int
    ConverterParams: type[TrafficMatrixConverterParamsBase]
    WarmIters: int

    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class TrafficEngineeringLPSolutionParams(SolverParams):
    """
    A dataclass for keeping data about solutions.

    Attributes
    ----------
    Name: str
        The name _prefix_ of the solution file.
    Path: Optional[str]
        The output path for the solution.
    """
    Name: str
    Path: Optional[str]

    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class TrafficEngineeringLPCheckResult:
    """
    A simple class for reporting the quality of a TE solution after it was
    checked for violations.

    Attributes
    ----------
    unsat_ratio: float
        Ratio of unsatisfied demands.
    congested_ratio: float
        Ratio of links that are congested.
    unsat_commodities: Set[int]
        A set of commodity indices that are unsatisfied.
    congested_links: Set[int]
        Set of link (edge) indices that are congested.
    density: Optional[float]
        Final solution density
    """
    unsat_ratio: float
    congested_ratio: float
    unsat_commodities: Set[int]
    congested_links: Set[int]
    density: Optional[float] = None

    def __str__(self) -> str:
        out = []
        if len(self.unsat_commodities) == 0:
            out.append(as_success("ALL DEMANDS WERE SATISFIED"))
        else:
            out.append(as_fail("{:.1f}% OF DEMANDS WERE NOT SATISFIED".format(self.unsat_ratio*100)))
        if len(self.congested_links) == 0:
            out.append(as_success("ALL LINK CAPCITIES WERE HONORED"))
        else:
            out.append(as_fail("{:.1f}% OF LINKS ARE CONGESTED".format(self.congested_ratio*100)))
        if self.density is not None:
            if self.density > 0.5:
                out.append(as_warning("DENSITY: {:.1f}%".format(self.density*100)))
            else:
                out.append(as_success("DENSITY: {:.1f}%".format(self.density*100)))
        return '\n'.join(out)


class TrafficEngineeringLPObjectiveTrace:
    def __init__(self, names: List[str]):
        self._names = names
        self._trace: List[Tuple[float]] = []
        self._n_dict = {name: i for i, name in enumerate(names)}
    
    @property
    def names(self) -> List[str]:
        return self._names
    @property
    def trace(self) -> List[Tuple[float]]:
        return self._trace

    def append(self, *args, **kwargs):
        ls = [None for _ in range(len(self.names))]
        for i, arg in enumerate(args):
            ls[i] = arg
        for k, v in kwargs.items():
            i = self._n_dict[k]
            assert ls[i] is None
            ls[i] = v
        self._trace.append(tuple(ls))
    
    def unravel(self) -> List[List[float]]:
        return list(zip(*self.trace))
    
    def plot(self, **kwargs):
        for trace in list(zip(*self.trace)):
            plt.plot(trace, **kwargs)
        plt.legend(self.names)


class TrafficEngineeringLPSolution(ABC):
    @abstractmethod
    def __init__(self, params: Optional[TrafficEngineeringLPSolutionParams] = None):
        super().__init__()
        self._params = params

    @property
    def params(self) -> TrafficEngineeringLPSolutionParams:
        return self._params

    def dump(self, name: Optional[str] = None, path: Optional[str] = None):
        name = name if name is not None else self._params.Name
        # TODO: This does not seem right!
        # path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        path = path if path is not None else self._params.Path
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @abstractmethod
    def regenerate(self) -> Tuple[nx.DiGraph, TrafficMatrixBase]:
        """
        Regenerate the graph and traffic matrix associated with this solution
        """
    
    @abstractmethod
    def add_solution_element(self, element, name: str):
        """Add a solution element to this TE solution instance"""
    
    @abstractmethod
    def get_solution_element_by_name(self, name: str) -> SolutionElementBase:
        """Get a solution element by name"""
    
    @abstractmethod
    def dump_elements(self):
        """Dump solution elements one-by-one"""


@dataclass
class TrafficEngineeringProblemDescription:
    """
    A full description of a TE problem and its required outputs.
    This can be passed to any TE solver to get the full description of the 
    problem; other parameters need only describe the solver properties.

    Attributes
    ----------
    EvalParams: TrafficEngineeringLPEvaluationParams
        Evaluation parameters, instructing the LP class about how optimal
        the solution need be and how much infeasibility are we willing to
        tolerate.
    Graph: nx.DiGraph
        The topology as a directed graph. Each edge must have a `capacity`
        attribute as a floating point number.'
    TM: TrafficMatrixBase
        The traffic matrix.
    Converter: Optional[TrafficMatrixConverterBase] = None
        The traffic matrix converter that can change the current traffic
        matrix into a new one to see if we can handle incremental problems.
    WarmStartParams: Optional[TrafficEngineeringLPWarmStartParams] = None
        Warm start parameters (e.g. how many TM conversion rounds must
        be done).
    Solution: Optional[TrafficEngineeringLPSolution] = None
        The TE solution object used to add and save solution elements.
    """
    EvalParams: TrafficEngineeringLPEvaluationParams
    Graph: nx.DiGraph
    TM: TrafficMatrixBase
    Converter: Optional[TrafficMatrixConverterBase] = None
    WarmStartParams: Optional[TrafficEngineeringLPWarmStartParams] = None
    Solution: Optional[TrafficEngineeringLPSolution] = None

    @property
    def is_mlu(self) -> bool:
        return self.EvalParams.is_mlu


class TrafficEngineeringLP(ABC):
    @abstractmethod
    def __init__(self, problem_description: TrafficEngineeringProblemDescription, 
                 solver_params: SolverParams, **kwargs):
        super().__init__(**kwargs)
        self._problem_description = problem_description
        self._solver_params = solver_params

    @property
    def problem_description(self) -> TrafficEngineeringProblemDescription:
        """TE problem description object"""
        return self._problem_description

    @property
    @abstractmethod
    def alg_name(cls) -> str:
        """Name of this algorithm"""

    @property
    @abstractmethod
    def graph(self) -> nx.DiGraph:
        """The graph of the network"""

    @property
    @abstractmethod
    def traffic(self) -> TrafficMatrixBase:
        """The traffic matrix input"""

    @property
    @abstractmethod
    def commodity_list(self) -> List[Commodity]:
        """List of input commodities"""

    @property
    def solver_params(self) -> SolverParams:
        """Solver parameters"""
        return self._solver_params

    @property
    @abstractmethod
    def objective_value(self) -> float:
        """Final objective value after optimization"""

    @property
    @abstractmethod
    def objective_trace(self) -> Optional[TrafficEngineeringLPObjectiveTrace]:
        """List of objective values during algorithm iterations"""

    @property
    def objective_gap_trace(self) -> Optional[List[float]]:
        """List of primal-dual objective gap during iterations"""
        return None
    
    @property
    @abstractmethod
    def assignments(self) -> np.ndarray:
        """Return current assignments based on the solution"""
    
    @property
    def check_result(self) -> TrafficEngineeringLPCheckResult:
        """
        Checking the LP result usually takes time. So every time `check` has been called, the
        result is cached in this property.
        Invoking this method on an unsolved LP will raise a ValueError
        """
        if not hasattr(self, '_check_result') or self._check_result is None:
            raise ValueError
        return self._check_result
    
    @check_result.setter
    def check_result(self, _res: TrafficEngineeringLPCheckResult):
        self._check_result = _res

    @abstractmethod
    def initialize_to(self, solution: TrafficEngineeringLPSolution):
        """
        Initialize the model to a particular solution
        """
    
    @abstractmethod
    def _make_variables(self, *args, **kwargs):
        """Add required variables to the problem model"""

    @abstractmethod
    def _add_constraints(self, *args, **kwargs):
        """Add all constraints needed to the problem model"""

    @abstractmethod
    def _add_objective(self, *args, **kwargs):
        """Add the objective function to the problem model"""

    @abstractmethod
    def make_lp(self, *args, **kwargs):
        """Create the LP and the Gurobi model object"""

    @abstractmethod
    def close(self):
        """Free and cleanup environment"""

    @abstractmethod
    def reset(self, with_params: False):
        """
        Reset solver as if no iterations were done.
        Optionally, may also reset parameters to their default values.
        """

    @abstractmethod
    def solve(self, params: SolverParams = None) -> float:
        """
        Solve the problem. May also accept an extra set of parameters,
        which will clear and override any existing ones using `reset`.

        return the runtime of the model, or `-1` if it failed.
        """

    @abstractmethod
    def check(self):
        """
        Performs sanity checks on the current solution and cache the summary of the checks.
        This result should be stored in the `check_result` property.
        """

    @abstractmethod
    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        """
        Get commodity allocations of the final solution
        """
    
    @abstractmethod
    def update_traffic_matrix(self, tm: TrafficMatrixBase):
        """
        Update the current traffic matrix and re-initialize the model
        """
    
    @abstractmethod
    def add_solution_elements(self, solution: TrafficEngineeringLPSolution):
        """
        Add solution elements to a given TE solution instance
        """

    def add_and_dump_lp_solutions(self, solution: TrafficEngineeringLPSolution):
        self.add_solution_elements()
        solution.dump_elements()
        solution.dump()


__all__ = [
    'SolverParams', 'TrafficEngineeringLPEvaluationParams', 'TEObjective',
    'TrafficEngineeringLPWarmStartParams', 'TrafficEngineeringLPSolutionParams',
    'TrafficEngineeringLPCheckResult', 'TrafficEngineeringLPSolution',
    'TrafficEngineeringProblemDescription', 'TrafficEngineeringLP',
    'TrafficEngineeringLPObjectiveTrace'
]