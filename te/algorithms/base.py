import os
import pickle
import numpy as np
import networkx as nx
import te.constants
import dataclasses
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Dict, Union, Any, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from te.algorithms import SOLUTION_DIR
from utils.logging import LINE_SEPARATOR_LENGTH, as_success, as_fail
from te.traffic_models.base import TrafficMatrixBase, Commodity


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


class SolverParams(ABC):
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
        for i in range(level+1):
            ancestor_class = ancestor_class.__base__
            if ancestor_class == ABC.__class__:
                ancestor_fields = []
                break
            else:
                assert issubclass(ancestor_class, SolverParams)
            if i == level:
                ancestor_fields = ancestor_class.field_names()
        child_dict = dataclasses.asdict(self)
        for key in ancestor_fields:
            child_dict.pop(key)
        return child_dict
    
    @classmethod
    def _list_or_tuple_to_str(cls, items: Union[List, Tuple]) -> Union[str, List[str]]:
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
        raise ValueError(f'Unexpected instance: {type(value)}')
    
    def _field_to_string(self, key: str, value: Any):
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
    
    def stringify_up_to_level(self, level: int):
        return '\n'.join(
            [self.line] +
            [self._field_to_string(key, value)
                for key, value in self.get_fields_up_to_level(level).items()] +
            [self.line]
        )


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
class GurobiSolverParams(SolverParams):
    Method: int = te.constants.DEFAULT_SOLVER_METHOD
    Crossover: int = te.constants.DEFAULT_CROSSOVER
    NumericFocus: int = te.constants.DEFAULT_NUMERIC_FOCUS
    ConvTol: float = te.constants.DEFAULT_OPTIMALITY_TOLERANCE
    FeasibilityTol: float = te.constants.DEFAULT_FEASIBILITY_TOLERANCE
    Presolve: int = te.constants.DEFAULT_PRESOLVE
    Threads: int = min(cpu_count(), 32)
    LogFile: str = te.constants.DEFAULT_GUROBI_LOG_FILE

    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class TrafficEngineeringLPCheckResult:
    unsat_ratio: float
    congested_ratio: float
    unsat_commodities: Set[int]
    congested_links: Set[int]

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
        return '\n'.join(out)


class TrafficEngineeringLPSolution(ABC):
    def dump(self, name: str, path: str = None):
        path = path if path is not None else os.path.join(SOLUTION_DIR, name)
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


class TrafficEngineeringLP(ABC):
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
    @abstractmethod
    def params(self) -> SolverParams:
        """Solver parameters"""

    @property
    @abstractmethod
    def objective_value(self) -> float:
        """Final objective value after optimization"""

    @property
    @abstractmethod
    def objective_trace(self) -> Optional[List]:
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
    # @abstractmethod
    def check_result(self) -> TrafficEngineeringLPCheckResult:
        """
        Checking the LP result usually takes time. So every time `check` has been called, the
        result is cached in this property.
        Invoking this method on an unsolved LP _MUST_ raise a ValueError
        """

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
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None, 
              report: bool = False, **kwargs):
        """
        Performs sanity checks on the current solution and cache the summary of the checks.
        This result should be stored in the `check_result` property.
        Violations during each check should be reported if `report` has been set to `True`.
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
