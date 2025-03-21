import os
import pickle
import numpy as np
import networkx as nx
import te.constants
import dataclasses
from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from te.algorithms import SOLUTION_DIR
from te.traffic_models.base import TrafficMatrixBase, Commodity


TE_SOLUTION_POSTFIX = '.tesol'
SOLUTION_ELEMENTS_POSTFIX = '.gurobi.elems'
JSON_SOLUTION_POSTFIX = '.gurobi.json'
SIMPLEX_BASIS_POSTFIX = '.gurobi.bas'


def with_postfix(name: str, postfix: str) -> str:
    if not name.endswith(postfix):
        return name + postfix
    return name

as_te_solution_name = lambda name: with_postfix(name, TE_SOLUTION_POSTFIX)
as_solution_elements_name = lambda name: with_postfix(name, SOLUTION_ELEMENTS_POSTFIX)
as_json_solution_name = lambda name: with_postfix(name, JSON_SOLUTION_POSTFIX)
as_simplex_basis_name = lambda name: with_postfix(name, SIMPLEX_BASIS_POSTFIX)


class SolverParams(ABC):
    LINE = "+" + "-"*65 + "+"
    PRINT_FORMAT = "| {:^30} : {:^30} |"

    @classmethod
    def field_names(cls) -> List[str]:
        if cls.__base__ == ABC:
            return []
        return [item.name for item in dataclasses.fields(cls)]
    
    @property
    def child_fields(self) -> Dict[str, Any]:
        parent_class = self.__class__.__base__
        if parent_class == ABC.__class__:
            parent_fields = []
        else:
            assert issubclass(parent_class, SolverParams)
            parent_fields = parent_class.field_names()
        child_dict = dataclasses.asdict(self)
        for key in parent_fields:
            child_dict.pop(key)
        return child_dict
    
    def _num_to_str(self, value) -> str:
        if isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return f'{value:.2e}'
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, str):
            return value
        elif value is None:
            return "None"
        raise ValueError(f'Unexpected instance: {type(value)}')

    def __str__(self) -> str:
        return '\n'.join([self.LINE]+ [
            self.PRINT_FORMAT.format(key, self._num_to_str(value))
            for key, value in self.child_fields.items()
        ] + [self.LINE])


@dataclass(frozen=True)
class GurobiSolutionElementBase(ABC):
    """Generic contrainer for Gurobi solution elements"""
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
    def get_gurobi_solution_element_by_name(self, name: str) -> GurobiSolutionElementBase:
        """Get a Gurobi solution element by name"""


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
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        """
        Performs sanity checks on the current solution
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
