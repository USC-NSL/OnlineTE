from abc import ABC, abstractmethod
from te.algorithms.base import SolverParams, TEObjective
from te.algorithms.array_utils.cpu_utils import CPUArray, DoublePrecisionCPUArray


class ControllerMLUException(Exception):
    """Base class for MLU solver exceptions ..."""
    def __init__(self, solver_name: str, solver_exception):
        self._solver_name = solver_name
        self._solver_exception = solver_exception
    
    def __str__(self):
        return f"Solver {self._solver_name} raised exception:\n{self._solver_exception}"


class ControllerMLUSolver(ABC):
    @classmethod
    @abstractmethod
    def name(self) -> str:
        """Name of this MLU backend"""

    @property
    @abstractmethod
    def num_edges(self) -> int:
        """Number of edges in the network"""
    @property
    @abstractmethod
    def objective_type(self) -> TEObjective:
        """The objective type being solved"""
    @property
    @abstractmethod
    def capacities(self) -> DoublePrecisionCPUArray:
        """Capacity vector"""
    @property
    @abstractmethod
    def solver_params(self) -> SolverParams:
        """MLU Solver Parameters"""
    @property
    @abstractmethod
    def num_domains(self) -> int:
        """Number of domains"""
    @property
    @abstractmethod
    def is_solved(self) -> bool:
        """Check if the model has been solved to completion given the curren `F` vector"""
    
    @abstractmethod
    def _make_variables(self):
        """Add `Z` and `u` variables (called only ONCE unless `reset`)"""
    @abstractmethod
    def _add_constraints(self):
        """Add capacity and bound constraint for utilization (called only ONCE unless `reset`)"""
    @abstractmethod
    def _add_objective(self):
        """Create the objective (called only ONCE unless `reset`)"""
    
    @property
    @abstractmethod
    def current_u(self) -> float:
        """(MLU ONLY) Return the current optimal utilization value"""
    @property
    @abstractmethod
    def current_Z(self) -> CPUArray:
        """Return the current optimal aggregation value over each link"""
    
    @abstractmethod
    def close(self):
        """Close the solver completely"""
    @abstractmethod
    def reset(self, with_params: False):
        """Reset the solver to a blank state"""
    @abstractmethod
    def update_F_m(self, F_m: CPUArray):
        """
        Update the current `F^{(m)}` value (i.e. `sum X_k + r` for iteration `m`) AND the objective.
        Note that calling `_add_objective` here _WILL_ cause an error, the optimization model _MUST_ 
        be modified in-place, not made from scratch as that would be too slow.
        """
    @abstractmethod
    def solve(self):
        """Solve the problem until stopping cirterion is met and record current values"""
