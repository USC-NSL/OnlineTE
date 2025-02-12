import networkx as nx
import te.constants
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from te.traffic_models.base import TrafficMatrixBase, Commodity


class SolverParams(ABC):
    pass

@dataclass
class GurobiSolverParams(SolverParams):
    Method: int = te.constants.DEFAULT_SOLVER_METHOD
    NumericFocus: int = te.constants.DEFAULT_NUMERIC_FOCUS
    BarConvTol: float = te.constants.DEFAULT_BARRIER_CONVERGENCE_TOLERANCE
    FeasibilityTol: float = te.constants.DEFAULT_FEASIBILITY_TOLERANCE
    LogFile: str = te.constants.DEFAULT_GUROBI_LOG_FILE


class TrafficEngineeringLP(ABC):
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
    def objective_trace(self) -> Optional[List[float]]:
        """List of objective values during algorithm iterations"""
    
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
