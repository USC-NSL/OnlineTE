import networkx as nx
import te.constants
from typing import List
from abc import ABC, abstractmethod
from collections import namedtuple
from te.traffic_models.base import TrafficMatrixBase, Commodity


class SolverParams(ABC):
    pass

GurobiSolverParams = namedtuple('GurobiSolverParams', [
    'Method', 'NumericFocus', 'BarConvTol',
    'OptimalityTol', 'FeasibilityTol',
    'LogFile'
], defaults=[
    te.constants.DEFAULT_SOLVER_METHOD,
    te.constants.DEFAULT_NUMERIC_FOCUS,
    te.constants.DEFAULT_BARRIER_CONVERGENCE_TOLERANCE,
    te.constants.DEFAULT_OPTIMALITY_TOLERANCE,
    te.constants.DEFAULT_FEASIBILITY_TOLERANCE,
    te.constants.DEFAULT_GUROBI_LOG_FILE
])

SolverParams.register(GurobiSolverParams)


class TrafficEngineeringLP(ABC):
    @property
    @abstractmethod
    def graph(self) -> nx.DiGraph:
        """The graph of the network"""
        pass

    @property
    @abstractmethod
    def traffic(self) -> TrafficMatrixBase:
        """The traffic matrix input"""
        pass

    @property
    @abstractmethod
    def commodity_list(self) -> List[Commodity]:
        """List of input commodities"""
        pass

    @property
    @abstractmethod
    def params(self) -> SolverParams:
        """Solver parameters"""
        pass

    @property
    @abstractmethod
    def objective_value(self) -> float:
        """Final objective value after optimization"""
        pass
    
    @abstractmethod
    def _make_variables(self):
        """Add required variables to the problem model"""
        pass

    @abstractmethod
    def _add_constraints(self):
        """Add all constraints needed to the problem model"""
        pass

    @abstractmethod
    def _add_objective(self):
        """Add the objective function to the problem model"""
        pass

    @abstractmethod
    def make_lp(self):
        """Create the LP and the Gurobi model object"""
        pass

    @abstractmethod
    def close(self):
        """Free and cleanup environment"""
        pass

    @abstractmethod
    def reset(self, with_params: False):
        """
        Reset solver as if no iterations were done.
        Optionally, may also reset parameters to their default values.
        """
        pass

    @abstractmethod
    def solve(self, params: SolverParams = None) -> float:
        """
        Solve the problem. May also accept an extra set of parameters,
        which will clear and override any existing ones using `reset`.

        return the runtime of the model, or `-1` if it failed.
        """
        pass
