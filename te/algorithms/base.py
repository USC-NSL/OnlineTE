from __future__ import annotations

import enum
import time
import numpy as np
import networkx as nx
import te.constants
import dataclasses
from collections import defaultdict
from typing import List, Optional, Tuple, Set, Dict, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from utils.logging import as_success, as_fail, as_warning, as_info
from utils.table_dataclass import TableDataclass
from topologies.utils import get_edge_indexing
from te.traffic_models.base import *
from te.algorithms.objective_evaluators import *
from te.algorithms.sub_algorithms.constraint_sanity_checks import *


class TEObjective(str, enum.Enum):
    """Describes the objective that we want to solve for"""
    MLU = "MLU"
    MAX_FLOW = "Max-Flow"
    MAX_CONCURRENT_FLOW = "Max-Concurrent-Flow"


@dataclass(frozen=True)
class SolverParams(TableDataclass):
    """Generic container for solver parameters"""
    pass


@dataclass
class TEEvaluationParams(SolverParams):
    """
    All generic evaluation parameters go into this class.

    Attributes
    ---------
    float_resolution: float
        A pair of floating point numbers that are less than this 
        value apart are treated as the same. May also be used as
        an `epsilon` tolerance when numerical problems may arise
        (e.g. when a divide by zero is likely).
    feasibility_tolerance: float
        Absolute contraint violation tolerance. Will override any
        other values if needed (including the one from Gurobi).
    feasibility_ratio: float
        Relative objective _AND_ constraint violation. 
    verbose: bool = False
        Print any constraint violation in full detail.
    skip_checks: bool = False
        Skip all per-TM checks.
    scale_factor: float = 1.0
        TM scale factor.
    sequence_length: int = 1
        Number of TMs in sequence to evaluate.
          
    Note
    ----
    For now, exactly one of `feasibility_tolerance` or `feasibility_ratio` must
    be given.
    """
    float_resolution: float = te.constants.FLOAT_RES
    feasibility_tolerance: Optional[float] = None
    feasibility_ratio: Optional[float] = None
    verbose: bool = False
    skip_checks: bool = False
    scale_factor: float = 1.0
    sequence_length: int = 1

    def __post_init__(self):
        # UPDATE: We will prevent both tolerances being given, it makes reasoning about things difficult ...
        assert (self.feasibility_ratio is None) ^ (self.feasibility_tolerance is None), \
            "Exactly one of `feasibility_tolerance` or `feasibility_ratio` MUST be given"
        self.left_column_share = 0.5
    
    def __call__(self, **kwargs):
        copy = dataclasses.replace(self)
        for k, v in kwargs.items:
            setattr(copy, k, v)


@dataclass
class TECheckResult:
    """
    A simple class for reporting the quality of a TE solution after it was
    checked for violations.

    Attributes
    ----------
    loop_witness: Optional[Tuple[int, int, int]]
        A commodity that has a loop. If we have one of these, then
        we messed up!
    congestions: List[Tuple[int, int, int, float, float]]
        A list of 5-tuples, containing edge index, edge source and
        edge destination, followed by the amount of routed flow over
        it and capacity.
    leaks: List[Tuple[int, int, int, float]]
        A list of 4-tuples, containing commodity index, source and
        destination followed by the leaking demand value from the
        destination.
    satisfaction: Optional[Tuple[int, int, int, float, float]]
        A list of 5-tuples, containing commodity index, source and
        destination followed by the routed demand and declared demand
        value.
    density: float
        Number of non-zero entries in the edge-based assignment.
    """
    loop_witness: Optional[Tuple[int, int, int]]
    congestions: List[Tuple[int, int, int, float, float]]
    leaks: List[Tuple[int, int, int, float]]
    satisfaction: Optional[Tuple[int, int, int, float, float]]
    density: float


class SolverCallbackType(str, enum.Enum):
    PreMake = "Pre-Make"
    "Called at the top of the body of `make_lp`"
    PostMake = "Post-Make"
    "Called at the end of the body of `make_lp`"
    PreSolve = "Pre-Solve"
    "Called at the head of the body of `solve`"
    PostSolve = "Post-Solve"
    "Called at the end of the body of `solve`"
    PreTMSolve = "Pre-TM-Solve"
    "Called at the head of the body of `solve_for_tm`"
    PostTMSolve = "Post-TM-Solve"
    "Called at the end of the body of `solve_for_tm`"
    PreIteration = "Pre-Iteration"
    "Called at the beginning of a single algorithm iteration"
    PostIteration = "Post-Iteration"
    "Called at the end of a single algorithm iteration"


class TETracer:
    """
    A container for collecting relevant data when running a TE algorithm.
    This will receive different callbacks that are configured to be called at
    different points of the solver runtime.

    Callbacks are arbitrary functions that receive the complete solver state and
    return a string key and a result. The key is used to keep the output in the
    tracer.
    """
    def __init__(self):
        self._traces: Dict[str, List] = defaultdict(list)
        self._callbacks: Dict[
            SolverCallbackType,
            List[
                Callable[
                    [TELP],
                    Optional[Tuple[str, Any]]
                ]
            ]
        ] = {
            tpe: [] for tpe in SolverCallbackType
        }
    
    @property
    def traces(self) -> Dict:
        return self._traces

    def add_callback(self, tpe: SolverCallbackType, cb: Callable):
        self._callbacks[tpe].append(cb)
    
    def execute_callbacks(self, te: TELP, tpe: SolverCallbackType):
        for cb in self._callbacks[tpe]:
            result = cb(te)
            if result is not None:
                key, output = result
                self._traces[key].append(output)

    def add_result(self, key: str, output: Any):
        self._traces[key].append(output)


@dataclass
class TEProblemDescription:
    """
    A full description of a TE problem and its required outputs.
    This can be passed to any TE solver to get the full description of the 
    problem; other parameters need only describe the solver properties.

    Attributes
    ----------
    objective: TEObjective
        The TE objective to solve for.
    eval_params: TEEvaluationParams
        Evaluation parameters, instructing the LP class about how optimal
        the solution need be and how much infeasibility are we willing to
        tolerate.
    graph: nx.DiGraph
        The topology as a directed graph. Each edge must have a `capacity`
        attribute as a floating point number.'
    tm_generator: TMGenerator
        The traffic matrix generator.
    """
    objective: TEObjective
    eval_params: TEEvaluationParams
    graph: nx.DiGraph
    tm_generator: TMGenerator


class TELP(ABC):
    """
    Base class for all TE solvers.
    This receives a problem description and a set of solver
    parameters, then iteratively solves for every TM in the
    generator.
    """

    @abstractmethod
    def __init__(self, problem_description: TEProblemDescription, 
                 solver_params: SolverParams, **kwargs):
        super().__init__(**kwargs)
        self._problem_description = problem_description
        self._solver_params = solver_params
        self._graph = problem_description.graph
        self._tm_generator = problem_description.tm_generator
        self._check_results: List[TECheckResult] = []
        self._tracer = TETracer()
        self._current_TM: Optional[np.ndarray] = None
        self._X_ek: Optional[np.ndarray] = None
        self._edge_indexing: Dict[Tuple[int, int], int] = \
            get_edge_indexing(self.graph)
        self._capacities = np.array([
            c_e for _, _, c_e in self.graph.edges(data='capacity')
        ])

    @property
    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def number_of_commodities(self) -> int:
        m = self.number_of_nodes
        return m * (m - 1)

    @property
    def problem_description(self) -> TEProblemDescription:
        """TE problem description object"""
        return self._problem_description

    @property
    def graph(self) -> nx.DiGraph:
        """The graph of the network"""
        return self._graph

    @property
    def tracer(self) -> TETracer:
        """Return the runtime tracer object"""
        return self._tracer

    def report_problem_size(self):
        print(as_info(f"Graph Size: {self.number_of_nodes} nodes |"
                      f" {self.number_of_edges} edges"))
        print(as_info(f"Number of commodities: {self.number_of_commodities}"))

    @property
    def solver_params(self) -> SolverParams:
        """Solver parameters"""
        return self._solver_params

    @property
    def check_results(self) -> List[TECheckResult]:
        """
        List of check results for each solved TM so far.
        """
        return self._check_result

    @property
    @abstractmethod
    def alg_name(cls) -> str:
        """Name of this algorithm"""

    @property
    def current_assignment(self) -> np.ndarray:
        """Return **current** edge-based assignments."""
        assert self._X_ek is not None
        return self._X_ek

    @property
    @abstractmethod
    def current_objective(self) -> float:
        """Return **current** objective."""

    def add_callback(self, tpe: SolverCallbackType, cb: Callable[[TELP], Tuple[str, Any]]):
        self._tracer.add_callback(tpe, cb)
    
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
    def _update_constraits(self, tm: np.ndarray):
        """Update constraints given a new traffic matrix"""

    @abstractmethod
    def _update_objective(self, tm: np.ndarray):
        """Update the objective given a new traffic matrix"""

    def make_lp(self):
        self._tracer.execute_callbacks(self, SolverCallbackType.PreMake)
        self.report_problem_size()
        t_start = time.perf_counter()
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        self._tracer.add_result("model_build_time", time.perf_counter() - t_start)
        self._tracer.execute_callbacks(self, SolverCallbackType.PostMake)

    @abstractmethod
    def close(self):
        """Free and cleanup environment"""

    @abstractmethod
    def _solve_for_tm(self, tm: np.ndarray):
        """Solve for a given TM."""
    
    def solve_for_tm(self, tm: np.ndarray):
        """Solve for a given TM."""
        self._tracer.execute_callbacks(self, SolverCallbackType.PreTMSolve)
        self._update_constraits(tm)
        self._update_objective(tm)
        t_start = time.perf_counter()
        self._solve_for_tm(tm)
        self._tracer.add_result("objective_trace", (self.current_objective, time.perf_counter() - t_start))
        if not self._problem_description.eval_params.skip_checks:
            self._check_results.append(self.check())
        self._tracer.execute_callbacks(self, SolverCallbackType.PostTMSolve)

    def solve(self) -> float:
        """Solve for each matrix in sequence."""
        self._tracer.execute_callbacks(self, SolverCallbackType.PreSolve)
        t_start = time.perf_counter()
        for tm in self._tm_generator:
            self._current_TM = tm
            self.solve_for_tm(tm)
        self._tracer.add_result("total_solve_time", time.perf_counter() - t_start)
        self._tracer.execute_callbacks(self, SolverCallbackType.PostSolve)

    def check(self) -> TECheckResult:
        """Performs sanity checks on the **current** solution."""
        assignments = self.assignments
        graph = self.graph
        commodity_list = self.commodity_list
        eval_params = self.problem_description.eval_params
        feasibility_tolerance = eval_params.feasibility_tolerance
        feasibility_ratio = eval_params.feasibility_ratio
        indexing = self._edge_indexing
        print(as_info(f"Checking for loops with absolute tolerance of {feasibility_tolerance}"))
        witness = check_loop_free_assignment(
            assignments, graph, feasibility_tolerance
        )
        if witness is not None:
            print(as_fail(f"Solution contains a loop for commodity {witness[0]}!"))
        leaks = check_flow_leaks(
            assignments, graph, commodity_list, feasibility_tolerance,
            feasibility_ratio
        )
        congestions = check_capacity_constraint(
            assignments, graph, commodity_list, eval_params
        )
        if self._problem_description.objective == TEObjective.MLU:
            satisfaction = check_flow_satisfaction(
                assignments, graph, commodity_list,
                feasibility_tolerance, feasibility_ratio,
                indexing
            )
        else:
            satisfaction = None
        return TECheckResult(
            loop_witness=witness,
            congestions=congestions,
            leaks=leaks,
            satisfaction=satisfaction,
            density=np.count_nonzero(
                np.clip(assignments, a_min=0, a_max=None)
            ) / assignments.size
        )


__all__ = [
    'SolverParams', 'TEEvaluationParams', 'TEObjective',
    'TECheckResult', 'TEProblemDescription', 'TELP',
    'TETracer', 'SolverCallbackType'
]