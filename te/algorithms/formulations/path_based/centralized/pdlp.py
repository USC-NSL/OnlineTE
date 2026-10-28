import numpy as np
from typing import Tuple, Optional, Dict
from . import PDLPPathBasedSolverParams
from ortools.pdlp.python import pdlp
from te.algorithms.base import *
from te.traffic_models.base import *
from topologies.utils import get_node_in_array, get_node_out_array
from utils.logging import as_info, ShortTQDM
from te.path_providers import *
from te.path_providers.sparse_ops import path_based_to_edge_based_nnz
from . import PDLPPathBasedSolverParams
from utils.pdlp_utils import ConstraintVector, get_pdhg_params, solve_qp_or_scream


class PDLPPathBasedTE(TELP[PDLPPathBasedSolverParams]):
    def __init__(self, problem_description: TEProblemDescription, solver_params: PDLPPathBasedSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._lp: Optional[pdlp.QuadraticProgram] = None
        self._utility: Optional[float] = None
        self._X_ek: Optional[np.ndarray] = None
        self._routed_demand_tk: Optional[np.ndarray] = None
        self._last_objective_value: Optional[float] = None
        self._last_solution: Optional[pdlp.PrimalAndDualSolution] = None

        self._path_object: Optional[PathProvider] = None

        self._NUM_VARIABLES: Optional[int] = None
        self._NUM_CONSTRAINTS: Optional[int] = None
        
        self._in_indexing: Dict[int, np.ndarray] = get_node_in_array(self._graph)
        self._out_indexing: Dict[int, np.ndarray] = get_node_out_array(self._graph)

        self._cv: Optional[ConstraintVector] = None

        self._pdhg_params = get_pdhg_params(
            params=self._solver_params,
            feasibility_tolerance=self.feasibility_tolerance,
            optimality_tolerance=self.optimality_tolerance,
            verbose=problem_description.eval_params.verbose
        )
        
        self._initialize()

    def _initialize(self):
        path = self._solver_params.path_file
        if path is not None:
            self._path_object = PathProvider.load(path)
        else:
            self._path_object = build_provider(
                T=self._solver_params.max_num_paths_per_commodity,
                graph=self._graph,
                per_commodity_provider=get_scheme(),
                edge_indexing=self._edge_indexing
            )
        K, N, _ = self._path_object.shape
        assert K == self.number_of_commodities,\
            'Commodity count does not match! This is the wrong path file!'
        assert N == self.number_of_edges,\
            'Number of edges do not match! This is the wrong path file!'

    @property
    def alg_name(self) -> str:
        return 'Path-Based PDLP'

    @property
    def current_objective(self) -> float:
        return abs(self._last_objective_value)

    def _set_solution(self, result: pdlp.SolverResult):
        ROWS = self._path_object.rows
        COLS = self._path_object.cols
        K, N, T = self._path_object.shape
        DEMANDS = traffic_to_demands(self._current_TM)
        match self.objective:
            case TEObjective.MLU:
                self._utility = result.primal_solution[-1]
                self._routed_demand_tk = np.reshape(result.primal_solution[:-1], shape=(K, T)).T / DEMANDS
            case TEObjective.MAX_FLOW:
                self._routed_demand_tk = np.reshape(result.primal_solution, shape=(K, T)).T / DEMANDS
            case _: raise NotImplementedError
        self._X_ek = path_based_to_edge_based_nnz(
            self._routed_demand_tk, ROWS, COLS, N, DEMANDS
        )
        self._set_last_objective(result)
        self._set_last_solution(result)
    
    def _set_last_objective(self, result: pdlp.SolverResult):
        ls = result.solve_log.solution_stats.convergence_information
        self._last_objective_value = -ls[0].primal_objective

    def _set_last_solution(self, result: pdlp.SolverResult):
        self._last_solution = pdlp.PrimalAndDualSolution()
        self._last_solution.primal_solution = result.primal_solution
        self._last_solution.dual_solution = result.dual_solution

    def _make_variables(self):
        N = self.graph.number_of_edges()
        T = self._solver_params.max_num_paths_per_commodity
        K = self.number_of_commodities

        # First `TK` variables are the splits, `Y_tk`. The last one is the utilization, `u`.
        # The splits are recorded as a `T x K` matrix in the end. The variables encode splits
        # as a flattened list, where each `T` variable becomes the associated column in the
        # final assignment.
        # As such, the final solution must be reshaped as a `K x T` matrix and then transposed.
        match self.objective:
            case TEObjective.MLU: self._NUM_VARIABLES = T * K + 1
            case TEObjective.MAX_FLOW: self._NUM_VARIABLES = T * K
            case _: raise NotImplementedError
        # Each edge has one capacity constraint (N)
        # Each commodity has 1 constraint for being on/inside a simplex. For MLU,
        # the condition turns from inequality into an equality. (K)
        self._NUM_CONSTRAINTS = N + K

        self._lp = pdlp.QuadraticProgram()
    
    @staticmethod
    def _get_flow_index(num_paths: int, k: int, t: int) -> int:
        return k * num_paths + t
    
    def _get_variable_lower_bound_vector(self) -> np.ndarray:
        return np.zeros(shape=(self._NUM_VARIABLES,))
    
    def _get_variable_upper_bound_vector(self) -> np.ndarray:
        out = np.full(shape=(self._NUM_VARIABLES,), fill_value=np.inf)
        K, _, T = self._path_object.shape
        BETA = self._path_object.beta
        for k in range(K):
            b = BETA[k]
            if b < T:
                start = k * T + b
                end = (k+1) * T
                out[start:end] = 0
        return out

    def _set_capacity_constraint_vector(
        self,
        constraits: ConstraintVector
    ):
        K, N, T = self._path_object.shape
        rows = self._path_object.rows
        cols = self._path_object.cols

        for k in ShortTQDM(range(K)):
            row = rows[k]
            col = cols[k]
            nnz = len(row)
            for i in range(nnz):
                n = row[i]
                t = col[i]
                constraits.coeffs[n, self._get_flow_index(T, k, t)] = 1
        match self.objective:
            case TEObjective.MLU:
                for e in range(N):
                    constraits.coeffs[e, -1] = -self._capacities[e]
                    constraits.lowers[e] = -np.inf
            case TEObjective.MAX_FLOW:
                for e in range(N): constraits.uppers[e] = self._capacities[e]
            case _: raise NotImplementedError
    
    def _set_demand_constraint_vector(self, constraints: ConstraintVector):
        K, N, T = self._path_object.shape
        BETA = self._path_object.beta
        
        for k in ShortTQDM(range(K)):
            start = k * T
            end = start + BETA[k]
            constraints.coeffs[N + k, start:end] = 1.0
        constraints.uppers[N:] = 1.0

        match self.objective:
            case TEObjective.MLU: constraints.lowers[N:] = 1.0
            case TEObjective.MAX_FLOW: pass # Just non-negative
            case _: raise NotImplementedError

    def _update_demand_constraint_vector(
        self,
        demands: np.ndarray
    ):
        K, N, _ = self._path_object.shape
        CONSTRAINTS = self._cv
        assert demands.shape == (K,)
        CONSTRAINTS.uppers[N:] = demands

        match self.objective:
            case TEObjective.MLU: CONSTRAINTS.lowers[N:] = demands
            case TEObjective.MAX_FLOW: pass # Just non-negative
            case _: raise NotImplementedError

    def _get_objective_vector(self) -> Tuple[float, np.ndarray]:
        vec = np.zeros(shape=(self._NUM_VARIABLES,))
        match self.objective:
            case TEObjective.MLU: vec[-1] = 1.0
            case TEObjective.MAX_FLOW: vec[:-1] = -1.0 / self.number_of_commodities
            case _: raise NotImplementedError
        return 0, vec
    
    def _get_constraints(self) -> ConstraintVector:
        constraits = ConstraintVector.allocate(self._NUM_VARIABLES, self._NUM_CONSTRAINTS)
        print(as_info("Adding capacity constraints"))
        self._set_capacity_constraint_vector(constraits)
        print(as_info("Adding demand constraints"))
        self._set_demand_constraint_vector(constraits)
        return constraits
    
    def _add_constraints(self):
        assert self._lp is not None
        
        LP = self._lp
        # Lower and upper variable bounds
        LP.variable_lower_bounds = self._get_variable_lower_bound_vector()
        LP.variable_upper_bounds = self._get_variable_upper_bound_vector()
        # Capacity/Demand constraints
        constraints = self._get_constraints()
        constraints.attach_to_program(LP)
        self._cv = constraints

    def _add_objective(self):
        assert self._lp is not None
        
        LP = self._lp
        offset, vector = self._get_objective_vector()
        LP.objective_offset = offset
        LP.objective_vector = vector
    
    def close(self):
        pass

    def _solve_for_tm(self, tm: np.ndarray):
        result = solve_qp_or_scream(
            self._lp,
            self._pdhg_params,
            self._last_solution
        )
        self._set_solution(result)

    def _update_constraits(self, tm: np.ndarray):
        self._update_demand_constraint_vector(
            traffic_to_demands(tm)
        )
        self._cv.update_bounds(self._lp)

    def _update_objective(self, tm):
        pass


import jsonargparse

def centralized_pdlp_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(PDLPPathBasedSolverParams, 'SolverParams', help='PDLP Solver Params')
    return parser


def parse_centralized_pdlp_solver_params(args: jsonargparse.Namespace) -> PDLPPathBasedSolverParams:
    return PDLPPathBasedSolverParams.make_from_args(args.SolverParams)
