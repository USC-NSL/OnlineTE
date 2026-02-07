import numpy as np
import networkx as nx
import scipy.sparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from . import PDLPParams
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pdlp
from te.algorithms.base import *
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, traffic_to_list_of_tuples, Commodity
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from topologies.utils import get_node_in_array, get_node_out_array
from utils.logging import as_info, as_fail, ShortTQDMEnumerate


@dataclass
class ConstraintVector:
    """
    Encodes as 3-tuple, the elements of:
    - constraint coefficients
    - constraint lower bounds
    - constraint upper bounds
    Coefficients are kept as a sparse matrix, but bounds are dense.
    """
    coeffs: scipy.sparse.lil_matrix
    lowers: np.ndarray
    uppers: np.ndarray

    @classmethod
    def allocate(cls, n_vars, n_constraints):
        return cls(
            scipy.sparse.lil_matrix((n_constraints, n_vars)),
            np.zeros((n_constraints,)), np.zeros((n_constraints,))
        )
    
    def attach_to_program(self, lp: pdlp.QuadraticProgram):
        lp.constraint_matrix = self.coeffs.tocsc()
        lp.constraint_lower_bounds = self.lowers
        lp.constraint_upper_bounds = self.uppers


class PDLPTE(TrafficEngineeringLP):
    def __init__(self, problem_description: TrafficEngineeringProblemDescription, solver_params: PDLPParams) -> None:
        super().__init__(problem_description, solver_params)
        self._graph = problem_description.Graph
        self._traffic = problem_description.TM
        self._solver_params: SolverParams = solver_params
        self._capacities = np.array([c_e for _, _, c_e in self._graph.edges(data='capacity')])
        self._lp: Optional[pdlp.QuadraticProgram] = None
        self._commodity_list: List[Commodity] = traffic_to_commodity(self._traffic)
        self._commodity_tuple_list: List[Commodity] = traffic_to_list_of_tuples(self._traffic)
        self._utility: Optional[float] = None
        self._X_ek: Optional[np.ndarray] = None
        self._last_objective_value: Optional[float] = None

        self._NUM_VARIABLES: Optional[int] = None
        self._NUM_CONSTRAINTS: Optional[int] = None
        
        self._in_indexing: Dict[int, np.ndarray] = get_node_in_array(self._graph)
        self._out_indexing: Dict[int, np.ndarray] = get_node_out_array(self._graph)
        
        self._report_problem_size()
    
    @property
    def alg_name(self) -> str:
        return 'Centralized-PDLP'
    
    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
    
    @property
    def traffic(self) -> TrafficMatrixBase:
        return self._traffic
    
    @property
    def commodity_list(self) -> List[Commodity]:
        return self._commodity_list

    @property
    def objective_value(self) -> float:
        if self._problem_description.is_mlu:
            assert self._utility is not None
            return self._utility
        else:
            assert self._last_objective_value is not None
            return self._last_objective_value
    
    @property
    def objective_trace(self) -> Optional[List[float]]:
        # TODO: Anyway to get this from Gurobi?
        return None
    
    @property
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek

    def _report_problem_size(self):
        M = len(self._graph.nodes)
        N = len(self._graph.edges)
        K = len(self._commodity_list)

        print(as_info(f"Graph Size: {M} nodes | {N} edges"))
        print(as_info(f"Number of commodities: {K}"))

    def initialize_to(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
        # assert self._model is not None and self._flows is not None
        # solution.initiate_model_from_basis(self._model)

    def _set_solution(self, result: pdlp.SolverResult):
        K = len(self._commodity_list)
        N = len(self._graph.edges)
        if self._problem_description.is_mlu:
            self._utility = result.primal_solution[-1]
            self._X_ek = np.reshape(result.primal_solution[:-1], newshape=(N, K))
        else:
            self._X_ek = np.reshape(result.primal_solution, newshape=(N, K))
        self._set_last_objective(result)
    
    def _set_last_objective(self, result: pdlp.SolverResult):
        ls = result.solve_log.solution_stats.convergence_information
        self._last_objective_value = -ls[0].primal_objective

    def _make_variables(self):
        N = self.graph.number_of_edges()
        M = self.graph.number_of_nodes()
        K = len(self.commodity_list)

        # First `NK` variables are the flows, `X_ek`. The last one is the utilization, `u`.
        if self._problem_description.is_mlu:
            self._NUM_VARIABLES = N * K + 1
        else:
            self._NUM_VARIABLES = N * K
        # Each edge has one capacity constraint (N)
        # Each commodity has two constraints on the source and destination (2 * 2 * K)
        # Each commodity has one constraint on every node other than the source or the destination ((M-2) * K)
        self._NUM_CONSTRAINTS = N + 2 * 2 * K + (M - 2) * K

        self._lp = pdlp.QuadraticProgram()
    
    @staticmethod
    def _get_flow_index(num_commodities: int, e: int, k: int) -> int:
        return e * num_commodities + k
    
    def _get_variable_lower_bound_vector(self) -> np.ndarray:
        return np.zeros(shape=(self._NUM_VARIABLES,))
    
    def _get_variable_upper_bound_vector(self) -> np.ndarray:
        out = np.ones(shape=(self._NUM_VARIABLES,)) * np.inf
        if self._problem_description.is_mlu:
            out[-1] = 1.0
        return out

    def _set_capacity_constraint_vector(self, constraits: ConstraintVector):
        N = self._graph.number_of_edges()
        K = len(self.commodity_list)

        for e in range(N):
            start = e * K
            end = start + K
            constraits.coeffs[e, start:end] = 1.0
            if self._problem_description.is_mlu:
                constraits.coeffs[e, -1] = -self._capacities[e]
            else:
                constraits.uppers[e] = self._capacities[e]
            constraits.lowers[e] = -np.inf

    def _set_endpoint_constraint_vector_for_commodity(self, demand: float, source: int, target: int, k: int, 
                                                      constraints: ConstraintVector):
        constraint_start_index = self._graph.number_of_edges() + k*4

        K = len(self.commodity_list)
        IN_INDEX = self._in_indexing
        OUT_INDEX = self._out_indexing

        for e in OUT_INDEX[source]:
            constraints.coeffs[constraint_start_index + 0, self._get_flow_index(K, e, k)] = 1.0
            if self._problem_description.is_mlu:
                constraints.lowers[constraint_start_index + 0] = demand
            constraints.uppers[constraint_start_index + 0] = demand
        for e in IN_INDEX[source]:
            constraints.coeffs[constraint_start_index + 1, self._get_flow_index(K, e, k)] = 1.0
            constraints.lowers[constraint_start_index + 1] = 0
            constraints.uppers[constraint_start_index + 1] = 0
        for e in OUT_INDEX[target]:
            constraints.coeffs[constraint_start_index + 2, self._get_flow_index(K, e, k)] = 1.0
            constraints.lowers[constraint_start_index + 2] = 0
            constraints.uppers[constraint_start_index + 2] = 0
        for e in IN_INDEX[target]:
            constraints.coeffs[constraint_start_index + 3, self._get_flow_index(K, e, k)] = 1.0
            if self._problem_description.is_mlu:
                constraints.lowers[constraint_start_index + 3] = demand
            constraints.uppers[constraint_start_index + 3] = demand

    def _set_transit_constraint_vector_for_commodity(self, source: int, target: int, k: int,
                                                     constraints: ConstraintVector):
        M = self._graph.number_of_nodes()
        K = len(self.commodity_list)
        constraint_start_index = self._graph.number_of_edges() + K*4 + (M-2)*k
        IN_INDEX = self._in_indexing
        OUT_INDEX = self._out_indexing

        counter = 0
        for v in range(M):
            if v == source or v == target:
                continue
            for e in OUT_INDEX[v]:
                constraints.coeffs[constraint_start_index + counter, self._get_flow_index(K, e, k)] = 1.0
            for e in IN_INDEX[v]:
                constraints.coeffs[constraint_start_index + counter, self._get_flow_index(K, e, k)] = -1.0
            counter += 1
        assert counter == M-2

    def _get_objective_vector(self) -> Tuple[float, np.ndarray]:
        vec = np.zeros(shape=(self._NUM_VARIABLES,))
        if self._problem_description.is_mlu:
            vec[-1] = 1.0
        else:
            print("Adding objective")
            OUT_INDEX = self._out_indexing
            K = len(self._commodity_list)
            for k, commodity in ShortTQDMEnumerate(self.commodity_list):
                for e in OUT_INDEX[commodity.source]:
                    vec[self._get_flow_index(K, e, k)] = -1.0
        return 0, vec
    
    def _get_constraints(self) -> ConstraintVector:
        constraits = ConstraintVector.allocate(self._NUM_VARIABLES, self._NUM_CONSTRAINTS)
        self._set_capacity_constraint_vector(constraits)
        # Demand/Flow-conservation cosntraints
        print("Adding demand/flow-conservation constraints")
        for k, commodity in ShortTQDMEnumerate(self.commodity_list):
            source = commodity.source
            target = commodity.destination
            demand = commodity.demand
            self._set_endpoint_constraint_vector_for_commodity(demand, source, target, k, constraits)
            self._set_transit_constraint_vector_for_commodity(source, target, k, constraits)
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

    def _add_objective(self):
        assert self._lp is not None
        
        LP = self._lp
        offset, vector = self._get_objective_vector()
        LP.objective_offset = offset
        LP.objective_vector = vector
    
    def close(self):
        pass
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()
    
    def reset(self, with_params: False):
        raise NotImplementedError
    
    def solve(self, params: SolverParams = None) -> float:
        assert self._lp is not None
        assert params is None
        self.check_result = None

        LP = self._lp
        SOLVER_PARAMS: PDLPParams = self._solver_params

        PDHG_PARAMS = solvers_pb2.PrimalDualHybridGradientParams()
        optimality_criteria = PDHG_PARAMS.termination_criteria.simple_optimality_criteria
        optimality_criteria.eps_optimal_relative = SOLVER_PARAMS.ConvTol
        PDHG_PARAMS.termination_criteria.time_sec_limit = np.inf
        PDHG_PARAMS.num_threads = SOLVER_PARAMS.Threads
        PDHG_PARAMS.presolve_options.use_glop = SOLVER_PARAMS.Presolve
        PDHG_PARAMS.verbosity_level = 3
        
        try:
            result: pdlp.SolverResult = pdlp.primal_dual_hybrid_gradient(LP, PDHG_PARAMS)
            if result.solve_log.termination_reason == solve_log_pb2.TERMINATION_REASON_OPTIMAL:
                self._set_solution(result)
                return result.solve_log.solve_time_sec
            self._set_solution(result)
            print(as_fail(f"Solution failed. Reason: {solve_log_pb2.TerminationReason.Name(result.solve_log.termination_reason)}"))
            return -1
        except Exception as e:
            print(as_fail(f"Error while solving: {e}"))
            return -1

    def check(self):
        eval_params = self._problem_description.EvalParams
        unsat_ratio, unsat_commodities, total_satisfcation = check_flow_conservation(
            self._X_ek, self._graph, self._commodity_list,
            eval_params
        )
        congested_ratio, congested_links = check_capacity_constraint(
            self._X_ek, self._graph, self._commodity_list,
            eval_params
        )
        self.check_result = TrafficEngineeringLPCheckResult(
            unsat_ratio=unsat_ratio,
            congested_ratio=congested_ratio,
            unsat_commodities=unsat_commodities,
            congested_links=congested_links,
            density=np.count_nonzero(np.clip(self._X_ek)) / self._X_ek.size,
            total_satisfcation=total_satisfcation
        )
    
    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        assert self._X_ek is not None

        COMMODITIES = self._commodity_list
        GRAPH = self._graph
        X = self._X_ek

        ls = []
        for k, commodity in enumerate(COMMODITIES):
            flow_out = defaultdict(list)
            flow_in = defaultdict(list)
            for e, edge in enumerate(GRAPH.edges()):
                flow_out[edge[0]].append(X[e, k])
                flow_in[edge[1]].append(X[e, k])
            commodity_sent = Commodity(
                source=commodity.source, destination=commodity.destination,
                demand=sum(flow_out[commodity.source])
            )
            commodity_received = Commodity(
                source=commodity.source, destination=commodity.destination,
                demand=sum(flow_in[commodity.destination])
            )
            ls.append((commodity_sent, commodity_received))
        return ls

    def update_traffic_matrix(self, tm: TrafficMatrixBase):
        raise NotImplementedError
    
    def add_solution_elements(self, solution: TrafficEngineeringLPSolution):
        raise NotImplementedError


import jsonargparse

def centralized_pdlp_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(PDLPParams, 'SolverParams', help='PDLP Solver Params')
    return parser


def parse_centralized_pdlp_solver_params(args: jsonargparse.Namespace) -> PDLPParams:
    return PDLPParams.make_from_args(args.SolverParams)
