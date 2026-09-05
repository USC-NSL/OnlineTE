import numpy as np
from typing import Tuple, Optional, Dict
from . import PDLPSolverParams
from ortools.pdlp.python import pdlp
from te.algorithms.base import *
from te.traffic_models.base import commodity_od_iterator, traffic_to_commodity
from topologies.utils import get_node_in_array, get_node_out_array
from utils.logging import ShortTQDMEnumerate
from utils.pdlp_utils import solve_qp_or_scream, ConstraintVector


class PDLPTE(TELP[PDLPSolverParams]):
    def __init__(self, problem_description: TEProblemDescription, solver_params: PDLPSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._lp: Optional[pdlp.QuadraticProgram] = None
        self._utility: Optional[float] = None
        self._X_ek: Optional[np.ndarray] = None
        self._last_objective_value: Optional[float] = None

        self._NUM_VARIABLES: Optional[int] = None
        self._NUM_CONSTRAINTS: Optional[int] = None
        
        self._in_indexing: Dict[int, np.ndarray] = get_node_in_array(self._graph)
        self._out_indexing: Dict[int, np.ndarray] = get_node_out_array(self._graph)

        self._constraints: Optional[ConstraintVector] = None
    
    @property
    def alg_name(self) -> str:
        return 'Centralized-PDLP'

    @property
    def current_objective(self) -> float:
        return abs(self._last_objective_value)

    def _set_solution(self, result: pdlp.SolverResult):
        K = self.number_of_commodities
        N = self.number_of_edges
        match self.objective:
            case TEObjective.MLU:
                self._utility = result.primal_solution[-1]
                self._X_ek = np.reshape(result.primal_solution[:-1], shape=(N, K))
            case TEObjective.MAX_FLOW:
                self._X_ek = np.reshape(result.primal_solution, shape=(N, K))
            case _ : raise ValueError
        self._set_last_objective(result)
    
    def _set_last_objective(self, result: pdlp.SolverResult):
        ls = result.solve_log.solution_stats.convergence_information
        self._last_objective_value = -ls[0].primal_objective

    def _make_variables(self):
        N = self.number_of_edges
        M = self.number_of_nodes
        K = self.number_of_commodities

        # First `NK` variables are the flows, `X_ek`. The last one is the utilization, `u`.
        if self.objective == TEObjective.MLU:
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
        # No upper bound, even for MLU
        out = np.ones(shape=(self._NUM_VARIABLES,)) * np.inf
        return out

    def _set_capacity_constraint_vector(self, constraits: ConstraintVector):
        N = self.number_of_edges
        K = self.number_of_commodities

        for e in range(N):
            start = e * K
            end = start + K
            constraits.coeffs[e, start:end] = 1.0
            match self.objective:
                case TEObjective.MLU:
                    constraits.coeffs[e, -1] = -self._capacities[e]
                case _:
                    constraits.uppers[e] = self._capacities[e]
            constraits.lowers[e] = -np.inf

    def _set_endpoint_constraint_vector_for_commodity(
        self, source: int, target: int, k: int, 
        constraints: ConstraintVector
    ):
        constraint_start_index = self._graph.number_of_edges() + k*4

        K = self.number_of_commodities
        IN_INDEX = self._in_indexing
        OUT_INDEX = self._out_indexing
        IS_MLU = self._problem_description.objective == TEObjective.MLU

        # Source out-flow must euqal demand (place-holder of 1)
        for e in OUT_INDEX[source]:
            constraints.coeffs[constraint_start_index + 0, self._get_flow_index(K, e, k)] = 1.0
        if IS_MLU:
            constraints.lowers[constraint_start_index + 0] = 1
        constraints.uppers[constraint_start_index + 0] = 1
        # Source in-flow must be 0
        for e in IN_INDEX[source]:
            constraints.coeffs[constraint_start_index + 1, self._get_flow_index(K, e, k)] = 1.0
        constraints.lowers[constraint_start_index + 1] = 0
        constraints.uppers[constraint_start_index + 1] = 0
        # Destination out-flow must be 0
        for e in OUT_INDEX[target]:
            constraints.coeffs[constraint_start_index + 2, self._get_flow_index(K, e, k)] = 1.0
        constraints.lowers[constraint_start_index + 2] = 0
        constraints.uppers[constraint_start_index + 2] = 0
        # Destination in-flow must equal demand (place-holder of 1)
        for e in IN_INDEX[target]:
            constraints.coeffs[constraint_start_index + 3, self._get_flow_index(K, e, k)] = 1.0       
        if IS_MLU:
            constraints.lowers[constraint_start_index + 3] = 1
        constraints.uppers[constraint_start_index + 3] = 1

    def _set_demand_for_commodity(
        self, demand: float, k: int, constraints: ConstraintVector
    ):
        constraint_start_index = self._graph.number_of_edges() + k*4

        IS_MLU = self.objective == TEObjective.MLU

        if IS_MLU:
            constraints.lowers[constraint_start_index + 0] = demand
        constraints.uppers[constraint_start_index + 0] = demand
        if IS_MLU:
            constraints.lowers[constraint_start_index + 3] = demand
        constraints.uppers[constraint_start_index + 3] = demand

    def _set_transit_constraint_vector_for_commodity(
        self, source: int, target: int, k: int,
        constraints: ConstraintVector
    ):
        M = self.number_of_nodes
        K = self.number_of_commodities
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
        K = self.number_of_commodities
        vec = np.zeros(shape=(self._NUM_VARIABLES,))
        match self.objective:
            case TEObjective.MLU:
                vec[-1] = 1.0
            case TEObjective.MAX_FLOW:
                OUT_INDEX = self._out_indexing
                for k, od_pair in enumerate(commodity_od_iterator(self.number_of_nodes)):
                    for e in OUT_INDEX[od_pair[0]]:
                        vec[self._get_flow_index(K, e, k)] = -1.0
            case _ :
                raise ValueError
        return 0, vec
    
    def _get_constraints(self) -> ConstraintVector:
        constraits = ConstraintVector.allocate(self._NUM_VARIABLES, self._NUM_CONSTRAINTS)
        self._set_capacity_constraint_vector(constraits)
        # Demand/Flow-conservation cosntraints
        print("Adding demand/flow-conservation constraints")
        for k, od_pair in ShortTQDMEnumerate(commodity_od_iterator(self.number_of_nodes), self.number_of_commodities):
            source, target = od_pair
            self._set_endpoint_constraint_vector_for_commodity(source, target, k, constraits)
            self._set_transit_constraint_vector_for_commodity(source, target, k, constraits)
        return constraits
    
    def _add_constraints(self):
        assert self._lp is not None
        
        LP = self._lp
        # Lower and upper variable bounds
        LP.variable_lower_bounds = self._get_variable_lower_bound_vector()
        LP.variable_upper_bounds = self._get_variable_upper_bound_vector()
        # Capacity/Demand constraints
        self._constraints = self._get_constraints()
        self._constraints.attach_to_program(LP)

    def _add_objective(self):
        assert self._lp is not None
        
        LP = self._lp
        offset, vector = self._get_objective_vector()
        LP.objective_offset = offset
        LP.objective_vector = vector
    
    def close(self):
        pass
    
    def _solve_for_tm(self, tm: np.ndarray):
        assert self._lp is not None

        result: pdlp.SolverResult = solve_qp_or_scream(
            qp = self._lp,
            params = self._solver_params,
            feasibility_tolerance = self._problem_description.eval_params.feasibility_tolerance,
            optimality_tolerance = self._problem_description.eval_params.optimality_tolerance,
            verbose = self._problem_description.eval_params.verbose
        )
        self._set_solution(result)

    def _update_constraits(self, tm: np.ndarray):
        assert self._constraints is not None
        COMMODITIES = traffic_to_commodity(tm)
        CONSTRAINTS = self._constraints
        for k, commodity in enumerate(COMMODITIES):
            self._set_demand_for_commodity(
                demand = commodity.demand, k = k,
                constraints = CONSTRAINTS
            )
        self._constraints.update_bounds(self._lp)

    def _update_objective(self, tm: np.ndarray):
        pass


import jsonargparse

def centralized_pdlp_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(PDLPSolverParams, 'SolverParams', help='PDLP Solver Params')
    return parser


def parse_centralized_pdlp_solver_params(args: jsonargparse.Namespace) -> PDLPSolverParams:
    return PDLPSolverParams.make_from_args(args.SolverParams)
