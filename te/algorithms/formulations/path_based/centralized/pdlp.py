import numpy as np
import networkx as nx
import scipy.sparse
from dataclasses import dataclass
# from joblib import Parallel, delayed
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from . import PDLPPathBasedSolverParams
from ortools.pdlp import solve_log_pb2
from ortools.pdlp import solvers_pb2
from ortools.pdlp.python import pdlp
from te.algorithms.base import *
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, traffic_to_list_of_tuples, Commodity
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from topologies.utils import get_node_in_array, get_node_out_array
from utils.logging import as_info, as_fail, ShortTQDM
from te.algorithms.sub_algorithms.paths import TShortestPaths, path_based_to_edge_based_nnz, get_or_make_path_object_for_topology_name
# from te.algorithms.sub_algorithms.utils import (get_slice_starts_and_exclusive_ends, get_number_of_required_workers,
#                                                 NUM_PROCS)


# MAX_NUMBER_OF_COMMODITIES_PER_CORE = 5000
# MAX_NUMBER_OF_WORKERS = min(8, NUM_PROCS)


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
    # coeffs: scipy.sparse.coo_matrix
    lowers: np.ndarray
    uppers: np.ndarray

    @classmethod
    def allocate(cls, n_vars, n_constraints):
        return cls(
            scipy.sparse.lil_matrix((n_constraints, n_vars)),
            # scipy.sparse.coo_matrix((n_constraints, n_vars)),
            np.zeros((n_constraints,)), np.zeros((n_constraints,))
        )
    
    # def set_from_indices(self, rows: List[int], cols: List[int], data: List[float]):
    #     shape = self.coeffs.shape
    #     self.coeffs = scipy.sparse.coo_matrix((data, (rows, cols)), shape=shape)
    
    def attach_to_program(self, lp: pdlp.QuadraticProgram):
        lp.constraint_matrix = self.coeffs.tocsc()
        lp.constraint_lower_bounds = self.lowers
        lp.constraint_upper_bounds = self.uppers


# def _set_capacity_constraint_vector_slice(
#     row_slice: List[np.ndarray], col_slice: List[np.ndarray],
#     demand_slice: np.ndarray, shift: int, T: int
# ) -> Tuple[List[int], List[int], List[float]]:
#     out_rows, out_cols, out_data = [], [], []
#     enum = range(len(row_slice)) if shift > 0 else ShortTQDM(range(len(row_slice)))
#     for k in enum:
#         row = row_slice[k]
#         col = col_slice[k]
#         nnz = len(row)
#         for i in range(nnz):
#             n = row[i]
#             t = col[i]
#             out_rows.append(n)
#             out_cols.append(T * (k + shift) + t)
#             out_data.append(demand_slice[k])
#     print(f'Finished on {shift}')
#     return out_rows, out_cols, out_data


class PDLPPathBasedTE(TrafficEngineeringLP):
    def __init__(self, problem_description: TrafficEngineeringProblemDescription, solver_params: PDLPPathBasedSolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._graph = problem_description.Graph
        self._traffic = problem_description.TM
        self._solver_params: PDLPPathBasedSolverParams = solver_params
        self._capacities = np.array([c_e for _, _, c_e in self._graph.edges(data='capacity')])
        self._lp: Optional[pdlp.QuadraticProgram] = None
        self._commodity_list: List[Commodity] = traffic_to_commodity(self._traffic)
        self._commodity_tuple_list: List[Commodity] = traffic_to_list_of_tuples(self._traffic)
        self._utility: Optional[float] = None
        self._X_ek: Optional[np.ndarray] = None
        self._splits: Optional[np.ndarray] = None
        self._last_objective_value: Optional[float] = None

        self._path_object: Optional[TShortestPaths] = None
        self._demands: np.ndarray = np.array([commodity.demand for commodity in self._commodity_list])

        self._NUM_VARIABLES: Optional[int] = None
        self._NUM_CONSTRAINTS: Optional[int] = None
        
        self._in_indexing: Dict[int, np.ndarray] = get_node_in_array(self._graph)
        self._out_indexing: Dict[int, np.ndarray] = get_node_out_array(self._graph)

        self._cv: Optional[ConstraintVector] = None
        
        self._report_problem_size()
        self._initialize()

    def _initialize(self):
        self._path_object = get_or_make_path_object_for_topology_name(
            topo_name=self._problem_description.EvalParams.TopologyName,
            T=self._solver_params.NumberOfPathsPerCommodity,
            edge_disjoint=False
        )
    
    @property
    def alg_name(self) -> str:
        return 'Path-Based PDLP'
    
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
        # TODO: Anyway to get this from PDLP?
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
        ALPHA = self._path_object.alpha
        K, N, T = ALPHA.shape
        D = self._demands
        if self._problem_description.is_mlu:
            self._utility = result.primal_solution[-1]
            self._splits = np.reshape(result.primal_solution[:-1], newshape=(K, T)).T
            self._X_ek = path_based_to_edge_based_nnz(self._splits, ALPHA.rows, ALPHA.cols, N, D)
        else:
            self._splits = np.reshape(result.primal_solution, newshape=(K, T)).T
            self._X_ek = path_based_to_edge_based_nnz(self._splits, ALPHA.rows, ALPHA.cols, N, D)
        self._set_last_objective(result)
    
    def _set_last_objective(self, result: pdlp.SolverResult):
        ls = result.solve_log.solution_stats.convergence_information
        self._last_objective_value = -ls[0].primal_objective

    def _make_variables(self):
        N = self.graph.number_of_edges()
        T = self._solver_params.NumberOfPathsPerCommodity
        K = len(self.commodity_list)

        # First `TK` variables are the splits, `Y_tk`. The last one is the utilization, `u`.
        # The splits are recorded as a `T x K` matrix in the end. The variables encode splits
        # as a flattened list, where each `T` variable becomes the associated column in the
        # final assignment.
        # As such, the final solution must be reshaped as a `K x T` matrix and then transposed.
        if self._problem_description.is_mlu:
            self._NUM_VARIABLES = T * K + 1
        else:
            self._NUM_VARIABLES = T * K
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
        out = np.ones(shape=(self._NUM_VARIABLES,)) * np.inf
        if self._problem_description.is_mlu:
            out[-1] = 1.0
        K, _, T = self._path_object.alpha.shape
        BETA = self._path_object.beta
        for k in range(K):
            b = BETA[k]
            if b < T:
                start = k * T + b
                end = (k+1) * T
                out[start:end] = 0
        return out

    def _set_capacity_constraint_vector(self, constraits: ConstraintVector):
        K, N, T = self._path_object.alpha.shape
        rows = self._path_object.alpha.rows
        cols = self._path_object.alpha.cols
        D = self._demands

        for k in ShortTQDM(range(K)):
            row = rows[k]
            col = cols[k]
            nnz = len(row)
            for i in range(nnz):
                n = row[i]
                t = col[i]
                constraits.coeffs[n, self._get_flow_index(T, k, t)] = D[k]
        for e in range(N):
            if self._problem_description.is_mlu:
                constraits.coeffs[e, -1] = -self._capacities[e]
            else:
                constraits.uppers[e] = self._capacities[e]
            constraits.lowers[e] = -np.inf
    # def _set_capacity_constraint_vector(self, constraits: ConstraintVector) -> Tuple[List[int], List[int], List[float]]:
    #     K, N, T = self._path_object.alpha.shape
    #     ALPHA = self._path_object.alpha
    #     D = self._demands
    #     if K <= MAX_NUMBER_OF_COMMODITIES_PER_CORE:
    #         rows, cols, data = _set_capacity_constraint_vector_slice(ALPHA.rows, ALPHA.cols, D, 0, T)
    #     else:
    #         slices = get_slice_starts_and_exclusive_ends(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
    #         print(slices)
    #         nprocs = get_number_of_required_workers(K, MAX_NUMBER_OF_WORKERS, MAX_NUMBER_OF_COMMODITIES_PER_CORE)
    #         print(as_info(f'Spawning {nprocs} workers to assign the coefficient matrix'))
    #         ls = Parallel(n_jobs=nprocs, return_as='generator', batch_size=1)\
    #             (delayed(_set_capacity_constraint_vector_slice)\
    #                 (ALPHA.rows[begin:end], ALPHA.cols[begin:end], D[begin:end], begin, T)
    #                 for begin, end in slices)
    #         print("finished!")
    #         rows, cols, data = [], [], []
    #         # for _ in range(nprocs):
    #         #     _row, _col, _data = ls.pop(0)
    #         for _row, _col, _data in ls:
    #             rows.extend(_row)
    #             cols.extend(_col)
    #             data.extend(_data)
    #     for e in range(N):
    #         if self._problem_description.is_mlu:
    #             rows.append(e)
    #             cols.append(self._NUM_VARIABLES-1)
    #             data.append(-self._capacities[e])
    #         else:
    #             constraits.uppers[e] = self._capacities[e]
    #         constraits.lowers[e] = -np.inf
    #     return rows, cols, data
    
    def _set_demand_constraint_vector(self, constraints: ConstraintVector):
        K, N, T = self._path_object.alpha.shape
        BETA = self._path_object.beta
        
        for k in ShortTQDM(range(K)):
            start = k * T
            end = start + BETA[k]
            constraints.coeffs[N + k, start:end] = 1.0
        constraints.uppers[N:] = 1.0
        if self._problem_description.is_mlu:
            constraints.lowers[N:] = 1.0
    # def _set_demand_constraint_vector(self, constraints: ConstraintVector) -> Tuple[List[int], List[int], List[float]]:
    #     K, N, T = self._path_object.alpha.shape
    #     BETA = self._path_object.beta
    #     rows, cols, data = [], [], []
    #     for k in ShortTQDM(range(K)):
    #         start = k * T
    #         end = start + BETA[k]
    #         rows.extend([N + k for _ in range(end - start)])
    #         cols.extend([i for i in range(start, end)])
    #         data.extend([1.0 for _ in range(end - start)])
    #     constraints.uppers[N:] = 1.0
    #     if self._problem_description.is_mlu:
    #         constraints.lowers[N:] = 1.0
    #     return rows, cols, data

    def _get_objective_vector(self) -> Tuple[float, np.ndarray]:
        vec = np.zeros(shape=(self._NUM_VARIABLES,))
        if self._problem_description.is_mlu:
            vec[-1] = 1.0
        else:
            vec[:-1] = 1.0
        return 0, vec
    
    def _get_constraints(self) -> ConstraintVector:
        constraits = ConstraintVector.allocate(self._NUM_VARIABLES, self._NUM_CONSTRAINTS)
        print("Adding capacity constraints")
        self._set_capacity_constraint_vector(constraits)
        print("Adding demand constraints")
        self._set_demand_constraint_vector(constraits)
        # print("Adding capacity constraints")
        # rows, cols, data = self._set_capacity_constraint_vector(constraits)
        # print("Adding demand constraints")
        # _rows, _cols, _data = self._set_demand_constraint_vector(constraits)
        # rows.extend(_rows)
        # cols.extend(_cols)
        # data.extend(_data)
        # constraits.set_from_indices(rows, cols, data)
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
        SOLVER_PARAMS: PDLPPathBasedSolverParams = self._solver_params

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
        unsat_ratio, unsat_commodities = check_flow_conservation(
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
            density=np.count_nonzero(self._X_ek) / self._X_ek.size
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
    parser.add_class_arguments(PDLPPathBasedSolverParams, 'SolverParams', help='PDLP Solver Params')
    return parser


def parse_centralized_pdlp_solver_params(args: jsonargparse.Namespace) -> PDLPPathBasedSolverParams:
    return PDLPPathBasedSolverParams.make_from_args(args.SolverParams)
