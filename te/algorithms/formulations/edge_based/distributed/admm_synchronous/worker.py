import time
import numpy as np
import networkx as nx
from typing import Optional, Tuple
from array_utils import set_global_precision
from array_utils.cpu.types import *
from . import SynchADMMSolverParams
from te.algorithms.communication import *
from topologies.utils import get_adjacency_null_space, get_graph_M_matrix, get_commodity_in_out_mask
from te.algorithms.objective_evaluators import get_total_routed_flow
from te.algorithms.sub_algorithms.pgd import do_memory_efficient_pgd, do_dual_pgd
from te.algorithms.sub_algorithms.lasso import sparse_range_lasso
from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment
from te.algorithms.sub_algorithms.cycle_remover import remove_all_cycles
from te.algorithms.objective_evaluators import get_total_routed_flow


# TODO: Add options on the controller for using Dual/Dense solvers.


class DenseSolver:
    def __init__(self,
        X_0: CPUArray,
        N: CPUArray,
        NNT: CPUArray,
        mask: BooleanCPUArray, 
        pgd_step: float, pgd_iters: int
    ):
        self._X_0: CPUArray = X_0
        self._X_ek: CPUArray = cpu_array(X_0)
        self._N = N
        self._NNT = NNT
        self._mask = mask
        self._lambda_ek = cpu_zeros(mask.shape)
        self._pgd_step = pgd_step
        self._pgd_iters = pgd_iters

    @property
    def X_ek(self) -> CPUArray:
        return self._X_ek

    @property
    def X_bar(self) -> CPUArray:
        return np.mean(self._X_ek, axis=1)

    def set_X_0(self, X_0: CPUArray):
        self._X_ek += (X_0 - self._X_0)
        self._X_0 = X_0
    
    def update(self, sharing_bias: CPUArray) -> CPUArray:
        assert self._X_0 is not None
        self._lambda_ek = do_memory_efficient_pgd(
            lambda_block=self._lambda_ek, 
            x_block=self._X_ek,
            nnt=self._NNT,
            bias=sharing_bias,
            x_block_0=self._X_0,
            step_size=self._pgd_step, 
            n_iter=self._pgd_iters, 
            mask=self._mask
        )
        # self._X_ek += self._NNT @ (
        #     self._lambda_ek -
        #     np.expand_dims(sharing_bias, axis=1)
        # )
        self._X_ek += self._NNT @ (self._lambda_ek - sharing_bias[:, None])
        return np.mean(self._X_ek, axis=1)


# class DualSolver:
#     def __init__(self,
#         X_0: CPUArray, NNT: CPUArray, mask: BooleanCPUArray, 
#         pgd_step: float, pgd_iters: int
#     ):
#         self._X_0 = X_0
#         self._X_0_bar = np.mean(X_0, axis=1)
#         self._NNT = NNT
#         self._mask = mask
#         self._lambda_ek = cpu_zeros(X_0.shape)
#         self._lambda_sum_ek = cpu_zeros(X_0.shape)
#         self._X_bar_e = np.mean(X_0, axis=1)
#         self._pgd_step = pgd_step
#         self._pgd_iters = pgd_iters

#     @property
#     def X_ek(self) -> CPUArray:
#         return self._X_0 + self._NNT @ self._lambda_sum_ek

#     def set_X_0(self, X_0: CPUArray):
#         self._X_0 = X_0
#         if self._X_ek is None:
#             self._X_ek = cpu_array(X_0)
    
#     def update(self, sharing_bias: CPUArray) -> CPUArray:
#         # TODO: Add loop regularizer for the dual solver
#         self._lambda_ek = do_dual_pgd(
#             lambda_block=self._lambda_ek, 
#             lambda_sum_block=self._lambda_sum_ek,
#             nnt=self._NNT,
#             bias=sharing_bias,
#             x_block_0=self._X_0,
#             step_size=self._pgd_step, 
#             n_iter=self._pgd_iters, 
#             mask=self._mask
#         )
#         self._lambda_sum_ek += (
#             self._lambda_ek - np.expand_dims(sharing_bias, axis=1)
#         )
#         self._X_bar_e = self._X_0_bar +\
#             self._NNT @ np.mean(self._lambda_sum_ek, axis=1)
#         return self._X_bar_e


# class SparseSolver:
#     def __init__(
#         self, X_0: CPUArray, NNT: CPUArray, mask: BooleanCPUArray, 
#         admm_step: float, admm_iters: int, l1_threshold: float
#     ):
#         self._X_0 = X_0
#         self._NNT = NNT
#         self._mask = mask
#         self._admm_step = admm_step
#         self._admm_iters = admm_iters
#         self._l1_threshold = l1_threshold
#         self._X_ek = cpu_array(X_0)
#         self._Z_ek = cpu_array(X_0)
#         self._L_ek = cpu_zeros(X_0.shape)
    
#     @property
#     def X_ek(self) -> CPUArray:
#         return self._Z_ek

#     def _get_current_C(self, sharing_bias: CPUArray) -> CPUArray:
#         return self._X_ek - np.expand_dims(sharing_bias, axis=1)
    
#     def update(self, sharing_bias: CPUArray) -> CPUArray:
#         self._X_ek, self._Z_ek, self._L_ek = sparse_range_lasso(
#             X_block=self._X_ek,
#             Z_block=self._Z_ek,
#             L_block=self._L_ek,
#             X_block_0=self._X_0,
#             NNT=self._NNT,
#             C_block=self._get_current_C(sharing_bias),
#             gamma=self._admm_step,
#             epsilon=self._l1_threshold,
#             n_iter=self._admm_iters,
#             mask=self._mask
#         )
#         return np.mean(self._X_ek, axis=1)


class SynchADMMWorkerNode(DistributedSolverNodeBase):
    def __init__(self, params: DistributedSolverNodeParams):
        super().__init__(params)
        self._solver_params: Optional[SynchADMMSolverParams] = None
        self._ready: bool = False

        self._T: Optional[int] = None
        self._graph_M: Optional[CPUArray] = None
        self._NULL_M: Optional[CPUArray] = None
        self._NNT_M: Optional[CPUArray] = None
        self._MASK_M: Optional[BooleanCPUArray] = None
        self._PINV_CACHE: Optional[CPUArray] = None

        self._sharing_bias_cached: Optional[CPUArray] = None

        self._dense_solver: Optional[DenseSolver] = None
        self._total_routed_flow: Optional[float] = None
        # self._dual_solver: Optional[DualSolver] = None
        # self._sparse_solver: Optional[SparseSolver] = None
        assert issubclass(
            params.CommunicationBackendCLS, WorkerBackendBase), "NOT TRUE!!!"
        self.backend: WorkerBackendBase =\
            params.CommunicationBackendCLS(params.RPCParams_)
        self.backend.start()

    def initialize(self):
        self.backend.set_solver_parameters = self.set_solver_parameters
        self.backend.set_topology = self.set_topology
        self.backend.do_inner_loop_update = self.do_inner_loop_pgd_update
        self.backend.update_cached_values = self.update_cached_values
        self.backend.report_chunk = self.report_chunk
        self.backend.report_aggregate = self.report_aggregate
        self.backend.update_demands = self.update_demands
    
    def run(self):
        self.backend.wait()

    def set_solver_parameters(self, new_params: SynchADMMSolverParams, num_workers: int):
        self._solver_params = new_params
        self.number_of_workers = num_workers
        set_global_precision(precision=new_params.Precision)

    def _set_chunk_alignment(self, num_endpoints: int):
        self._K = num_endpoints * (num_endpoints - 1)
        assert self._K % self._backend.number_of_peers == 0
        self._CHUNK_LEN = self._K // self._backend.number_of_peers
        self._K_START = self.worker_id * self._CHUNK_LEN

    def set_topology(self, graph: nx.DiGraph):
        self.graph = graph
        self._graph_M = get_graph_M_matrix(
            graph,
            self._capacities if self._solver_params.ScaleWithCapacity else None
        )
        self._NULL_M = cpu_array(get_adjacency_null_space(self._graph_M))
        self._NNT_M = self._NULL_M @ self._NULL_M.T
        self._PINV_CACHE = get_feasible_flow_assignment(
            self._graph_M, self.assigned_commodity_start_id, self.assigned_commodity_end_id
        )
        self._PINV_CACHE = remove_all_cycles(self._graph, self._PINV_CACHE)
        self._MASK_M = get_commodity_in_out_mask(
            graph, self.edge_indexing,
            commodity_id_start=self.assigned_commodity_start_id,
            commodity_id_end_exclusive=self.assigned_commodity_end_id
        )
        self._sharing_bias_cached = cpu_zeros((graph.number_of_edges(),))
        if self._solver_params.Beta is None:
            if self._dense_solver is None:
                self._dense_solver = DenseSolver(
                    cpu_array(self._PINV_CACHE),
                    self._NULL_M, self._NNT_M,
                    self._MASK_M, 
                    self._solver_params.Gamma,
                    self._solver_params.SwitchIterations
                )
                # self._dual_solver = DualSolver(
                #     self._X_ek_start_chunk, self._NNT_M, self._MASK_M, 
                #     self._solver_params.Gamma, self._solver_params.SwitchIterations
                # )
        else:
            raise NotImplementedError
            # self._sparse_solver = SparseSolver(
            #     self._NNT_M, self._MASK_M,
            #     self._solver_params.Gamma,
            #     self._solver_params.SwitchIterations,
            #     self._solver_params.Beta
            # )

    # TODO: Record the last epoch that we managed to handle ...
    def do_inner_loop_pgd_update(self, epoch: int) -> Tuple[int, CPUArray]:
        start = time.perf_counter_ns()
        if self._solver_params.Beta is None:
            means = self._dense_solver.update(self._sharing_bias_cached)
            # means = self._dual_solver.update(self._sharing_bias_cached)
        else:
            raise NotImplementedError
            # means = self._sparse_solver.update(self._sharing_bias_cached)
        return (time.perf_counter_ns() - start) // 1000, means

    def update_cached_values(self, sharing_bias: CPUArray):
        self._sharing_bias_cached = sharing_bias
    
    def report_chunk(self) -> CPUArray:
        if self._solver_params.Beta is None:
            assignments = self._dense_solver.X_ek
            assignments = remove_all_cycles(self._graph, assignments)
            # self._total_routed_flow = get_total_routed_flow(assignments, self._graph_M)
            return assignments
            # return self._dual_solver.X_ek
        else:
            raise NotImplementedError
            # return self._sparse_solver.X_ek
    
    def report_aggregate(self) -> CPUArray:
        return np.sum(self._dense_solver.X_ek, axis=1)

    def update_demands(self, demands: CPUArray) -> CPUArray:
        X_0 = demands * self._PINV_CACHE
        # print(f"Updated total flow: {get_total_routed_flow(X_0, self._graph_M)}")
        self._dense_solver.set_X_0(X_0)
        return self._dense_solver.X_bar

    def close(self):
        self.backend.close()
