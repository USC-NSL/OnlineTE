import time
import cupy as cp
import numpy as np
import networkx as nx
from typing import List, Optional
from te.algorithms.base import *
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from te.algorithms.sub_algorithms.pgd import do_pgd_gpu
from topologies.utils import get_graph_M_matrix, get_adjacency_null_space, get_commodity_in_out_mask
from utils.exceptions import SolutionInterrupted
from utils.logging import as_success, ShortTQDM
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.gpu_utils import (ScatteredGPUArray, PartitionedGPUArray,
                                                 gpu_partitioned_zeros, gpu_scattered_zeros,
                                                 as_cpu_array, as_gpu_array, reduce_in_place,
                                                 scattered_shape, as_scattered_gpu_arrray, zip_map,
                                                 set_gpu_float_precision, gpu_sparse_to_dense)
from te.algorithms.array_utils.cpu_utils import (CPUArray, BooleanCPUArray, cpu_array, cpu_double_array, 
                                                 set_cpu_float_precision, cpu_cast_float)
from te.algorithms.sub_algorithms.admm import ADMMWrapper
from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment, InitialSolutionType
from te.algorithms.sub_algorithms.noisy_loop_remover import remove_noisy_loops
from te.algorithms.sub_algorithms.admm_consensus_test import outer_admm_consensus_test, inner_admm_consensus_test
from te.algorithms.statistics.helpers import record_cpu_runtime, record_gpu_runtime, record_reserved_gpu_memory
from . import GPUParams
from ..base import EdgeBasedTEBase
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver, ControllerMLUException


class DenseSolver:
    def __init__(self, X_0: PartitionedGPUArray, NNT: ScatteredGPUArray, mask: PartitionedGPUArray, 
                pgd_step: float, pgd_iters: int):
        self._X_0 = X_0
        self._NNT = NNT
        self._mask = mask
        self._lambda_ek = gpu_partitioned_zeros(scattered_shape(X_0))
        # TODO: Is there a way to get away with caching this?
        self._X_ek = gpu_sparse_to_dense(X_0)
        self._pgd_step = pgd_step
        self._pgd_iters = pgd_iters
        self._epsilon = cpu_cast_float((scattered_shape(self._X_ek)[1])**(-1.5))

    @property
    def X_ek(self) -> PartitionedGPUArray:
        return self._X_ek
    
    def get_X_ek(self):
        return as_cpu_array(self._X_ek)
    
    def update(self, sharing_bias: ScatteredGPUArray) -> PartitionedGPUArray:
        self._lambda_ek = do_pgd_gpu(
            lambda_block=self._lambda_ek, 
            x_block=self._X_ek,
            nnt=self._NNT,
            bias=sharing_bias,
            x_block_0=self._X_0,
            step_size=self._pgd_step, 
            n_iter=self._pgd_iters, 
            mask=self._mask,
            epsilon=self._epsilon
        )
        self._X_ek = zip_map(
            [self._X_ek, self._NNT, self._lambda_ek, sharing_bias],
            lambda x, nnt, l, sb: x + nnt @ (l - self._epsilon - cp.expand_dims(sb, axis=1))
        )
        return self._X_ek


class GPUADMMTE(EdgeBasedTEBase):
    """
    A GPU accelerated implementation of our distributed, synchronous algorithm
    that can run locally on one or more GPU units.
    """
    def __init__(self,
                 problem_description: TrafficEngineeringProblemDescription,
                 solver_params: GPUParams, mlu_cls: type[ControllerMLUSolver],
                 mlu_params: SolverParams) -> None:
        super().__init__(problem_description, solver_params)
        self._graph = problem_description.Graph
        self._M = get_graph_M_matrix(self._graph)
        self._traffic = problem_description.TM
        self._solver_params: GPUParams = solver_params
        self._rng = np.random.default_rng(seed=solver_params.TMSeed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        """
        These two matrixes always appear next to Y_tk,
        which lives in the GPU memory. As such, we would be
        much better off keeping them in GPU as well.
        They each require `m x T` and `m x m` entries, and are quite
        small compared to other matrices, so they have little memory
        footprint.
        """
        self._NULL_M: Optional[ScatteredGPUArray] = None
        self._NNT_M: Optional[ScatteredGPUArray] = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._M_MASK: Optional[BooleanCPUArray] = None

        self._capacities: Optional[CPUArray] = None
        self._c_norm: Optional[float] = None
        self._alpha: Optional[float] = None

        self._mlu_solver_cls: type[ControllerMLUSolver] = mlu_cls
        self._mlu_params: SolverParams = mlu_params
        self._mlu_solver: Optional[ControllerMLUSolver] = None

        self._Z_e_start: Optional[CPUArray] = None
        self._Z_e_start: Optional[CPUArray] = None
        self._X_ek_sum_e: Optional[CPUArray] = None

        self._X_ek_start: Optional[PartitionedGPUArray] = None
        """
        An `n x K` matrix, this is our first heavy hitter in terms of
        memory. This matrix is involved in the PGD step as input, and
        as such must be kept in GPU memory.
        One silver lining is that this matrix is almost always very sparse,
        and as such we don't have to pay the price of keeping all of it
        in memory.
        """
        self._X_ek: Optional[CPUArray] = None
        """
        The `n x K` assignment matrix, we need not keep this matrix in the
        GPU memory, as it is only reconstructed when the algorithm is finished.
        """
        
        self._sharing_mean_1: Optional[ScatteredGPUArray] = None
        self._sharing_mean_2: Optional[ScatteredGPUArray] = None
        self._sharing_dual: Optional[ScatteredGPUArray] = None
        """
        The inner ADMM loop is solved entirely on the GPUs.
        Hopefully, we will not have to worry too much about the overhead
        of copying these small arrays back and forth.
        """

        self._dense_solver: Optional[DenseSolver] = None

        self._objective_trace: TrafficEngineeringLPObjectiveTrace = \
            TrafficEngineeringLPObjectiveTrace(['Perceived Utilization', 'Actual Utilization'])
        self._objective_gap_trace = []

        set_global_precision(self._solver_params.Precision)
        set_gpu_float_precision()
        set_cpu_float_precision()

    def initialize(self):
        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._report_problem_size()

    @property
    def alg_name(self) -> str:
        return 'Centralized GPU'

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
    
    @property
    def traffic(self) -> TrafficMatrixBase:
        return self._traffic

    @property
    def params(self) -> SolverParams:
        return self._solver_params
    
    @property
    def commodity_list(self) -> List[Commodity]:
        return self._commodity_list

    @property
    def objective_value(self) -> float:
        return self._mlu_solver.current_u
    
    @property
    def objective_trace(self) -> Optional[TrafficEngineeringLPObjectiveTrace]:
        return self._objective_trace

    @property
    def objective_gap_trace(self) -> Optional[List[float]]:
        return self._objective_gap_trace
    
    @property
    def assignments(self) -> CPUArray:
        assert self._X_ek is not None
        return self._X_ek

    @record_cpu_runtime('Feasible-Assignment')
    def _set_initial_feasible_solution(self):
        # Since we want to partition these over GPUs, we need a CSC array,
        # even though CSR performed a little better in our experiments.
        self._capacities = cpu_double_array([item[-1] for item in self._graph.edges(data='capacity')])
        self._X_ek_start = as_gpu_array(
            get_feasible_flow_assignment(self._graph, self._commodity_list, solution_type=InitialSolutionType.PSEUDO_INVERSE),
            partition=True
        )
        self._X_ek_start = zip_map([self._X_ek_start, as_scattered_gpu_arrray(self._capacities[:, None])], lambda x, c: cp.divide(x, c))
        self._Z_e_start = reduce_in_place(
            self._X_ek_start,
            lambda x0: x0.sum(axis=1),
            gather=True
        )
    
    def _set_NULL_M(self):
        M = self._M
        assert len(M.shape) == 2
        m, n = M.shape
        assert m < n
        N = as_scattered_gpu_arrray(get_adjacency_null_space(M))
        T = N[0].shape[1]
        self._NULL_M = N
        self._NNT_M = zip_map([N], lambda n: n @ n.T)
        self._T = T
        self._NUM_EDGES = n
        self._M_MASK = as_gpu_array(get_commodity_in_out_mask(self.graph, self.commodity_list), partition=True)

    def _get_Z_value(self) -> CPUArray:
        return self._mlu_solver.current_Z
    
    def _initialize_variables_and_residuals(self):
        K = len(self._commodity_list)
        NUM_EDGES = self._NUM_EDGES
        self._c_norm = np.linalg.norm(self._capacities)
        self._alpha = 1 / np.sqrt(NUM_EDGES)
        self._X_ek_sum_e = cpu_array(self._Z_e_start)
        self._sharing_mean_1 = as_scattered_gpu_arrray(self._Z_e_start / K)
        self._sharing_mean_2 = as_scattered_gpu_arrray(self._Z_e_start / K)
        self._sharing_dual = gpu_scattered_zeros((NUM_EDGES,))
        # The objective convergance tolerance for the MLU problem _MUST_ stricter than the
        # tolerance for the distributed algorithm itself.
        assert self._solver_params.ConvTol >= self._mlu_params.ConvTol, \
            f"{self._solver_params.ConvTol} < {self._mlu_params.ConvTol}"
        # TODO: Find a better way to handle `Rho` and `Alpha` here ...
        # self._mlu_solver = self._mlu_solver_cls(NUM_EDGES, self._capacities, self._mlu_params)
        self._mlu_solver = self._mlu_solver_cls(NUM_EDGES, np.ones_like(self._capacities), self._mlu_params)
        self._mlu_solver.rho = self._solver_params.Rho
        self._mlu_solver.alpha = self._alpha

        self._outer_admm_wrapper = ADMMWrapper(NUM_EDGES, self._solver_params.Rho)
        self._outer_admm_wrapper.initialize(self._X_ek_sum_e)

        self._dense_solver = DenseSolver(
            self._X_ek_start, self._NNT_M, self._M_MASK,
            self._solver_params.Gamma, self._solver_params.SwitchIterations
        )

    def initialize_to(self, assignment: CPUArray):
        raise NotImplementedError
        
    def _make_variables(self):
        assert self._mlu_solver is not None
        self._mlu_solver._make_variables()
    
    def _get_F(self) -> ScatteredGPUArray:
        return as_scattered_gpu_arrray(self._outer_admm_wrapper.get_X_step_bias())
    
    def _set_X_ek(self):
        self._X_ek = self._dense_solver.get_X_ek()
        np.multiply(self._X_ek, cpu_array(self._capacities)[:, None], out=self._X_ek)
        remove_noisy_loops(self._X_ek, self._graph)
    
    def _add_constraints(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_constraints()
    
    @record_cpu_runtime('Controller-Update')
    def _update_controller_objective(self):
        assert self._mlu_solver is not None
        self._mlu_solver.update_F_m(-self._outer_admm_wrapper.get_Z_step_bias())
    
    def _add_objective(self):
        assert self._mlu_solver is not None
        self._mlu_solver._add_objective()

    @record_gpu_runtime('NetworkUpdate')
    @record_reserved_gpu_memory('reserved-NetworkUpdate')
    def _do_network_update(self) -> float:
        sharing_bias = zip_map(
            [self._sharing_mean_1, self._sharing_mean_2, self._sharing_dual],
            lambda a, b, c: a - b + c
        )
        self._dense_solver.update(sharing_bias)    
    
    @record_gpu_runtime('Sharing-Mean-1')
    def _update_sharing_mean_1(self):
        self._sharing_mean_1 = reduce_in_place(
            self._dense_solver.X_ek,
            lambda x: cp.mean(x, axis=1)
        )
    
    @record_gpu_runtime('Sharing-Mean-2')
    def _update_sharing_mean_2(self):
        assert self._mlu_solver is not None

        K = len(self._commodity_list)
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        U_E = self._sharing_dual
        X_BAR_E = self._sharing_mean_1
        F_E = self._get_F()
        self._sharing_mean_2 = zip_map(
            [F_E, U_E, X_BAR_E],
            lambda f, u, xbar: (f / K + (ETA/RHO) * (u + xbar)) / (1 + (ETA/RHO))
        )

    @record_gpu_runtime('Sharing-Dual')
    def _update_sharing_dual(self):
        assert self._mlu_solver is not None
        
        self._sharing_dual = zip_map(
            [self._sharing_mean_1, self._sharing_mean_2, self._sharing_dual],
            lambda xbar, pbar, dual: dual + (xbar - pbar)
        )

    @record_gpu_runtime('Update-Reconvene')
    @record_reserved_gpu_memory('reserved-Reconvene')
    def _reconvene_network_updates(self):
        self._update_sharing_mean_1()
        self._update_sharing_mean_2()
        self._update_sharing_dual()

    @record_cpu_runtime('Update-X-EK-SUM')
    def _update_X_ek_sum(self):
        self._X_ek_sum_e = len(self._commodity_list) * as_cpu_array(self._sharing_mean_1)
        self._outer_admm_wrapper.record_X_update(self._X_ek_sum_e)
    
    @record_cpu_runtime('Update-Re')
    def _update_r_e(self):
        assert self._mlu_solver is not None

        self._outer_admm_wrapper.record_Z_update(self._get_Z_value())
        self._outer_admm_wrapper.update_dual_var(True)

    def close(self):
        if self._mlu_solver is not None:
            self._mlu_solver.close()
    
    def make_lp(self):
        self.initialize()
        super().make_lp()
    
    def reset(self, with_params: False):
        self._mlu_solver.reset(with_params)
    
    @record_cpu_runtime('Solve')
    @record_reserved_gpu_memory('reserved-Solve')
    def solve(self, params: SolverParams = None) -> float:
        self.check_result = None
        MODEL_CONTROLLER = self._mlu_solver
        PARAMS = self._solver_params if params is None else params

        try:
            t = time.time()
            self._update_controller_objective()
            MODEL_CONTROLLER.solve()
            self._update_r_e()
            progress_bar = ShortTQDM(range(PARAMS.OuterLoopRounds))
            for _ in progress_bar:
                for i in reversed(range(PARAMS.InnerLoopRounds)):
                    self._do_network_update()
                    if i > 0 and self._reconvene_network_updates():
                        break
                self._reconvene_network_updates()
                self._update_X_ek_sum()
                self._update_controller_objective()
                MODEL_CONTROLLER.solve()
                self._update_r_e()
                max_util = float(cp.max(self._sharing_mean_1[0]) * len(self._commodity_list))
                self._objective_trace.append(float(self.objective_value), max_util)
                # Inner loop infeasibility is usually very small, no need to bother with it!
                err = self._outer_admm_wrapper.infeasibility
                self._objective_gap_trace.append(self._outer_admm_wrapper.infeasibility)
                progress_bar.set_postfix({
                    'Cont. Util.': str(round(self._mlu_solver.current_u, 4)),
                    'Net. Util.': str(round(max_util, 4)),
                    'Outer Inf.': str(round(err, 4))
                })
                if err < PARAMS.ConvTol:
                    print(as_success("Crossed the convergance bound. Breaking early ..."))
                    break                
            self._set_X_ek()
            return time.time() - t
        except ControllerMLUException as e:
            print(f'MLU solver failed: {e}')
            return -1
        except SolutionInterrupted:
            self._set_X_ek()
            return time.time() - t
    
    def check(self):
        eval_params = self._problem_description.EvalParams
        # Check if outer and inner ADMM loops are in consensus
        outer_admm_consensus_test(self._X_ek_sum_e, self._get_Z_value(), eval_params=eval_params)
        # TODO: Do we need to load into RAM?
        inner_admm_consensus_test(as_cpu_array(self._sharing_mean_1), as_cpu_array(self._sharing_mean_2), eval_params=eval_params)
        super().check()

    def update_traffic_matrix(self, tm):
        raise NotImplementedError
    
    def initialize_to(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def set_target(self, solution: EdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def add_solution_elements(self, solution):
        raise NotImplementedError


import jsonargparse

def centralized_gpuadmm_solver_params_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(GPUParams, 'SolverParams', help='GPU-ADMM Solver Params')
    return parser


def parse_centralized_gpuadmm_solver_params(args: jsonargparse.Namespace) -> GPUParams:
    return GPUParams.make_from_args(args.SolverParams)
