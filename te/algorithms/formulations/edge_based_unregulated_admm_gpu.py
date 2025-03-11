import math
import time
import tqdm
import gurobipy
import cupy as cp
import numpy as np
import networkx as nx
import te.constants
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
from gurobipy import GRB, GurobiError
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.algorithms.solution import GurobiEdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from te.algorithms.sub_algorithms.pgd import do_gpu_plain_pgd_with_step_reduction
from te.algorithms.sub_algorithms.feasible_assignment import get_feasible_flow_assignment
from topologies.utils import (get_edge_indexing, get_graph_M_matrix, 
                              get_adjacency_null_space)
from te.algorithms.utils import (check_capacity_constraint, optimize_or_scream, make_model, 
                                 get_solution_maximum_utilization, check_centralized_flow_conservation,
                                 careful_norm, careful_norm_squared, as_fail, as_warning)
from te.algorithms.statistics.helpers import (record_cpu_runtime, record_gpu_runtime, record_reserved_gpu_memory, 
                                              record_used_gpu_memory)
from te.algorithms.gpu_utils import *


@dataclass
class GPUUnregulatedADMMSolverParams(GurobiSolverParams):
    """
    :param `NumberOfEpochs`: Number of total controller + network iterations
    :param `NumberOfNetworkUpdates`: Number of network updates for each epoch
    :param `Rho`: Outer ADMM step size
    :param `Eta`: Inner ADMM step size
    :param `Gamma`: PGD step size (mandatory, cannot use exact line-search here)
    :param `PGDIterations`: Number of PGD iterations for each commodity
    :param `Kappa`: PGD step size reduction factor. Must be
    :param `UseVariableRho`: Whether or not to use variable step sizes for ADMM
    :param `Mu`: Primal/Dual residual bound factor
    :param `TauIncrease`: Multiplicative step size increase factor
    :param `TauDecrease`: Multiplicative step size decrease factor
    :param `BigTheta`: Loose error bound for the whole solution
    :param `BigGamma`: Tight error bound for controller solution
    :param `FloatPrecision`: Floating point operation precision. A string choice
                             between `double`, `single` and `half`
    :param `Seed`: RNG seed
    """
    NumberOfEpochs: int = 20
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Gamma: float = 1.0
    PGDIterations: int = 5
    UseVariableRho: bool = True
    Kappa: float = 0.5
    Mu: float = te.constants.DEFAULT_MU
    TauIncrease: float = te.constants.DEFAULT_TAU_INC 
    TauDecrease: float = te.constants.DEFAULT_TAU_DEC 
    BigTheta: float = te.constants.DEFAULT_BIG_THETA
    BigGamma: float = te.constants.DEFAULT_BIG_GAMMA
    FloatPrecision: Literal['half', 'single', 'double'] = 'single'
    Seed: int = te.constants.DEFAULT_SEED

    def __post_init__(self):
        if self.Mu < 2:
            as_warning(f"Mu = {self.Mu}: Small values of `Mu` can hinder convergence.")
        assert self.Kappa >= 0 and self.Kappa <=1,\
            as_fail(f"Kappa = {self.Kappa}: Values of `Kappa` MUST be within `[0, 1]`")
        if self.Rho > self.Eta:
            as_warning(f"Outer ADMM step size (`Rho`) = {self.Rho} is strictly larger "
                       f"than inner ADMM step size (`Eta`) = {self.Eta}. This is almost never beneficial.")
        assert self.FloatPrecision in {'half', 'single', 'double'}, \
            as_fail(f"Unknown float precision string specifier {self.FloatPrecision}")
        set_global_precision(self.FloatPrecision)


class GPUUnregulatedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: GPUUnregulatedADMMSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)

        """
        These two matrixes always appear next to Y_tk,
        which lives in the GPU memory. As such, we would be
        much better off keeping them in GPU as well.
        They each require `m x T` and `m x m` entries, and are quite
        small compared to other matrices, so they have little memory
        footprint.
        """
        self._NULL_M: Optional[GPUArray] = None
        self._NNT_M: Optional[GPUArray] = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None
        self._target_u: Optional[float] = None

        self._capacities: Optional[CPUArray] = None
        self._Xo_e_start: Optional[CPUArray] = None
        self._Xo_e: Optional[gurobipy.tupledict] = None
        self._Zo_e: Optional[CPUArray] = None
        self._Xo_e_sol: Optional[CPUArray] = None
        self._Zo_e_old: Optional[CPUArray] = None
        self._r_e_old: Optional[CPUArray] = None
        self._utility: Optional[gurobipy.Var] = None
        self._capacity_constraints: List[gurobipy.Constr] = None

        self._X_ek_start: Optional[GPUArray] = None
        """
        An `n x K` matrix, this is our first heavy hitter in terms of
        memory. This matrix is involved in the PGD step as input, and
        as such must be kept in GPU memory.
        """
        self._X_ek: Optional[CPUArray] = None
        """
        The `n x K` assignment matrix, we need not keep this matrix in the
        GPU memory, as it is only reconstructed when the algorithm is finished.
        """
        self._Y_tk: Optional[GPUArray] = None
        """
        This matrix is one of the inputs of the inner PGD loop, and
        is updated at the end of it.
        It is much better to keep it in GPU memory, but it is very large
        (it has `T x K` entries).
        """
        self._Y_tk_old: Optional[GPUArray] = None
        """
        This matrix is involved in calculating the dual residual of the 
        inner ADMM loop, which adjusts the step size.
        This matrix can get large, and passing it to CPU is a big hit.
        For very large problems where memory is tight, we would be much
        better off just disabling variable step sizes.
        """
        self._P_bar_t: Optional[GPUArray] = None
        self._Y_bar_t: Optional[GPUArray] = None
        """
        These running means are updated within the inner ADMM loop.
        They are small enough to be kept in GPU memory comfortably.
        """
        self._P_bar_t_old: Optional[GPUArray] = None
        self._Y_bar_t_old: Optional[GPUArray] = None
        """
        This matrix is involved in calculating the dual residual of the 
        inner ADMM loop, which adjusts the step size.
        Small enough to be kept in GPU memory, but can be ignored when
        step size is fixed.
        """
        self._lambda_ek: Optional[GPUArray] = None
        """
        The main entity to keep in GPU memory. This is the main heavy-hitter,
        it has `n x K` elements.
        """
        self._r_e: Optional[CPUArray] = None
        self._u_t: Optional[GPUArray] = None
        """
        Outer and inner ADMM dual variables.
        There is no reason to keep the outer one in GPU, but
        there are benefits to keeping the inner one.
        """

        self._outer_primal_residual_norm: float = None
        self._outer_dual_residual_norm: float = None
        self._inner_primal_residual_norm: float = None
        self._inner_dual_residual_norm: float = None

        self._rho_coeff: Optional[float] = None
        self._rho_coeff_trace: List[float] = []
        self._eta_coeff: Optional[float] = None
        self._eta_coeff_trace: List[float] = []

        self._objective_trace = []
        self._objective_gap_trace = []

        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._report_problem_size()

    @property
    def alg_name(self) -> str:
        return 'GPU Unregulated ADMM'

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
    def rho_coeff_trace(self) -> List[float]:
        return self._rho_coeff_trace

    @property
    def eta_coeff_trace(self) -> List[float]:
        return self._eta_coeff_trace

    @property
    def objective_value(self) -> float:
        return self._utility.X
    
    @property
    def objective_trace(self) -> Optional[List[float]]:
        return self._objective_trace

    @property
    def objective_gap_trace(self) -> Optional[List[float]]:
        return self._objective_gap_trace
    
    @property
    def assignments(self) -> CPUArray:
        assert self._X_ek is not None
        return self._X_ek

    def _set_initial_feasible_solution(self):
        self._X_ek_start = as_gpu_array(get_feasible_flow_assignment(self._graph, self._commodity_list))
        self._Xo_e_start = as_cpu_array(cp.sum(self._X_ek_start, axis=1))
    
    def _set_NULL_M(self):
        M = self._M
        assert len(M.shape) == 2
        m, n = M.shape
        assert m < n
        N = as_gpu_array(get_adjacency_null_space(M))
        T = N.shape[1]
        # TODO: This is off by 1, since the columns of `M` are not independent
        # assert T == (n - m), f'{n}, {m}, {T}'
        self._NULL_M = N
        self._NNT_M = N @ N.T
        self._T = T
        self._NUM_EDGES = n
        self._rho_coeff = 1
        self._eta_coeff = 1
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        K = len(self._commodity_list)
        NUM_EDGES = self._NUM_EDGES
        self._capacities = as_cpu_array([item[-1] for item in self._graph.edges(data='capacity')])
        self._r_e = cpu_zeros(shape=(NUM_EDGES,))
        self._u_t = gpu_zeros(shape=(T,))
        self._Zo_e = as_cpu_array(self._Xo_e_start)
        self._P_bar_t = gpu_zeros((T,))
        self._Y_bar_t = gpu_zeros((T,))
        self._Y_tk = gpu_zeros((T, K))
        self._X_ek = as_cpu_array(self._X_ek_start)
        self._lambda_ek = gpu_zeros((NUM_EDGES, K))
    
    def _report_problem_size(self):
        M = len(self._graph.nodes)
        N = len(self._graph.edges)
        T = self._T
        K = len(self._commodity_list)

        print(f"Graph Size: {M} nodes | {N} edges")
        print(f"Number of commodities: {K}")
        print(f"Nullity of commodity assignment matrix: {T}")
        print("-"*60)
        print("CONTROLLER PROBLEM:\n" +
              f"\t TOTAL NUMBER OF VARIABLES: {N + 1}\n"
              f"\t TOTAL NUMBER OF CONSTRAINTS: {N + 1}\n")
        print("-"*60)
        print("NODE PROBLEM:\n" +
              f"\t NUMBER OF INDEPENDENT QPs PER NODE: {M - 1}\n"
              f"\t NUMBER OF VARIABLES PER QP PER NODE: {T}\n"
              f"\t NUMBER CONSTRAINTS PER QP PER NODE: {T}\n")

    def initialize_to(self, assignment: CPUArray):
        raise NotImplementedError
        
    def _make_variables(self):
        assert self._model_controller is None
        
        NUM_EDGES = self._NUM_EDGES

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        PARAMS = self._solver_params
        MODEL_CONTROLLER: gurobipy.Model = \
            make_model('EdgeBasedDistributedTE_Controller', params=PARAMS, env=ENV, BarConvTol=PARAMS.BigGamma)
        
        self._Xo_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name=f'XO_E')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        # Set starting values ...
        self._model_controller = MODEL_CONTROLLER
    
    def _get_F(self) -> GPUArray:
        return as_gpu_array(self._Zo_e + self._r_e - self._Xo_e_start)
    
    def _set_X_ek(self):
        self._X_ek = as_cpu_array(self._X_ek_start + self._NULL_M @ self._Y_tk)
    
    @record_gpu_runtime('GeetXKSum')
    def _get_X_k_sum(self) -> CPUArray:
        assert self._X_ek is not None
        return as_cpu_array(cp.sum(self._X_ek_start + self._NULL_M @ self._Y_tk, axis=1))
    
    def _add_constraints(self):
        assert self._model_controller is not None

        GRAPH = self._graph
        XO_E = self._Xo_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        # Capacity constraint
        self._capacity_constraints = [
            MODEL_CONTROLLER.addConstr(XO_E[i] / c_e <= UTILITY)
                for i, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity'))
        ]
    
    @record_cpu_runtime('ControllerObjectiveUpdate')
    def _update_controller_objective(self):
        NUM_EDGES = self._NUM_EDGES
        UTILITY = self._utility
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        R_E = self._r_e
        RHO = self._solver_params.Rho * self._rho_coeff
        MODEL_CONTROLLER = self._model_controller

        """
        Controller objective is:
            u + rho/2 sum_e (X_oe - Z_oe + r_e)^2
        """

        OBJECTIVE_CONTROLLER = gurobipy.QuadExpr()
        OBJECTIVE_CONTROLLER.addTerms(1, UTILITY)
        for e in range(NUM_EDGES):
            OBJECTIVE_CONTROLLER += (RHO/2) * (XO_E[e] - ZO_E[e] + R_E[e]) ** 2
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _add_objective(self):
        assert self._model_controller is not None

        self._update_controller_objective()

    @record_gpu_runtime('GetCurrentC')
    def _get_current_C(self) -> GPUArray:
        Y_TK = self._Y_tk
        Y_BAR = self._Y_bar_t
        P_BAR = self._P_bar_t
        U_T = self._u_t
        return Y_TK - cp.expand_dims(Y_BAR - P_BAR + U_T, axis=1)

    @record_gpu_runtime('NetworkUpdate')
    @record_reserved_gpu_memory('reserverd-NetworkUpdate')
    def _do_network_update(self, epoch: int) -> float:
        PARAMS = self._solver_params
        GAMMA = PARAMS.Gamma
        KAPPA = PARAMS.Kappa
        PGD_ITERS = PARAMS.PGDIterations
        NULL_M = self._NULL_M
        NNT_M = self._NNT_M
        C_TK = self._get_current_C()
        LAMBDA_EK = self._lambda_ek
        X_EK_START = self._X_ek_start
        
        t_start = time.time()
        lambda_block, y_block = do_gpu_plain_pgd_with_step_reduction(
            LAMBDA_EK, X_EK_START, NNT_M, NULL_M, C_TK, GAMMA, 
            PGD_ITERS, KAPPA, epoch)
        
        self._lambda_ek = lambda_block
        self._Y_tk = y_block
        return time.time() - t_start
    
    @record_gpu_runtime('YBarUpdate')
    def _update_Y_bar(self):
        # Record old `Y_tk` and `Y_bar_t` value only when using variable step sizes
        if self._solver_params.UseVariableRho:
            self._Y_bar_t_old = np.copy(self._Y_bar_t)
            self._Y_tk_old = cp.copy(self._Y_tk)
        self._Y_bar_t = cp.average(self._Y_tk, axis=1)
    
    @record_gpu_runtime('PBarUpdate')
    def _update_P_bar(self):
        assert self._model_controller is not None
        
        """
        The update rule for `P_bar` is:

            P_bar \gets (NULL_M^T F + (\eta/\rho) (u + Y_bar)) / (K + (\eta/\rho))
        """

        K = len(self._commodity_list)
        PARAMS = self._solver_params
        ETA = PARAMS.Eta * self._eta_coeff
        RHO = PARAMS.Rho * self._rho_coeff
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        F_E = self._get_F()
        NULL_M = self._NULL_M
        P_BAR_T = (NULL_M.T @ F_E + (ETA/RHO) * (U_T + Y_BAR_T)) / (K + (ETA/RHO))
        self._P_bar_t = P_BAR_T

        if PARAMS.UseVariableRho:
            self._P_bar_t_old = cp.array(self._P_bar_t)
            self._inner_primal_residual_norm = careful_norm((P_BAR_T - Y_BAR_T), scaled=True)
            self._inner_dual_residual_norm = careful_norm(
                (self._Y_tk - self._Y_tk_old) + 
                (P_BAR_T - self._P_bar_t_old)[:, np.newaxis] +
                (self._Y_bar_t_old - Y_BAR_T)[:, np.newaxis],
                scaled=True
            ) * self._eta_coeff
    
    @record_gpu_runtime('UUpdate')
    def _update_u_t(self):
        assert self._model_controller is not None
        
        """
        The update rule for `u` is:

            u \gets u + (Y_bar - P_bar)
        """

        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t

        self._u_t = U_T + (Y_BAR_T - P_BAR_T)
    
    @record_gpu_runtime('Reconvene')
    @record_reserved_gpu_memory('reserverd-Reconvene')
    def _reconvene_network_updates(self):
        self._update_Y_bar()
        self._update_P_bar()
        self._update_eta_coeff()
        self._update_u_t()

    @record_cpu_runtime('ZOUpdate')
    def _update_Zo_e(self):
        assert self._model_controller is not None

        """
        The update rule for Zo_e is:
            Zo_e \gets (X_oe + \sum_k X_ke)/2
        """
        
        NUM_EDGES = self._NUM_EDGES
        XO_E = self._Xo_e
        XO_E_ = np.array([XO_E[e].X for e in range(NUM_EDGES)], dtype=np.float16)
        X_KE_SUM_E = self._get_X_k_sum()
        PARAMS = self._solver_params
        Zo_e = (XO_E_ + X_KE_SUM_E) / 2
        self._Zo_e_old = np.array(self._Zo_e, dtype=np.float16)
        self._Xo_e_sol = XO_E_
        self._Zo_e = Zo_e
        if PARAMS.UseVariableRho:
            self._outer_primal_residual_norm = careful_norm((Zo_e - XO_E_), scaled=True) + careful_norm((Zo_e - X_KE_SUM_E), scaled=True)
            self._outer_dual_residual_norm = careful_norm((Zo_e - self._Zo_e_old), scaled=True) * self._rho_coeff
    
    def _update_rho_coeff(self):
        PARAMS = self._solver_params
        primal_norm = self._outer_primal_residual_norm
        dual_norm = self._outer_dual_residual_norm
        if PARAMS.UseVariableRho:
            self._rho_coeff_trace.append(self._rho_coeff)
            if primal_norm > PARAMS.Mu * dual_norm:
                self._rho_coeff *= PARAMS.TauIncrease
                self._r_e = (self._r_e / PARAMS.TauIncrease)
            elif dual_norm > PARAMS.Mu * primal_norm:
                self._rho_coeff /= PARAMS.TauDecrease
                self._r_e = (self._r_e * PARAMS.TauDecrease)
    
    def _update_eta_coeff(self):
        PARAMS = self._solver_params
        primal_norm = self._inner_primal_residual_norm
        dual_norm = self._inner_dual_residual_norm
        if PARAMS.UseVariableRho:
            self._eta_coeff_trace.append(self._eta_coeff)
            if primal_norm > PARAMS.Mu * dual_norm:
                self._eta_coeff *= PARAMS.TauIncrease
                self._u_t = (self._u_t / PARAMS.TauIncrease)
            elif dual_norm > PARAMS.Mu * primal_norm:
                self._eta_coeff /= PARAMS.TauDecrease
                self._u_t = (self._u_t * PARAMS.TauDecrease)
    
    @record_cpu_runtime('REUpdate')
    def _update_r_e(self):
        assert self._model_controller is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        R_E = self._r_e
        XO_E = self._Xo_e
        NUM_EDGES = self._NUM_EDGES
        X_KE_SUM_E = self._get_X_k_sum()

        r_e = np.zeros((NUM_EDGES,))
        for e in range(NUM_EDGES):
            r_e[e] = R_E[e] + (XO_E[e].X - X_KE_SUM_E[e]) / 2
        self._r_e_old = np.copy(self._r_e)
        self._r_e = r_e
    
    # def _get_controller_objective_shifts(self) -> Tuple[float, float]:
    #     XO_E = self._Xo_e_sol
    #     Z_HAT_OLD = self._Zo_e_old - self._r_e_old
    #     Z_HAT = self._Zo_e - self._r_e
    #     RHO = self._solver_params.Rho * self._rho_coeff
    #     LAMBDA_E = np.array([constr.Pi for constr in self._capacity_constraints])
    #     C_E = self._capacities

    #     primal_shift = (RHO/2) * (careful_norm_squared(XO_E - Z_HAT) - careful_norm_squared(XO_E - Z_HAT_OLD))
    #     dual_shift = -(RHO/2) * (careful_norm_squared(Z_HAT) - careful_norm_squared(Z_HAT_OLD)) \
    #                  -np.dot(np.divide(LAMBDA_E, C_E), (Z_HAT - Z_HAT_OLD))
    #     return primal_shift, dual_shift
    
    # def _check_objective_gap(self) -> bool:
    #     BIG_THETA = self._solver_params.BigTheta
    #     if self._target_u:
    #         actual_utilization = get_solution_maximum_utilization(self._X_ek, self.graph)
    #         apparent_utulization = self._utility.X
    #         actual_gap = np.abs(actual_utilization - self._target_u) / self._target_u
    #         apparent_gap = np.abs(apparent_utulization - self._target_u) / self._target_u
    #         relative_gap = max(actual_gap, apparent_gap)
    #         print(f"Utilization gap: {str(round(max(actual_gap, apparent_gap) * 100, 4))} percent")
    #     else:
    #         primal_shift_controller, dual_shift_controller = self._get_controller_objective_shifts()
    #         primal_shift_network, dual_shift_network = self._get_network_objective_shifts()
    #         primal_objective = self._model_controller.ObjVal + self._get_network_objective()
    #         primal_shift = primal_shift_controller + primal_shift_network
    #         dual_shift = dual_shift_controller + dual_shift_network
    #         relative_gap = (np.abs(primal_shift) + np.abs(dual_shift)) / np.abs(primal_objective + primal_shift)
    #         print(f"Objective gap: {str(round(relative_gap * 100, 4))} percent")
    #     self._objective_gap_trace.append(relative_gap)
    #     return relative_gap <= BIG_THETA
    
    def close(self):
        if self._model_controller:
            self._model_controller.close()
        if self._env:
            self._env.close()
        # self.proc_pool.close()
    
    def make_lp(self):
        t_start = time.time()
        print("Starting to create the model")
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        print(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds.")
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        if with_params:
            self._model_controller.resetParams()
    
    @record_cpu_runtime('Solve')
    @record_reserved_gpu_memory('reserved-Solve')
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._model_controller
        PARAMS = self._solver_params

        total_runtime = 0
        epoch = 0
        max_iters = PARAMS.NumberOfEpochs
        try:
            # for epoch in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
            while True:
                if ((max_iters is not None) and (epoch == max_iters)):
                    break
                t_network = 0

                # First, let the controller decide what the utilization is
                optimize_or_scream(MODEL_CONTROLLER)
                # self._check_objective_gap()

                # Now, do in-network optimization
                for i in reversed(range(PARAMS.NumberOfNetworkUpdates)):
                    t_network += self._do_network_update(epoch)
                    """
                    Defer the update for the last iteration.
                    This final update is moot, since the controller has to
                    update `Zo` anyway after this, which affects the final
                    result of this.
                    So let the controller update itself first, and then push
                    the chnages to the dual variables within the network.
                    """
                    if i > 0:
                        self._reconvene_network_updates()
                # self._set_X_ek()

                # print(f"Total Network Update Gap: {total_gap}")

                # Now that we have non-zero flow assignments, inform the controller
                self._update_Zo_e()
                self._update_rho_coeff()
                self._update_r_e()

                # Finalize the last update that we deferred ...
                self._reconvene_network_updates()

                # Update the objectives and start again
                self._update_controller_objective()

                # Houskeeping
                self._objective_trace.append(self._utility.X)
                total_runtime += MODEL_CONTROLLER.Runtime + t_network
                # Check primal-dual objective gap
                # if ((epoch > 0) and (self._check_objective_gap())):
                #     break
                epoch += 1
            self._set_X_ek()
            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        NUM_EDGES = self._NUM_EDGES
        T = self._T
        PARAMS = self._solver_params

        _atol = 0.0 if feasibility_tol is None else feasibility_tol
        _rtol = 0.0 if feasibility_ratio is None else feasibility_ratio

        # TODO: This is not numerically stable ...
        def in_consensus(primal, pair):
            if abs(primal - pair) < te.constants.FLOAT_RES:
                return True
            return math.isclose(primal, pair, rel_tol=_rtol, abs_tol=_atol)
            # if feasibility_tol is not None:
            #     return abs(primal - pair) < feasibility_tol
            # return abs((primal - pair) / (primal + te.constants.FLOAT_RES)) < feasibility_ratio

        # Are outer ADMM pairs in consensus?
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        for e in range(NUM_EDGES):
            primal = XO_E[e].X
            pair = ZO_E[e]
            primal_str = f'{primal:.4f}'
            pair_str = f'{pair:.4f}'
            if not in_consensus(primal, pair):
                print(as_fail(f"Edge {e} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
        
        # Are inner ADMM pairs in consensus?
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        for t in range(T):
            primal = Y_BAR_T[t]
            pair = P_BAR_T[t]
            primal_str = f'{primal:.4f}'
            pair_str = f'{pair:.4f}'
            if not in_consensus(primal, pair):
                print(as_fail(f"Axis {t} --> Inner ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
        
        # Now, check flow conservation ...
        X_EK = self._X_ek
        # check_centralized_flow_conservation(X_EK, self._graph, self._commodity_list, PARAMS.FeasibilityTol)
        check_capacity_constraint(
            X_EK, self._graph, self._commodity_list, 
            feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio
        )

    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        COMMODITIES = self._commodity_list
        X_EK = self._X_ek
        GRAPH = self._graph
        INDICES = self._edge_indexing

        return [
            (
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        X_EK[INDICES[(v, commodity.destination)], i] \
                            for v in GRAPH.predecessors(commodity.destination)
                    ])
                ),
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        X_EK[INDICES[(commodity.source, v)], i] \
                            for v in GRAPH.successors(commodity.source)
                    ])
                )
            )
            for i, commodity in enumerate(COMMODITIES)
        ]
    
    def update_traffic_matrix(self, tm):
        raise NotImplementedError
    
    def initialize_to(self, solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def set_target(self, solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution):
        raise NotImplementedError
    
    def add_solution_elements(self, solution):
        raise NotImplementedError

