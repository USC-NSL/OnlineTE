import time
import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from multiprocessing import Pool
from typing import List, Tuple, Optional
from dataclasses import dataclass
from gurobipy import GRB, GurobiError
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.algorithms.solution import GurobiEdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from te.algorithms.sub_algorithms.pgd import (do_iterative_plain_pgd, do_iterative_pgd_with_exact_line_search,
                                              do_block_plain_pgd, do_block_pgd_with_exact_line_search)
from topologies.utils import (get_edge_indexing, get_graph_M_matrix, 
                              get_adjacency_null_space, get_feasible_flow_assignment)
from te.algorithms.utils import (check_capacity_constraint, optimize_or_scream, make_model, 
                                 get_solution_maximum_utilization, as_fail,
                                 careful_norm, careful_norm_squared)


@dataclass
class RegularizedADMMSolverParams(GurobiSolverParams):
    """
    :param `NumberOfEpochs`: If not `None`, we only do this many iterations, otherwise,
                             will keep hammering until stopping criterion has been met.
    :param `NumberOfNetworkUpdates`: Number of network updates for each epoch
    :param `Rho`: Outer ADMM step size
    :param `Eta`: Inner ADMM step size
    :param `Epsilon`: Regularization factor
    :param `Gamma`: PGD step size (if `None`, will use exact line search)
    :param `PGDConvTol: PGD convergence tolerance
    :param `PGDIterations`: Number of PGD iterations for each commodity
    :param `UseVariableRho`: Whether or not to use variable step sizes for ADMM
    :param `Mu`: Primal/Dual residual bound factor
    :param `TauIncrease`: Multiplicative step size increase factor
    :param `TauDecrease`: Multiplicative step size decrease factor
    :param `BigTheta`: Loose error bound for the whole solution
    :param `BigGamma`: Tight error bound for controller solution
    :param `Alpha`: Over-relaxation parameter
    :param `NumWorkers`: Number of worker nodes to partition commodities on to
    :param `BlockMode`: If `True`, distributes commodities in big blocks among
                        worker nodes rather than one-by-one.
    :param `CheckBlockConv`: If `True`, checks individual commodity convergence
                             when running under `BlockMode`. Reduces processing
                             time if we have high degree of multi-processing.
    :param `Seed`: RNG seed
    """
    NumberOfEpochs: Optional[int] = None
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Epsilon: float = te.constants.DEFAULT_EPSILON_KE
    Gamma: Optional[float] = None
    PGDConvTol: float = 1e-8
    PGDIterations: int = 5
    UseVariableRho: bool = True
    Mu: float = te.constants.DEFAULT_MU
    TauIncrease: float = te.constants.DEFAULT_TAU_INC 
    TauDecrease: float = te.constants.DEFAULT_TAU_DEC 
    BigTheta: float = te.constants.DEFAULT_BIG_THETA
    BigGamma: float = te.constants.DEFAULT_BIG_GAMMA
    Alpha: float = 1
    NumWorkers: int = 1
    BlockMode: bool = False
    CheckBlockConv: bool = False
    Seed: int = te.constants.DEFAULT_SEED


class RegularizedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: RegularizedADMMSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._NULL_M: np.ndarray = None
        self._NNT_M: np.ndarray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None
        self._target_u: Optional[float] = None

        # List of edge capacities
        self._capacities: Optional[np.ndarray] = None
        # Both just vectors of length `T`
        self._Xo_e_sol: Optional[np.ndarray] = None
        self._Yo_t: Optional[gurobipy.tupledict] = None
        self._Zo_t: Optional[np.ndarray] = None
        self._Zo_t_old: Optional[np.ndarray] = None
        self._C_tk_old: Optional[np.ndarray] = None
        # Just a variable between 0 and 1
        self._utility: Optional[gurobipy.Var] = None
        # List of controller constraints
        self._capacity_constraints: List[gurobipy.Constr] = None
        self._controller_non_negativity_constraints: List[gurobipy.Constr] = None
        # This need not be managed by Gurobi
        self._X_ek_start: Optional[np.ndarray] = None
        self._X_e_start: Optional[np.ndarray] = None
        self._X_ek: Optional[np.ndarray] = None
        # Global value. A `T x K` matrix
        self._Y_tk: Optional[np.ndarray] = None
        self._Y_tk_old: Optional[np.ndarray] = None
        # ADMM running average variable. A vector of length `T`.
        self._P_bar_t: Optional[np.ndarray] = None
        self._P_bar_t_old: Optional[np.ndarray] = None
        self._Y_bar_t: Optional[np.ndarray] = None
        self._Y_bar_t_old: Optional[np.ndarray] = None
        # For PGD, an `N x K` matrix
        self._lambda_ek: Optional[np.ndarray] = None
        # Dual vairable of outer ADMM. A vector of length `T`.
        self._r_t: Optional[np.ndarray] = None
        self._r_t_old: Optional[np.ndarray] = None
        # Dual vairable of inner ADMM. A vector of length `T`.
        self._u_t: Optional[np.ndarray] = None
        # These residual norms will show how far from optimality we are
        self._outer_primal_residual_norm: float = None
        self._outer_dual_residual_norm: float = None
        self._inner_primal_residual_norm: float = None
        self._inner_dual_residual_norm: float = None
        # These coefficients give the current value of step sizes
        self._rho_coeff: Optional[float] = None
        self._eta_coeff: Optional[float] = None
        # These step size coefficient traces are used for debugging ...
        self._rho_coeff_trace: List[float] = []
        self._eta_coeff_trace: List[float] = []
        # The trace of the objective function as the algorithm progresses
        self._objective_trace = []
        # Estimate of the duality gap as the algorithm progresses
        self._objective_gap_trace = []

        self.proc_pool = Pool(processes=self._solver_params.NumWorkers)

        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._report_problem_size()

    @property
    def alg_name(self) -> str:
        return 'Multi-Proces Regularized ADMM'

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
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek
    
    @property
    def current_rho(self) -> float:
        return self._rho_coeff * self._solver_params.Rho
    
    @property
    def current_eta(self) -> float:
        return self._eta_coeff * self._solver_params.Eta

    def _set_initial_feasible_solution(self):
        self._X_ek_start = get_feasible_flow_assignment(self._graph, self._commodity_list)
        self._X_e_start = np.sum(self._X_ek_start, axis=1)
    
    def _set_NULL_M(self):
        M = self._M
        assert len(M.shape) == 2
        m, n = M.shape
        assert m < n
        N = get_adjacency_null_space(M)
        T = N.shape[1]
        # TODO: This is off by 1, since the columns of `M` are not independent
        # assert T == (n - m), f'{n}, {m}, {T}'
        assert np.allclose(np.matmul(N.T, N) - np.eye(T), 0)
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
        self._capacities = np.array([item[-1] for item in self._graph.edges(data='capacity')])
        self._X_ek = np.copy(self._X_ek_start)
        self._r_t = np.zeros(shape=(T,))
        self._u_t = np.zeros(shape=(T,))
        self._Zo_t = np.zeros(shape=(T,))
        self._P_bar_t = np.zeros((T,))
        self._Y_bar_t = np.zeros((T,))
        self._Y_tk = np.zeros((T, K))
        self._lambda_ek = np.zeros((NUM_EDGES, K))
    
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

    def initialize_to(self, assignment: np.ndarray):
        assert self._model_controller is not None
        self._X_ek = assignment
        self._Y_tk = self._NULL_M.T @ (assignment - self._X_ek_start)
        self._Yo_t = np.sum(self._Y_tk, axis=1)
        self._Zo_t = np.copy(self._Zo_t)
        self._Y_bar_t = np.average(self._Y_tk, axis=1)
        self._P_bar_t = np.copy(self._Y_bar_t)
        
    def _make_variables(self):
        assert self._model_controller is None
        
        T = self._T

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        PARAMS = self._solver_params
        MODEL_CONTROLLER: gurobipy.Model = \
            make_model('EdgeBasedDistributedTE_Controller', params=PARAMS, env=ENV, BarConvTol=PARAMS.BigGamma)
        
        self._Yo_t = MODEL_CONTROLLER.addVars(T, lb=-float('inf'), vtype=GRB.CONTINUOUS, name=f'YO_T')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        self._model_controller = MODEL_CONTROLLER
    
    def _set_X_ek(self):
        self._X_ek = np.clip(self._X_ek_start + self._NULL_M @ self._Y_tk, a_min=0, a_max=None)
    
    def _get_X_k_sum(self) -> np.ndarray:
        assert self._X_ek is not None
        return np.sum(self._X_ek, axis=1)
    
    def _add_constraints(self):
        assert self._model_controller is not None

        T = self._T
        NUM_EDGES = self._NUM_EDGES

        GRAPH = self._graph
        YO_T = self._Yo_t
        NULL_M = self._NULL_M
        X_E_START = self._X_e_start
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        total_flow_expressions: List[gurobipy.LinExpr] = [
            gurobipy.quicksum([X_E_START[e]] + [NULL_M[e, t] * YO_T[t] for t in range(T)])
                for e in range(NUM_EDGES)
        ]
        self._controller_non_negativity_constraints = [
            MODEL_CONTROLLER.addConstr(0 <= total_flow_expressions[e])
                for e in range(NUM_EDGES)
        ]
        self._capacity_constraints = [
            MODEL_CONTROLLER.addConstr(total_flow_expressions[e] <= UTILITY * c_e)
                for e, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity'))
        ]
    
    def _update_controller_objective(self):
        T = self._T
        UTILITY = self._utility
        YO_T = self._Yo_t
        ZO_T = self._Zo_t
        R_T = self._r_t
        RHO = self.current_rho
        MODEL_CONTROLLER = self._model_controller

        """
        Controller objective is:
            u + rho/2 sum_e (X_oe - Z_oe + r_e)^2
        """

        OBJECTIVE_CONTROLLER: gurobipy.QuadExpr = gurobipy.quicksum(
            [UTILITY] + [(RHO/2) * (YO_T[t] - ZO_T[t] + R_T[t]) ** 2 for t in range(T)]
        )
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _add_objective(self):
        assert self._model_controller is not None
        self._update_controller_objective()

    @staticmethod
    def do_plain_pgd(lambda_k: np.ndarray, x_k_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, c: np.ndarray, 
                     gamma: float, thresh: Optional[float], n_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        _c = x_k_0 + n @ c
        for i in range(n_iter):
            lambda_k_old = lambda_k
            lambda_k = np.clip(lambda_k - gamma * (nnt @ lambda_k + _c), a_min=0, a_max=None)
            if thresh and careful_norm(lambda_k - lambda_k_old) < thresh:
                break
        y_k = c + n.T @ lambda_k
        return lambda_k, y_k

    def _get_current_C(self) -> np.ndarray:
        ETA = self._solver_params.Eta
        EPSILON = self._solver_params.Epsilon
        Y_TK = self._Y_tk
        Y_BAR = self._Y_bar_t
        P_BAR = self._P_bar_t
        U_T = self._u_t
        return (ETA / (ETA + EPSILON)) * (Y_TK - np.expand_dims(Y_BAR - P_BAR + U_T, axis=1))

    def _update_rho_coeff(self):
        PARAMS = self._solver_params
        primal_norm = self._outer_primal_residual_norm
        dual_norm = self._outer_dual_residual_norm
        if PARAMS.UseVariableRho:
            self._rho_coeff_trace.append(self._rho_coeff)
            if primal_norm > PARAMS.Mu * dual_norm:
                self._rho_coeff *= PARAMS.TauIncrease
                self._r_t = (self._r_t / PARAMS.TauIncrease)
            elif dual_norm > PARAMS.Mu * primal_norm:
                self._rho_coeff /= PARAMS.TauDecrease
                self._r_t = (self._r_t * PARAMS.TauDecrease)
    
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
    
    def _update_Y_bar(self):
        self._Y_bar_t_old = np.copy(self._Y_bar_t)
        self._Y_tk_old = np.copy(self._Y_tk)
        self._Y_bar_t = np.average(self._Y_tk, axis=1)
    
    def _update_P_bar(self):
        assert self._model_controller is not None
        
        """
        The update rule for `P_bar` is:

            P_bar \gets (Z_o + r + alpha * (\eta/\rho) (u + alpha * Y_bar)) / (K + alpha^2 * (\eta/\rho))
        """

        K = len(self._commodity_list)
        ALPHA = self._solver_params.Alpha
        ETA = self._solver_params.Eta * self._eta_coeff
        RHO = self._solver_params.Rho * self._rho_coeff
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        ZO_T = self._Zo_t
        R_T = self._r_t
        self._P_bar_t_old = np.copy(self._P_bar_t)
        P_BAR_T = (ZO_T + R_T + ALPHA * (ETA/RHO) * (U_T + ALPHA * Y_BAR_T)) / (K + ALPHA**2 * (ETA/RHO))
        self._P_bar_t = P_BAR_T

        # TODO: These look really shifty ...
        self._inner_primal_residual_norm = careful_norm((P_BAR_T - Y_BAR_T), scaled=True)
        self._inner_dual_residual_norm = careful_norm(
            (self._Y_tk - self._Y_tk_old) + 
            (P_BAR_T - self._P_bar_t_old)[:, np.newaxis] +
            (self._Y_bar_t_old - Y_BAR_T)[:, np.newaxis],
            scaled=True
        ) * self._eta_coeff
    
    def _update_u_t(self):
        assert self._model_controller is not None
        
        """
        The update rule for `u` is:

            u \gets u + alpha * (Y_bar - P_bar)
        """

        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        ALPHA = self._solver_params.Alpha

        self._u_t = U_T + ALPHA * (Y_BAR_T - P_BAR_T)

    def _update_Zo_t(self):
        assert self._model_controller is not None

        """
        The update rule for Zo_t is:
            Zo_t \gets (Y_ot + \sum_k Y_tk)/2
        """
        
        T = self._T
        YO_T = self._Yo_t
        YO_T_ = np.array([YO_T[t].X for t in range(T)])
        Y_TK_SUM_T = np.sum(self._Y_tk, axis=1)
        ZO_T = (YO_T_ + Y_TK_SUM_T) / 2
        self._Zo_t_old = np.copy(self._Zo_t)
        self._Zo_t = ZO_T
        self._outer_primal_residual_norm = \
            careful_norm((ZO_T - YO_T_), scaled=True) + careful_norm((ZO_T - Y_TK_SUM_T), scaled=True)
        self._outer_dual_residual_norm = \
            careful_norm((ZO_T - self._Zo_t_old), scaled=True) * self._rho_coeff
    
    def _update_r_t(self):
        assert self._model_controller is not None

        """
        The update rule for r_e is:
            r_t \gets r_t + alpha * (Y_ot - \sum_k Y_tk)/2
        """

        T = self._T
        YO_T = self._Yo_t
        YO_T_ = np.array([YO_T[t].X for t in range(T)])
        Y_TK_SUM_T = np.sum(self._Y_tk, axis=1)
        ALPHA = self._solver_params.Alpha

        R_T = self._r_t + ALPHA * (YO_T_ - Y_TK_SUM_T) / 2
        self._r_t_old = np.copy(self._r_t)
        self._r_t = R_T

    def _do_network_update(self) -> float:
        GAMMA = self._solver_params.Gamma
        PGD_ITERS = self._solver_params.PGDIterations
        PGD_CONV_TOL = self._solver_params.PGDConvTol
        NUM_BLOCKS = self._solver_params.NumWorkers
        BLOCK_MODE = self._solver_params.BlockMode
        CHECK_BLOCK_CONV = self._solver_params.CheckBlockConv
        NULL_M = self._NULL_M
        NNT_M = self._NNT_M
        X_EK_START_BLOCKS = np.array_split(self._X_ek_start, NUM_BLOCKS, axis=1)
        LAMBDA_EK_BLOCKS = np.array_split(self._lambda_ek, NUM_BLOCKS, axis=1)
        C_TK = self._get_current_C()
        C_TK_BLOCKS = np.array_split(C_TK, NUM_BLOCKS, axis=1)
        
        if GAMMA is None:
            do_pgd = do_block_pgd_with_exact_line_search if BLOCK_MODE else do_iterative_pgd_with_exact_line_search
        else:
            do_pgd = do_block_plain_pgd if BLOCK_MODE else do_iterative_plain_pgd

        def pgd_iterator():
            for i in range(NUM_BLOCKS):
                if GAMMA is None and BLOCK_MODE:
                    yield (LAMBDA_EK_BLOCKS[i], X_EK_START_BLOCKS[i], NNT_M, NULL_M, C_TK_BLOCKS[i], PGD_CONV_TOL, PGD_ITERS, CHECK_BLOCK_CONV)
                elif GAMMA is None and not BLOCK_MODE:
                    yield (LAMBDA_EK_BLOCKS[i], X_EK_START_BLOCKS[i], NNT_M, NULL_M, C_TK_BLOCKS[i], PGD_CONV_TOL, PGD_ITERS)
                elif GAMMA is not None and BLOCK_MODE:
                    yield (LAMBDA_EK_BLOCKS[i], X_EK_START_BLOCKS[i], NNT_M, NULL_M, C_TK_BLOCKS[i], GAMMA, PGD_CONV_TOL, PGD_ITERS, CHECK_BLOCK_CONV)
                else:
                    yield (LAMBDA_EK_BLOCKS[i], X_EK_START_BLOCKS[i], NNT_M, NULL_M, C_TK_BLOCKS[i], GAMMA, PGD_CONV_TOL, PGD_ITERS)

        t_start = time.time()
        lambda_holder: List[np.ndarray] = []
        Y_holder: List[np.ndarray] = []
        for item in self.proc_pool.starmap(do_pgd, pgd_iterator()):
            lambda_block, y_block = item
            lambda_holder.append(lambda_block)
            Y_holder.append(y_block)
        self._lambda_ek = np.hstack(lambda_holder)
        self._Y_tk = np.hstack(Y_holder)
        self._C_tk_old = C_TK
        return time.time() - t_start

    def _reconvene_network_updates(self):
        self._update_Y_bar()
        self._update_P_bar()
        self._update_eta_coeff()
        self._update_u_t()
    
    def _get_controller_objective_shifts(self) -> Tuple[float, float]:
        XO_E = self._Xo_e_sol
        Z_HAT_OLD = self._Zo_t_old - self._r_t_old
        Z_HAT = self._Zo_t - self._r_t
        RHO = self.current_rho
        LAMBDA_E = np.array([constr.Pi for constr in self._capacity_constraints])
        C_E = self._capacities

        # TODO: These need to updated!
        primal_shift = (RHO/2) * (careful_norm_squared(XO_E - Z_HAT) - careful_norm_squared(XO_E - Z_HAT_OLD))
        dual_shift = -(RHO/2) * (careful_norm_squared(Z_HAT) - careful_norm_squared(Z_HAT_OLD)) \
                     -np.dot(np.divide(LAMBDA_E, C_E), (Z_HAT - Z_HAT_OLD))
        return primal_shift, dual_shift
    
    def _get_network_objective(self) -> float:
        Y_TK = self._Y_tk
        C_TK_OLD = self._C_tk_old
        return (1/2) * careful_norm_squared(Y_TK - C_TK_OLD)
    
    def _get_network_objective_shifts(self) -> Tuple[float, float]:
        n = self._NULL_M
        K = len(self._commodity_list)
        RHO = self.current_rho
        LAMBDA_EK = self._lambda_ek
        Y_TK = self._Y_tk
        C_TK = self._get_current_C()
        C_TK_OLD = self._C_tk_old

        # TODO: We need RHO here ... right?
        primal_shift = (RHO/2) * (careful_norm_squared(Y_TK - C_TK) - careful_norm_squared(Y_TK - C_TK_OLD))
        dual_shift = -(RHO/2) * np.sum([np.dot(LAMBDA_EK[:, k].T, n @ (C_TK[:, k] - C_TK_OLD[:, k])) for k in range(K)])
        return primal_shift, dual_shift
    
    def _check_objective_gap(self) -> bool:
        BIG_THETA = self._solver_params.BigTheta
        if self._target_u:
            actual_utilization = get_solution_maximum_utilization(self._X_ek, self.graph)
            apparent_utulization = self._utility.X
            actual_gap = np.abs(actual_utilization - self._target_u) / self._target_u
            apparent_gap = np.abs(apparent_utulization - self._target_u) / self._target_u
            relative_gap = max(actual_gap, apparent_gap)
            print(f"Utilization gap: {str(round(max(actual_gap, apparent_gap) * 100, 4))} percent")
        else:
            primal_shift_controller, dual_shift_controller = self._get_controller_objective_shifts()
            primal_shift_network, dual_shift_network = self._get_network_objective_shifts()
            primal_objective = self._model_controller.ObjVal + self._get_network_objective()
            primal_shift = primal_shift_controller + primal_shift_network
            dual_shift = dual_shift_controller + dual_shift_network
            relative_gap = (np.abs(primal_shift) + np.abs(dual_shift)) / np.abs(primal_objective + primal_shift)
            print(f"Objective gap: {str(round(relative_gap * 100, 4))} percent")
        self._objective_gap_trace.append(relative_gap)
        return relative_gap <= BIG_THETA
    
    def close(self):
        if self._model_controller:
            self._model_controller.close()
        if self._env:
            self._env.close()
        self.proc_pool.close()
    
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
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._model_controller
        PARAMS = self._solver_params

        total_runtime = 0
        epoch = 0
        max_iters = PARAMS.NumberOfEpochs
        try:
            for _ in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
            # while True:
                t_network = 0

                # First, let the controller decide what the utilization is
                optimize_or_scream(MODEL_CONTROLLER)
                # print("Finished controller optimization problem")
                # self._check_objective_gap()

                # Now, do in-network optimization
                for i in reversed(range(PARAMS.NumberOfNetworkUpdates)):
                    t_network += self._do_network_update()
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
                self._set_X_ek()

                # print(f"Total Network Update Gap: {total_gap}")

                # Now that we have non-zero flow assignments, inform the controller
                self._update_Zo_t()
                self._update_rho_coeff()
                self._update_r_t()

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
                if ((max_iters is not None) and (epoch == max_iters)):
                    break
            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        T = self._T

        # TODO: This is not numerically stable ...
        def in_consensus(primal, pair):
            if abs(primal - pair) < te.constants.FLOAT_RES:
                return True
            if feasibility_tol is not None:
                return abs(primal - pair) < feasibility_tol
            return abs((primal - pair) / (primal + te.constants.FLOAT_RES)) < feasibility_ratio

        # Are outer ADMM pairs in consensus?
        YO_T = self._Yo_t
        ZO_T = self._Zo_t
        for t in range(T):
            primal = YO_T[t].X
            pair = ZO_T[t]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            if not in_consensus(primal, pair):
                print(as_fail(f"Axis {t} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
        
        # Are inner ADMM pairs in consensus?
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t
        for t in range(T):
            primal = Y_BAR_T[t]
            pair = P_BAR_T[t]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
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
        self._traffic = tm
        self._commodity_list = traffic_to_commodity(tm)
        self._set_initial_feasible_solution()
    
    @staticmethod
    def do_pgd_with_exact_line_search_for_optimal_lambda(x_k_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, Y_t: np.ndarray, 
                                                         thresh: float, eta: float, epsilon: float, n_iter: int) -> np.ndarray:
        num_edges, _ = np.shape(n)
        lambda_k = np.zeros((num_edges,))
        c = (eta) / (eta + epsilon) * Y_t

        def get_alpha(current_lambda) -> float:
            t1 = careful_norm_squared(n.T @ x_k_0 + c)
            t2 = careful_norm_squared(n.T @ (current_lambda + x_k_0) + c)
            return np.clip(1 - t1 / t2, a_min=0, a_max=None)

        i = 0
        while i < n_iter:
            lambda_k_old = lambda_k
            grad = nnt @ lambda_k + x_k_0 + n @ c
            lambda_k = np.clip(lambda_k_old - get_alpha(lambda_k_old) * grad, a_min=0, a_max=None)
            if careful_norm(lambda_k - lambda_k_old) < thresh:
                break
            i += 1
        return lambda_k

    def get_optimal_lambda(self):
        K = len(self._commodity_list)
        ETA = self.current_eta
        EPSILON = self._solver_params.Epsilon
        GAMMA = self._solver_params.Gamma
        assert GAMMA is None
        PGD_ITERS = self._solver_params.PGDIterations
        PGD_CONV_TOL = self._solver_params.PGDConvTol
        NULL_M = self._NULL_M
        NNT_M = self._NNT_M
        Y_TK = self._Y_tk
        X_EK_START = self._X_ek_start

        def pgd_iterator():
            for k in range(K):
                yield (X_EK_START[:, k], NNT_M, NULL_M, Y_TK[:, k], PGD_CONV_TOL, ETA, EPSILON, PGD_ITERS)

        for k, lambda_k in enumerate(self.proc_pool.starmap(self.do_pgd_with_exact_line_search_for_optimal_lambda, pgd_iterator())):
            self._lambda_ek[:, k] = lambda_k
    
    def initialize_to(self, solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution):
        T = self._T
        NULL_M = self._NULL_M
        ETA = self.current_eta
        EPSILON = self._solver_params.Epsilon

        assignments, u = solution.get_vars()
        print(f"Target Utilization: {u}")

        # Set all ADMM values, assuming the given assignments are optimal
        self._rho_coeff = 1
        self._eta_coeff = 1
        self._X_ek = assignments
        self._Y_tk = NULL_M.T @ (self._X_ek - self._X_ek_start)
        self._Zo_t = np.sum(self._Y_tk, axis=1)
        self._Zo_t_old = np.copy(self._Zo_t)
        self._Xo_e_sol = np.sum(assignments, axis=1)
        self._Y_bar_t = np.average(self._Y_tk, axis=1)
        self._P_bar_t = np.copy(self._Y_bar_t)
        self._Y_bar_t_old = np.copy(self._Y_bar_t)
        self._P_bar_t_old = np.copy(self._Y_bar_t)
        self._C_tk_old = (ETA / (ETA + EPSILON)) * np.copy(self._Y_tk)
        self.get_optimal_lambda()
        self._r_t = np.zeros(shape=(T,))
        self._r_t_old = np.zeros(shape=(T,))
        self._u_t = np.zeros(shape=(T,))

        # Now, optimize the controller model, it should not move ...
        self._target_u = u
        self._update_controller_objective()
        optimize_or_scream(self._model_controller)
        print("Finished warm start optimization problem")
        self._check_objective_gap()

        self._rho_coeff_trace = []
        self._eta_coeff_trace = []
        self._objective_trace = []
        self._objective_gap_trace = []
    
    def set_target(self, solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution):
        _, u = solution.get_vars()
        self._target_u = u

