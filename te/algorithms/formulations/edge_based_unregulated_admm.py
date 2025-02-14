import time
import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from collections import defaultdict
from multiprocessing import Pool
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from gurobipy import GRB, GurobiError
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import (get_edge_indexing, get_node_and_out_edge_index_mapping, 
                              get_edge_to_out_index_mapping, get_graph_M_matrix, 
                              get_adjacency_null_space, get_feasible_flow_assignment)
from te.algorithms.utils import (check_centralized_flow_conservation,
                                 check_capacity_constraint,
                                 optimize_or_scream)


@dataclass
class UnregulatedADMMSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 1000
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Gamma: float = 1e-1
    PGDIterations: int = 5
    NumWorkers: int = 1
    Seed: int = te.constants.DEFAULT_SEED
    UseVariableRho: bool = True
    Mu: float = te.constants.DEFAULT_MU
    TauIncrease: float = te.constants.DEFAULT_TAU_INC
    TauDecrease: float = te.constants.DEFAULT_TAU_DEC


class UnregulatedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: UnregulatedADMMSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._out_edge_mapping = get_node_and_out_edge_index_mapping(graph)
        self._edge_out_indexing = get_edge_to_out_index_mapping(graph)
        self._out_degrees = {k: v for k, v in graph.out_degree()}
        self._NULL_M: np.ndarray = None
        self._NNT_M: np.ndarray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None
        
        # Both just vectors of length `n`
        self._Xo_e_start: Optional[np.ndarray] = None
        self._Xo_e: Optional[gurobipy.tupledict] = None
        self._Zo_e: Optional[np.ndarray] = None
        self._Zo_e_old: Optional[np.ndarray] = None
        # Just a variable between 0 and 1
        self._utility: Optional[gurobipy.Var] = None

        # This need not be managed by Gurobi
        self._X_ek_start: Optional[np.ndarray] = None
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

        self.proc_pool = Pool(processes=self._solver_params.NumWorkers)

        # Dual vairable of outer ADMM. A vector of length `n`.
        self._r_e: Optional[np.ndarray] = None
        # Dual vairable of inner ADMM. A vector of length `T`.
        self._u_t: Optional[np.ndarray] = None

        self._outer_primal_residual_norm: float = None
        self._outer_dual_residual_norm: float = None
        self._inner_primal_residual_norm: float = None
        self._inner_dual_residual_norm: float = None

        self._rho_coeff: Optional[float] = None
        self._rho_coeff_trace: List[float] = []
        self._eta_coeff: Optional[float] = None
        self._eta_coeff_trace: List[float] = []

        self._objective_trace = []

        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self._report_problem_size()

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
    def assignments(self) -> np.ndarray:
        assert self._X_ek is not None
        return self._X_ek

    def _set_initial_feasible_solution(self):
        self._X_ek_start = get_feasible_flow_assignment(self._graph, self._commodity_list)
        self._Xo_e_start = np.sum(self._X_ek_start, axis=1)
        # Just a sanity check ...
        # check_centralized_flow_conservation(self._X_ek_start, self._graph, self._commodity_list, self._solver_params.FeasibilityTol)
    
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

        # We should scale the ADMM step sizes by the size of these values as well ...
        # self._rho_coeff = 1/(n**2)
        # self._eta_coeff = 1/(T**2)
        self._rho_coeff = 1
        self._eta_coeff = 1
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        K = len(self._commodity_list)
        NUM_EDGES = self._NUM_EDGES

        self._r_e = np.zeros(shape=(NUM_EDGES,))
        self._u_t = np.zeros(shape=(T,))
        self._Zo_e = np.copy(self._Xo_e_start)
        self._P_bar_t = np.zeros((T,))
        self._Y_bar_t = np.zeros((T,))
        self._Y_tk = np.zeros((T, K))
        self._X_ek = np.copy(self._X_ek_start)
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
        
    def _make_variables(self):
        assert self._model_controller is None
        
        NUM_EDGES = self._NUM_EDGES

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        MODEL_CONTROLLER = self._model_controller
        
        self._Xo_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name=f'XO_E')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        # Set starting values ...
        self._Xo_e.Start = self._Xo_e_start
    
    def _get_F(self) -> np.ndarray:
        return self._Zo_e + self._r_e - self._Xo_e_start
    
    def _set_X_ek(self):
        self._X_ek = self._X_ek_start + self._NULL_M @ self._Y_tk
    
    def _get_X_k_sum(self) -> np.ndarray:
        assert self._X_ek is not None
        return np.sum(self._X_ek, axis=1)
    
    def _add_constraints(self):
        assert self._model_controller is not None

        GRAPH = self._graph
        XO_E = self._Xo_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller

        # Capacity constraint
        MODEL_CONTROLLER.addConstrs(
            XO_E[i] / c_e <= UTILITY
                for i, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity'))
        )
    
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
            x = XO_E[e]
            c = R_E[e] - ZO_E[e]
            OBJECTIVE_CONTROLLER.addTerms(RHO/2, x, x)
            OBJECTIVE_CONTROLLER.addTerms(RHO * c, x)
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _add_objective(self):
        assert self._model_controller is not None

        self._update_controller_objective()
    
    @staticmethod
    def do_pgd(lambda_k: np.ndarray, x_k_0: np.ndarray, nnt: np.ndarray, n: np.ndarray, c: np.ndarray, gamma: float, thresh: float, n_iter: int):
        _c = x_k_0 + n @ c
        for i in range(n_iter):
            lambda_k_old = lambda_k
            lambda_k = np.clip(lambda_k - gamma * (nnt @ lambda_k + _c), a_min=0, a_max=None)
            if (np.linalg.norm(lambda_k - lambda_k_old) / np.sqrt(lambda_k.shape[0])) < thresh:
                break
        return (lambda_k, c + n.T @ lambda_k)
    
    def _do_network_update(self) -> float:
        def pgd_iterator():
            K = len(self._commodity_list)
            PGD_ITERS = self._solver_params.PGDIterations
            GAMMA = self._solver_params.Gamma
            NULL_M = self._NULL_M
            NNT_M = self._NNT_M
            X_EK_START = self._X_ek_start
            Y_TK = self._Y_tk
            Y_BAR = self._Y_bar_t
            P_BAR = self._P_bar_t
            U_T = self._u_t
            LAMBDA_EK = self._lambda_ek
            for k in range(K):
                C_K = Y_TK[:, k] - Y_BAR + P_BAR - U_T
                # TODO: Make threshold parameter a solver parameter input ...
                yield (LAMBDA_EK[:, k], X_EK_START[:, k], NNT_M, NULL_M, C_K, GAMMA, 1e-8, PGD_ITERS)
        t_start = time.time()
        for k, item in enumerate(self.proc_pool.starmap(self.do_pgd, pgd_iterator())):
            lambda_k, y_k = item
            self._lambda_ek[:, k] = lambda_k
            self._Y_tk[:, k] = y_k
        return time.time() - t_start
    
    def _update_Y_bar(self):
        self._Y_bar_t_old = np.copy(self._Y_bar_t)
        self._Y_tk_old = np.copy(self._Y_tk)
        self._Y_bar_t = np.average(self._Y_tk, axis=1)
    
    def _update_P_bar(self):
        assert self._model_controller is not None
        
        """
        The update rule for `P_bar` is:

            P_bar \gets (NULL_M^T F + (\eta/\rho) (u + Y_bar)) / (K + (\eta/\rho))
        """

        T = self._T
        K = len(self._commodity_list)
        ETA = self._solver_params.Eta * self._eta_coeff
        RHO = self._solver_params.Rho * self._rho_coeff
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        F_E = self._get_F()
        NULL_M = self._NULL_M
        self._P_bar_t_old = np.copy(self._P_bar_t)
        P_BAR_T = (NULL_M.T @ F_E + (ETA/RHO) * (U_T + Y_BAR_T)) / (K + (ETA/RHO))
        self._P_bar_t = P_BAR_T

        self._inner_primal_residual_norm = np.linalg.norm((P_BAR_T - Y_BAR_T)) / np.sqrt(T)
        self._inner_dual_residual_norm = np.linalg.norm(
            (self._Y_tk - self._Y_tk_old) + 
            (P_BAR_T - self._P_bar_t_old)[:, np.newaxis] +
            (self._Y_bar_t_old - Y_BAR_T)[:, np.newaxis]
        ) * self._eta_coeff / np.sqrt(T * K)
    
    def _update_u_t(self):
        assert self._model_controller is not None
        
        """
        The update rule for `u` is:

            u \gets (u + Y_bar - P_bar)
        """

        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t

        self._u_t = (U_T + Y_BAR_T - P_BAR_T)

    def _update_Zo_e(self):
        assert self._model_controller is not None

        """
        The update rule for Zo_e is:
            Zo_e \gets (X_oe + \sum_k X_ke)/2
        """
        
        NUM_EDGES = self._NUM_EDGES
        XO_E = self._Xo_e
        XO_E_ = np.array([XO_E[e].X for e in range(NUM_EDGES)])
        X_KE_SUM_E = self._get_X_k_sum()
        Zo_e = (XO_E_ + X_KE_SUM_E) / 2
        self._Zo_e_old = np.copy(self._Zo_e)
        self._Zo_e = Zo_e
        scaling = np.sqrt(NUM_EDGES)
        self._outer_primal_residual_norm = np.linalg.norm((Zo_e - XO_E_)) / scaling + np.linalg.norm((Zo_e - X_KE_SUM_E)) / scaling
        self._outer_dual_residual_norm = np.linalg.norm((Zo_e - self._Zo_e_old)) * self._rho_coeff / scaling
    
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
            print(f"OUTER PRIMAL = {str(round(primal_norm, 4))} | OUTER DUAL = {str(round(dual_norm, 4))}")
    
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
            print(f"\tINNER PRIMAL = {str(round(primal_norm, 4))} | INNER DUAL = {str(round(dual_norm, 4))}")
    
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
        self._r_e = r_e
    
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
        K = len(self._commodity_list)

        total_runtime = 0

        try:
            # for _ in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
            for _ in range(PARAMS.NumberOfEpochs):
                t_network = 0

                # First, let the controller decide what the utilization is
                optimize_or_scream(MODEL_CONTROLLER)

                # Now, do in-network optimization
                for _ in range(PARAMS.NumberOfNetworkUpdates):
                    t_network += self._do_network_update()
                    self._update_Y_bar()
                    self._update_P_bar()
                    self._update_eta_coeff()
                    self._update_u_t()
                self._set_X_ek()

                # Now that we have non-zero flow assignments, inform the controller
                self._update_Zo_e()
                self._update_rho_coeff()
                self._update_r_e()

                # Update the objectives and start again
                self._update_controller_objective()

                # Houskeeping
                self._objective_trace.append(self._utility.X)
                total_runtime += MODEL_CONTROLLER.Runtime + t_network

            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        NUM_EDGES = self._NUM_EDGES
        T = self._T
        PARAMS = self._solver_params

        # TODO: This is not numerically stable ...
        def in_consensus(primal, pair):
            if abs(primal - pair) < te.constants.FLOAT_RES:
                return True
            if feasibility_tol is not None:
                return abs(primal - pair) < feasibility_tol
            return abs((primal - pair) / (primal + te.constants.FLOAT_RES)) < feasibility_ratio

        # # Are outer ADMM pairs in consensus?
        # XO_E = self._Xo_e
        # ZO_E = self._Zo_e
        # for e in range(NUM_EDGES):
        #     primal = XO_E[e].X
        #     pair = ZO_E[e]
        #     primal_str = str(np.round(primal, 4))
        #     pair_str = str(np.round(pair, 4))
        #     assert in_consensus(primal, pair), \
        #         f"Edge {e} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"
        
        # # Are inner ADMM pairs in consensus?
        # Y_BAR_T = self._Y_bar_t
        # P_BAR_T = self._P_bar_t
        # for t in range(T):
        #     primal = Y_BAR_T[t]
        #     pair = P_BAR_T[t]
        #     primal_str = str(np.round(primal, 4))
        #     pair_str = str(np.round(pair, 4))
        #     assert in_consensus(primal, pair), \
        #         f"Axis {t} --> Inner ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"
        
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
