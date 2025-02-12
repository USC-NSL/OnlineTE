import time
import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from collections import defaultdict
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
class RegularizedADMMSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 1000
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Epsilon: float = te.constants.DEFAULT_EPSILON_KE
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
        self._out_edge_mapping = get_node_and_out_edge_index_mapping(graph)
        self._edge_out_indexing = get_edge_to_out_index_mapping(graph)
        self._out_degrees = {k: v for k, v in graph.out_degree()}
        self._NULL_M: np.ndarray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None

        self._env: gurobipy.Env = None

        self._model_controller: Optional[gurobipy.Model] = None
        self._model_nodes: Optional[List[gurobipy.Model]] = None
        self._objective_controller: Optional[gurobipy.QuadExpr] = None
        self._objective_nodes: Optional[List[gurobipy.QuadExpr]] = None
        
        # Both just vectors of length `n`
        self._Xo_e_start: Optional[np.ndarray] = None
        self._Xo_e: Optional[gurobipy.tupledict] = None
        self._Zo_e: Optional[np.ndarray] = None
        # Just a variable between 0 and 1
        self._utility: Optional[gurobipy.Var] = None

        # This need not be managed by Gurobi
        self._X_ek_start: Optional[np.ndarray] = None
        self._X_ek: Optional[np.ndarray] = None
        # Global value. A `T x K` matrix that we treat as a list of `K` vectors of length `T`
        self._Y_tk: Optional[List[gurobipy.tupledict]] = None
        # ADMM running average variable. A vector of length `T`.
        self._P_bar_t: Optional[np.ndarray] = None

        # Residual of outer ADMM. A vector of length `n`.
        self._r_e: Optional[np.ndarray] = None
        # Residual of inner ADMM. A vector of length `T`.
        self._u_t: Optional[np.ndarray] = None

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
    def objective_value(self) -> float:
        return self._utility.X
    
    @property
    def objective_trace(self) -> Optional[List[float]]:
        return self._objective_trace

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
        self._T = T
        self._NUM_EDGES = n
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        NUM_EDGES = self._NUM_EDGES

        self._r_e = np.zeros(shape=(NUM_EDGES,))
        self._u_t = np.zeros(shape=(T,))
        self._Zo_e = np.copy(self._Xo_e_start)
        self._P_bar_t = np.zeros((T,))
    
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
        assert self._model_controller is None and \
               self._model_nodes is None
        
        NUM_EDGES = self._NUM_EDGES
        T = self._T
        K = len(self._commodity_list)

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        # Each commodity now gets its own small model.
        self._model_nodes = [
            gurobipy.Model(f'EdgeBasedDistributedTE_Commodity_{k}', env=ENV) for k in range(K)
        ]
    
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes
        
        self._Xo_e = MODEL_CONTROLLER.addVars(NUM_EDGES, lb=0.0, vtype=GRB.CONTINUOUS, name=f'XO_E')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')
        # Set starting values ...
        self._Xo_e.Start = self._Xo_e_start

        # This one is just a list of K vectors, each of length `T`
        self._Y_tk = [
            model.addVars(T, lb=-float('inf'), vtype=GRB.CONTINUOUS, name=f'Y_{k}') for k, model in enumerate(MODEL_NODES)
        ]
    
    def _get_F(self) -> np.ndarray:
        return self._Zo_e + self._r_e - self._Xo_e_start
    
    def _get_X_ek(self, e: int, k: int) -> gurobipy.LinExpr:
        T = self._T
        exp = gurobipy.LinExpr()
        exp.addConstant(self._X_ek_start[e, k])
        for t in range(T):
            exp.addTerms(self._NULL_M[e, t], self._Y_tk[k][t])
        return exp
    
    def _get_X_k_sum(self) -> np.ndarray:
        K = len(self._commodity_list)
        NUM_EDGES = self._NUM_EDGES
        return np.array([np.sum([self._get_X_ek(e, k).getValue() for k in range(K)]) for e in range(NUM_EDGES)])
    
    def _get_Y_k_old(self, k: int) -> np.ndarray:
        T = self._T
        try:
            return np.array([self._Y_tk[k][t].X for t in range(T)])
        except AttributeError:
            return np.zeros((T,))
    
    def _get_Y_bar(self) -> np.ndarray:
        K = len(self._commodity_list)
        return np.average([self._get_Y_k_old(k) for k in range(K)], axis=0)

    def _add_constraints(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        GRAPH = self._graph
        NUM_EDGES = self._NUM_EDGES
        XO_E = self._Xo_e
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes

        # Capacity constraint
        MODEL_CONTROLLER.addConstrs(
            XO_E[i] / c_e <= UTILITY
                for i, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity'))
        )
        
        # Non-negativity constraint for node models
        for k, node_model in enumerate(MODEL_NODES):
            for e in range(NUM_EDGES):
                node_model.addConstr(0 <= self._get_X_ek(e, k))
    
    def _update_controller_objective(self):
        NUM_EDGES = self._NUM_EDGES
        UTILITY = self._utility
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        R_E = self._r_e
        RHO = self._solver_params.Rho
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
    
    def _update_node_objective(self):
        K = len(self._commodity_list)
        T = self._T
        NUM_EDGES = self._NUM_EDGES
        EPSILON = self._solver_params.Epsilon
        ETA = self._solver_params.Eta

        Y_TK = self._Y_tk
        NULL_M = self._NULL_M
        X_EK_0 = self._X_ek_start
        Y_TK_old = [self._get_Y_k_old(k) for k in range(K)]
        Y_BAR_T = self._get_Y_bar()
        P_BAR_T = self._P_bar_t
        U_T = self._u_t
        MODEL_NODES = self._model_nodes

        # print(np.shape(Y_TK_old))
        # print(np.shape(Y_BAR_T))
        # print(np.shape(P_BAR_T))
        # print(np.shape(U_T))

        """
        The node objective for commodity `k` is:

            (\epsilon/2) || X_k^0 + NULL_M @ Y_k ||_2^2 + 
            (\eta/2) || Y_k - Y_k^(old) + Y_bar - P_bar + u ||_2^2
        
        We will benefit greatly from openning this expression and writing it out as
        incremental terms rather than `quicksum`, as it makes it much faster.

        To this end, the first expression (the regularizer) can be expanded to (ignoring constants!):

            (\epsilon/2) (\sum_t (Y_kt^2) + 2 \sum_e X_ke^(0) \sum_t NULL_M_et Y_tk))
        
        (Note that columns of `NULL_M` were orthonormal).
        The section expression is just:

            (\eta/2) (\sum_t (Y_kt^2) + 2*\sum_t (Y_kt)(u_t - P_bar_t + Y_bar_t - Y_k^(old)_t))
        """

        OBJECTIVE_NODES = [gurobipy.QuadExpr() for k in range(K)]
        for k, obj in enumerate(OBJECTIVE_NODES):
            for t in range(T):
                y = Y_TK[k][t]
                c = U_T[t] - P_BAR_T[t] + Y_BAR_T[t] - Y_TK_old[k][t]
                obj.addTerms((EPSILON + ETA)/2, y, y)
                obj.addTerms(ETA * c, y)
                for e in range(NUM_EDGES):
                    x = X_EK_0[e, k]
                    n = NULL_M[e, t]
                    obj.addTerms(ETA * x * n, y)
                    
        assert len(OBJECTIVE_NODES) == len(MODEL_NODES)
        for node_obj, node_model in zip(OBJECTIVE_NODES, MODEL_NODES):
            node_model.setObjective(node_obj, GRB.MINIMIZE)
        self._objectives_nodes = OBJECTIVE_NODES
    
    def _add_objective(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        self._update_controller_objective()
        self._update_node_objective()
    
    def _update_P_bar(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None
        
        """
        The update rule for `P_bar` is:

            P_bar \gets (NULL_M^T F + (\eta/\rho) (u + Y_bar)) / (K + (\eta/\rho))
        """

        K = len(self._commodity_list)
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        U_T = self._u_t
        Y_BAR_T = self._get_Y_bar()
        F_E = self._get_F()
        NULL_M = self._NULL_M

        self._P_bar_t = (NULL_M.T @ F_E + (ETA/RHO) * (U_T + Y_BAR_T)) / (K + (ETA/RHO))        
    
    def _update_u_t(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None
        
        """
        The update rule for `u` is:

            u \gets (u + Y_bar - P_bar)
        """

        U_T = self._u_t
        Y_BAR_T = self._get_Y_bar()
        P_BAR_T = self._P_bar_t

        self._u_t = (U_T + Y_BAR_T - P_BAR_T)

    def _update_Zo_e(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        """
        The update rule for Zo_e is:
            Zo_e \gets (X_oe + \sum_k X_ke)/2
        """

        XO_E = self._Xo_e
        NUM_EDGES = self._NUM_EDGES
        X_KE_SUM_E = self._get_X_k_sum()
        Zo_e = np.zeros((NUM_EDGES,))
        for e in range(NUM_EDGES):
            Zo_e[e] = (XO_E[e].X + X_KE_SUM_E[e]) / 2
        self._Zo_e = Zo_e
    
    def _update_r_e(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

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
        if self._model_nodes:
            for node_model in self._model_nodes:
                node_model.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        t_start = time.time()
        print("Starting to create the model")
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        print(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds.")
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        for node_model in self._model_nodes:
            node_model.reset()
        if with_params:
            self._model_controller.resetParams()
            for node_model in self._model_nodes:
                node_model.resetParams()

    def _build_X_ek(self):
        K = len(self._commodity_list)
        NUM_EDGE = self._NUM_EDGES
        X_EK = np.zeros((NUM_EDGE, K))
        for e in range(NUM_EDGE):
            for k in range(K):
                X_EK[e, k] = self._get_X_ek(e, k).getValue()
        self._X_ek = X_EK
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes
        M = len(MODEL_NODES)
        PARAMS = self._solver_params

        total_runtime = 0

        try:
            for _ in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
                t_commodities: Dict[int, List] = defaultdict(list)

                # First, let the controller decide what the utilization is
                optimize_or_scream(MODEL_CONTROLLER)

                # Now, do in-network optimization
                for _ in range(PARAMS.NumberOfNetworkUpdates):
                    for k, node_model in enumerate(MODEL_NODES):
                        optimize_or_scream(node_model)
                        t_commodities[k].append(node_model.Runtime)
                    self._update_P_bar()
                    self._update_u_t()
                    self._update_node_objective()

                # Now that we have non-zero flow assignments, inform the controller
                self._update_Zo_e()
                self._update_r_e()

                # Update the objectives and start again
                self._update_controller_objective()
                self._update_node_objective()

                # Houskeeping
                self._objective_trace.append(self._utility.X)
                total_runtime += MODEL_CONTROLLER.Runtime + max(sum(t_commodities[v]) for v in range(M))
            
            # Build flow assignments
            self._build_X_ek()

            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        assert (feasibility_tol is None) ^ (feasibility_ratio is None)
        NUM_EDGES = self._NUM_EDGES
        T = self._T
        PARAMS = self._solver_params

        # Are outer ADMM pairs in consensus?
        XO_E = self._Xo_e
        ZO_E = self._Zo_e
        for e in range(NUM_EDGES):
            primal = XO_E[e].X
            pair = ZO_E[e]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            assert abs(primal - pair) < 2*PARAMS.FeasibilityTol, \
                f"Edge {e} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"
        
        # Are inner ADMM pairs in consensus?
        Y_BAR_T = self._get_Y_bar()
        P_BAR_T = self._P_bar_t
        for t in range(T):
            primal = Y_BAR_T[t]
            pair = P_BAR_T[t]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            assert abs(primal - pair) < 2*PARAMS.FeasibilityTol, \
                f"Axis {t} --> Inner ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"
        
        # Now, check flow conservation ...
        X_EK = self._X_ek
        check_centralized_flow_conservation(X_EK, self._graph, self._commodity_list, PARAMS.FeasibilityTol)
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
