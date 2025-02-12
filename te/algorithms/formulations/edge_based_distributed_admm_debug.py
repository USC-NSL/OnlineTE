import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from gurobipy import GRB, GurobiError, quicksum
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import (get_edge_indexing, get_node_and_out_edge_index_mapping, 
                              get_in_edge_mapping, get_edge_to_out_index_mapping,
                              get_graph_M_matrix, get_adjacency_null_space,
                              get_feasible_flow_assignment)
from te.algorithms.utils import (check_centralized_flow_conservation,
                                 check_capacity_constraint,
                                 optimize_or_scream)


@dataclass
class DistributedADMMDebugSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 1000
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Seed: int = te.constants.DEFAULT_SEED


class SemiDistributedEdgeBasedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: DistributedADMMDebugSolverParams) -> None:
        super().__init__()
        self._graph = graph
        self._M = get_graph_M_matrix(graph)
        self._traffic = traffic
        self._solver_params = solver_params
        self._rng = np.random.default_rng(seed=solver_params.Seed)
        self._commodity_list = traffic_to_commodity(self._traffic)

        self._edge_indexing = get_edge_indexing(graph)
        self._out_edge_mapping = get_node_and_out_edge_index_mapping(graph)
        self._in_edge_mapping: Dict[int, List[Tuple[int, int]]] = get_in_edge_mapping(graph)
        self._edge_out_indexing = get_edge_to_out_index_mapping(graph)
        self._out_degrees = {k: v for k, v in graph.out_degree()}
        self._NULL_M: np.ndarray = None
        self._T: int = None

        self._env: gurobipy.Env = None

        self._model_controller: gurobipy.Model = None
        self._model_network: gurobipy.Model = None
        self._objective_controller: gurobipy.QuadExpr = None
        self._objective_network: gurobipy.QuadExpr = None
        
        self._X_oe: gurobipy.tupledict = None
        self._Z_oe: np.ndarray = None
        self._utility: gurobipy.Var = None

        # This need not be managed by Gurobi
        self._X_ke_start: np.ndarray = None
        self._X_ke: np.ndarray = None
        # A `T x K` matrix
        self._Y_kt: gurobipy.tupledict = None
        # This is an auxiliary variable to help write the QP for each node. It is a T entry vector.
        self._Y_SUM_t: gurobipy.tupledict = None
        # A vector with `N` entries
        self._re: np.ndarray = None

        self._objective_trace = []

        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initiate_dual_weights()

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
        x_ke_centralized = get_feasible_flow_assignment(self._graph, self._commodity_list)
        # Just a sanity check ...
        check_centralized_flow_conservation(x_ke_centralized, self._graph, self._commodity_list, self._solver_params.FeasibilityTol)
        self._X_ke_start = x_ke_centralized

    def _set_NULL_M(self):
        null_m_centralized = get_adjacency_null_space(self._M)
        self._T = null_m_centralized.shape[1]
        self._NULL_M = null_m_centralized
    
    def _initiate_dual_weights(self):
        N = len(self._graph.edges)

        self._re = np.zeros(shape=(N,))

    def _make_variables(self):
        """
        This is again very similar to primal/dual ascent, but the additional
        variable here would be `Z_oe`.
        """

        assert self._model_controller is None and \
               self._model_network is None
        
        N = len(self._graph.edges)
        K = len(self._commodity_list)
        T = self._T

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        self._model_network = gurobipy.Model('EdgeBasedDistributedTE_Network', env=ENV)
    
        MODEL_CONTROLLER = self._model_controller
        MODEL_NETWORK = self._model_network
        
        self._X_oe = MODEL_CONTROLLER.addVars(N, lb=0.0, vtype=GRB.CONTINUOUS, name='X_OE')
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

        # This need not be managed by Gurobi, since each step has a closed form solution
        self._Z_oe = np.zeros((N,))

        self._Y_kt = MODEL_NETWORK.addVars(T, K, vtype=GRB.CONTINUOUS, name='Y_TK')
        self._Y_SUM_t = MODEL_NETWORK.addVars(T, vtype=GRB.CONTINUOUS, name='Y_SUM_T')

    def _get_X_ke(self, k: int, e: int) -> gurobipy.LinExpr:
        T = self._T
        return self._X_ke_start[e, k] + quicksum([
            self._NULL_M[e, t] * self._Y_kt[t, k] for t in range(T)
        ])

    def _get_X_ke_sums(self, e: int) -> gurobipy.LinExpr:
        K = len(self._commodity_list)
        X_KE_GETTER = self._get_X_ke

        return quicksum([X_KE_GETTER(k, e) for k in range(K)])

    def _add_constraints(self):
        """
        This is exactly the same as before.
        """

        assert self._model_controller is not None and \
               self._model_network is not None

        T = self._T
        N = len(self._graph.edges)
        K = len(self._commodity_list)
        GRAPH = self._graph
        X_OE = self._X_oe
        Y_TK = self._Y_kt
        Y_SUM_T = self._Y_SUM_t
        X_KE_GETTER = self._get_X_ke
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller
        MODEL_NETWORK = self._model_network

        # Capacity constraint
        MODEL_CONTROLLER.addConstrs(
            X_OE[e] / c_e <= UTILITY
                for e, (_, _, c_e) in enumerate(GRAPH.edges(data='capacity'))
        )
        
        # Non-negativity constraint for network model
        for k in range(K):
            for e in range(N):
                MODEL_NETWORK.addConstr(0 <= X_KE_GETTER(k, e))
        
        # Auxiliary constraints
        for t in range(T):
            MODEL_NETWORK.addConstr(Y_SUM_T[t] == Y_TK.sum(t, '*'))
    
    def _update_controller_objective(self):
        N = len(self._graph.edges)
        UTILITY = self._utility
        X_OE = self._X_oe
        Z_OE = self._Z_oe
        RE = self._re
        RHO = self._solver_params.Rho
        MODEL_CONTROLLER = self._model_controller

        """
        Controller objective is:
            u + rho/2 sum_e (X_oe - Z_oe + r_e)^2
        """

        OBJECTIVE_CONTROLLER = \
            UTILITY + \
            RHO/2 * quicksum([
                (X_OE[e] - Z_OE[e] + RE[e]) ** 2
                for e in range(N)
            ])
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER

    def _get_F_e(self, e: int) -> float:
        X_OE_START = np.sum(self._X_ke_start[e, :])

        return self._Z_oe[e] + self._re[e] - X_OE_START
    
    def _update_network_objective(self):
        N = len(self._graph.edges)
        T = self._T
        RHO = self._solver_params.Rho
        N_ET = self._NULL_M
        Y_SUM_T = self._Y_SUM_t
        F_E_GETTER = self._get_F_e
        MODEL_NETWORK = self._model_network

        """
        The network objective is:

            \sum_e (
                rho/2 * (\sum_k \sum_t N_et Y_tk - F_e)^2 +
            )
        
        Where `F_e := Z_oe + r_e - \sum_k X^(0)_ke`.

        With the auxiliary variable `Y_SUM`:

            Y_SUM_t := \sum_k Y_tk
        
        Which allows us to write the first part of the objective as:

            \sum_e (
                rho/2 * (\sum_t N_et Y_SUM_t - F_e)^2 +
            )

        Which can be easily expanded and written as a normal QuadExpr.
        """

        OBJECTIVE_NETWORK = gurobipy.QuadExpr()

        # Part 1
        for e in range(N):
            f_e = F_E_GETTER(e)
            n_e = N_ET[e]

            # Square expressions
            for t in range(T):
                OBJECTIVE_NETWORK.addTerms(RHO/2 * n_e[t]**2, Y_SUM_T[t], Y_SUM_T[t])
            # Corss terms
            for t in range(T):
                for t_prime in range(t+1, T):
                    OBJECTIVE_NETWORK.addTerms(RHO * n_e[t] * n_e[t_prime], Y_SUM_T[t], Y_SUM_T[t_prime])
            # Linear terms
            for t in range(T):
                OBJECTIVE_NETWORK.addTerms(-RHO * f_e * n_e[t], Y_SUM_T[t])
            # Constant terms
            OBJECTIVE_NETWORK.addConstant(RHO/2 * f_e ** 2)

        MODEL_NETWORK.setObjective(OBJECTIVE_NETWORK, GRB.MINIMIZE)
        self._objectives_nodes = OBJECTIVE_NETWORK
    
    def _add_objective(self):
        assert self._model_controller is not None and \
               self._model_network is not None

        self._update_controller_objective()
        self._update_network_objective()
    
    def _update_re(self):
        assert self._model_controller is not None and \
               self._model_network is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        N = len(self._graph.edges)
        X_OE = self._X_oe
        X_KE_SUM_GETTER = self._get_X_ke_sums
        RE = self._re

        for e in range(N):
            RE[e] += (
                X_OE[e].X - X_KE_SUM_GETTER(e).getValue()
            ) / 2
    
    def _update_Z_oe(self):
        assert self._model_controller is not None and \
               self._model_network is not None

        """
        The update rule for lambda_e is:
            Z_oe \gets (X_oe + \sum_k X_ke)/2
        """

        N = len(self._graph.edges)
        X_OE = self._X_oe
        X_KE_SUM_GETTER = self._get_X_ke_sums
        Z_OE = self._Z_oe

        for e in range(N):
            Z_OE[e] = (
                X_OE[e].X + X_KE_SUM_GETTER(e).getValue()
            ) / 2
    
    def close(self):
        self._model_controller.close()
        self._model_network.close()
        if self._env:
            self._env.close()
    
    def make_lp(self):
        self._make_variables()
        self._add_constraints()
        self._add_objective()
    
    def reset(self, with_params: False):
        self._model_controller.reset()
        self._model_network.reset()
        if with_params:
            self._model_controller.resetParams()
            self._model_network.resetParams()

    def _build_X_ke(self):
        K = len(self._commodity_list)
        N = len(self._graph.edges)
        X_KE_GETTER = self._get_X_ke
        X_KE = np.zeros(shape=(N, K))
        
        for e in range(N):
            for k in range(K):
                X_KE[e, k] = X_KE_GETTER(k, e).getValue()
        
        self._X_ke = X_KE
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        CONTROLLER_MODEL = self._model_controller
        NETWORK_MODEL = self._model_network

        total_runtime = 0
        try:
            for _ in tqdm.tqdm(range(self._solver_params.NumberOfEpochs)):
                # First, concurrently solve both controller and network models
                optimize_or_scream(CONTROLLER_MODEL)
                optimize_or_scream(NETWORK_MODEL)

                # Second, update Z_oe and r_e
                self._update_Z_oe()
                self._update_re()

                # Update the objective and start again
                self._update_controller_objective()
                self._update_network_objective()

                # Houskeeping
                self._objective_trace.append(self._utility.X)
                total_runtime += max(CONTROLLER_MODEL.Runtime, NETWORK_MODEL.Runtime)
            # Build flow assignments
            self._build_X_ke()
            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1

    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        assert (feasibility_tol is None) ^ (feasibility_ratio is None)
        N = len(self._graph.edges)
        X_OE = self._X_oe
        Z_OE = self._Z_oe
        X_KE = self._X_ke
        PARAMS = self._solver_params

        for e in range(N):
            primal = X_OE[e].X
            pair = Z_OE[e]
            primal_str = str(np.round(primal, 4))
            pair_str = str(np.round(pair, 4))
            assert abs(primal - pair) < 2*PARAMS.FeasibilityTol, \
                f"Edge {e} --> ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"
        
        check_centralized_flow_conservation(
            X_KE, self._graph, self._commodity_list, PARAMS.FeasibilityTol
        )
        check_capacity_constraint(
            X_KE, self._graph, self._commodity_list,
            feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio
        )

    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        COMMODITIES = self._commodity_list
        X_KE = self._X_ke
        GRAPH = self._graph
        INDICES = self._edge_indexing

        return [
            (
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        X_KE[INDICES[(v, commodity.destination)], i] \
                            for v in GRAPH.predecessors(commodity.destination)
                    ])
                ),
                Commodity(
                    source=commodity.source,
                    destination=commodity.destination,
                    demand=sum([
                        X_KE[INDICES[(commodity.source, v)], i] \
                            for v in GRAPH.successors(commodity.source)
                    ])
                )
            )
            for i, commodity in enumerate(COMMODITIES)
        ]