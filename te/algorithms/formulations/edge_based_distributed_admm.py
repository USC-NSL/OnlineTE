import time
import tqdm
import gurobipy
import numpy as np
import networkx as nx
import te.constants
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from gurobipy import GRB, GurobiError, quicksum
from te.algorithms.base import TrafficEngineeringLP, GurobiSolverParams, SolverParams
from te.traffic_models.base import TrafficMatrixBase, traffic_to_commodity, Commodity
from topologies.utils import (get_edge_indexing, get_node_and_out_edge_index_mapping, 
                              get_edge_to_out_index_mapping, get_graph_M_matrix, 
                              get_adjacency_null_space, get_feasible_flow_assignment)
from te.algorithms.utils import (check_distributed_flow_conservation,
                                 check_centralized_flow_conservation,
                                 check_capacity_constraint,
                                 optimize_or_scream)


@dataclass
class DistributedADMMSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 1000
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Seed: int = te.constants.DEFAULT_SEED


class DistributedEdgeBasedADMMLP(TrafficEngineeringLP):
    def __init__(self, graph: nx.DiGraph, traffic: TrafficMatrixBase, solver_params: DistributedADMMSolverParams) -> None:
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
        self._NULL_M: List[np.ndarray] = None
        self._T: int = None

        self._env: gurobipy.Env = None

        self._model_controller: gurobipy.Model = None
        self._model_nodes: List[gurobipy.Model] = None
        self._objective_controller: gurobipy.QuadExpr = None
        self._objective_nodes: List[gurobipy.QuadExpr] = None
        
        self._X_oe: List[gurobipy.MVar] = None
        self._Z_oe: List[np.ndarray] = None
        self._utility: gurobipy.Var = None

        # This need not be managed by Gurobi
        self._X_ke_start: List[np.ndarray] = None
        self._X_ke: List[np.ndarray] = None
        # Global value, a `T x K` matrix
        self._Y_kt: np.ndarray = None
        # Individual node values, each a `T x K` matrix
        self._P_vkt: List[gurobipy.MVar] = None
        # This is an auxiliary variable to help write the QP for each node. Each is a T entry vector.
        self._P_SUM_vt: List[gurobipy.MVar] = None

        self._re: List[np.ndarray] = None
        # Assigned per indivual node, each a `T x K` matrix as well
        self._mu: List[np.ndarray] = None

        self._objective_trace = []

        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initiate_dual_weights()
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
        x_ke_centralized = get_feasible_flow_assignment(self._graph, self._commodity_list)
        # Just a sanity check ...
        check_centralized_flow_conservation(x_ke_centralized, self._graph, self._commodity_list, self._solver_params.FeasibilityTol)

        # Now, we need to chop it up
        M = len(self._graph.nodes)
        OUT_DEGREE = self._out_degrees
        OUT_EDGE_MAPPING = self._out_edge_mapping

        self._X_ke_start = [
            np.array([x_ke_centralized[OUT_EDGE_MAPPING[(v, e_out)][0], :] for e_out in range(OUT_DEGREE[v])])
            for v in range(M)
        ]
    
    def _set_NULL_M(self):
        null_m_centralized = get_adjacency_null_space(self._M)
        T = null_m_centralized.shape[1]
        self._T = T
        M = len(self._graph.nodes)
        OUT_DEGREE = self._out_degrees
        OUT_EDGE_MAPPING = self._out_edge_mapping

        self._NULL_M = [
            np.array([null_m_centralized[OUT_EDGE_MAPPING[(v, e_out)][0], :] for e_out in range(OUT_DEGREE[v])])
            for v in range(M)
        ]
    
    def _initiate_dual_weights(self):
        M = len(self._graph.nodes)
        T = self._T
        K = len(self._commodity_list)

        # self._re = [
        #     self._rng.random(size=(self._out_degrees[i],))
        #         for i in range(M)
        # ]
        # self._mu = [
        #     self._rng.random(size=(T, K)) for _ in range(M)
        # ]
        self._re = [
            np.zeros(shape=(self._out_degrees[i],))
                for i in range(M)
        ]
        self._mu = [
            np.zeros(shape=(T, K)) 
                for _ in range(M)
        ]
    
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
              f"\t TOTAL NUMBER OF VARIABLES FOR EACH NODE: {T * K}\n"
              f"\t EXPECTED NUMBER OF TRUE CONSTRAINTS FOR EACH NODE: {int(N * K / M)}\n"
              f"\t NUMBER OF AUXILIARY CONSTRAINTS FOR EACH NODE: {T}\n")
        
    def _make_variables(self):
        """
        This is again very similar to primal/dual ascent, but the additional
        variable here would be `Z_oe`.
        """

        assert self._model_controller is None and \
               self._model_nodes is None
        
        M = len(self._graph.nodes)
        T = self._T
        K = len(self._commodity_list)
        OUT_DEGREES = self._out_degrees

        ENV = gurobipy.Env()
        ENV.setParam('OutputFlag', 0)
        ENV.start()
        self._env = ENV

        self._model_controller = gurobipy.Model('EdgeBasedDistributedTE_Controller', env=ENV)
        self._model_nodes = [
            gurobipy.Model(f'EdgeBasedDistributedTE_Node_{i}', env=ENV) for i in range(M)
        ]
    
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes
        
        self._X_oe = [
            MODEL_CONTROLLER.addVars(OUT_DEGREES[v], lb=0.0, vtype=GRB.CONTINUOUS, name=f'X_OE_{v}') \
                for v in range(M)
        ]
        self._utility = MODEL_CONTROLLER.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='U')

        # This need not be managed by Gurobi, since each step has a closed form solution
        self._Z_oe = [np.zeros((OUT_DEGREES[v],)) for v in range(M)]

        # This is just a simple 2D array
        self._Y_kt = np.zeros(shape=(T, K))
        # These are independent Gurobi variables
        self._P_vkt = [
            node_model.addVars(T, K, vtype=GRB.CONTINUOUS, name=f'P_{v}')
                for v, node_model in enumerate(MODEL_NODES)
        ]
        self._P_SUM_vt = [
            node_model.addVars(T, vtype=GRB.CONTINUOUS, name=f'P_SUM_{v}')
                for v, node_model in enumerate(MODEL_NODES)
        ]
    
    def _get_X_ke(self, v, k, e_out) -> gurobipy.LinExpr:
        T = self._T
        X_KE_START = self._X_ke_start[v][e_out, k]
        N_v = self._NULL_M[v]
        P_v = self._P_vkt[v]
        return X_KE_START + quicksum([
            N_v[e_out, t] * P_v[t, k] for t in range(T)
        ])

    def _get_X_ke_sums(self, v, e_out) -> gurobipy.LinExpr:
        K = len(self._commodity_list)
        X_KE_GETTER = self._get_X_ke

        return quicksum([X_KE_GETTER(v, k, e_out) for k in range(K)])

    def _add_constraints(self):
        """
        This is exactly the same as before.
        """

        assert self._model_controller is not None and \
               self._model_nodes is not None

        M = len(self._graph.nodes)
        K = len(self._commodity_list)
        T = self._T
        OUT_DEGREE = self._out_degrees
        GRAPH = self._graph
        X_OE = self._X_oe
        P_VKT = self._P_vkt
        P_SUM_VT = self._P_SUM_vt
        X_KE_GETTER = self._get_X_ke
        UTILITY = self._utility
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes

        # Capacity constraint
        for v in range(M):
            MODEL_CONTROLLER.addConstrs(
                X_OE[v][i] / c_e <= UTILITY
                    for i, (_, _, c_e) in enumerate(GRAPH.out_edges(v, data='capacity'))
            )
        
        # Non-negativity constraint for node models
        for v, node_model in enumerate(MODEL_NODES):
            for k in range(K):
                for e_out in range(OUT_DEGREE[v]):
                    node_model.addConstr(0 <= X_KE_GETTER(v, k, e_out))
        
        # Auxiliary constraints
        for v, node_model in enumerate(MODEL_NODES):
            for t in range(T):
                node_model.addConstr(P_SUM_VT[v][t] == P_VKT[v].sum(t, '*'))
    
    def _update_controller_objective(self):
        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
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
                quicksum([
                    (X_OE[v][i] - Z_OE[v][i] + RE[v][i]) ** 2
                    for i in range(OUT_DEGREES[v])
                ]) for v in range(M)
            ])
        
        MODEL_CONTROLLER.setObjective(OBJECTIVE_CONTROLLER, GRB.MINIMIZE)
        self._objective_controller = OBJECTIVE_CONTROLLER
    
    def _get_F_e(self, v, e_out) -> float:
        X_OE_START = np.sum(self._X_ke_start[v][e_out, :])
        Z_OE_V = self._Z_oe[v]
        R_E_V = self._re[v]

        return Z_OE_V[e_out] + R_E_V[e_out] - X_OE_START
    
    def _update_node_objective(self):
        M = len(self._graph.nodes)
        K = len(self._commodity_list)
        T = self._T
        OUT_DEGREES = self._out_degrees
        RHO = self._solver_params.Rho
        ETA = self._solver_params.Eta
        MU_V = self._mu
        Y_KT = self._Y_kt
        N_VET = self._NULL_M
        P_VKT = self._P_vkt
        P_SUM_VT = self._P_SUM_vt
        F_E_GETTER = self._get_F_e
        MODEL_NODES = self._model_nodes

        """
        The node objective is:

            \sum_{e \in E_v^{out}} (
                rho/2 * (\sum_k \sum_t N_et P^(v)_tk - F_e)^2 +
            ) + 
            \sum_k (
                eta/2 * \sum_t (
                    (P^(v)_tk - Y_tk + MU^(v)_tk) ** 2
                )
            )
        
        Where `F_e := Z_oe + r_e - \sum_k X^(0)_ke`.

        If we write this literarily, then Gurobi _WILL_ explode just openning
        that sum as is. We need to do this beforehand. Luckily, this is
        not a terrible prospect.
        We introduce the auxiliary variable:

            P_SUM^(v)_t := \sum_k P^(v)_tk
        
        Which allows us to write the first part of the objective as:

            \sum_{e \in E_v^{out}} (
                rho/2 * (\sum_t N_et P_SUM^(v)_t - F_e)^2
            )

        Which can be easily expanded and written as a normal QuadExpr.
        As for the second part, the same still applies, and we can expand
        it manually to build the objective.
        """

        OBJECTIVE_NODES = [gurobipy.QuadExpr() for _ in range(M)]
        for v in range(M):
            obj = OBJECTIVE_NODES[v]

            # Part 1
            for e_out in range(OUT_DEGREES[v]):
                f_e = F_E_GETTER(v, e_out)
                n_e = N_VET[v][e_out]
                p_sum_v = P_SUM_VT[v]

                # Square expressions
                for t in range(T):
                    obj.addTerms(RHO/2 * n_e[t]**2, p_sum_v[t], p_sum_v[t])
                # Corss terms
                for t in range(T):
                    for t_prime in range(t+1, T):
                        obj.addTerms(RHO * n_e[t] * n_e[t_prime], p_sum_v[t], p_sum_v[t_prime])
                # Linear terms
                for t in range(T):
                    obj.addTerms(-RHO * f_e * n_e[t], p_sum_v[t])
                # Constant terms
                obj.addConstant(RHO/2 * f_e ** 2)
            
            # Part 2
            for t in range(T):
                for k in range(K):
                    d_tk = Y_KT[t, k] - MU_V[v][t, k]
                    obj.addTerms(ETA/2, P_VKT[v][t, k], P_VKT[v][t, k])
                    obj.addTerms(-ETA * d_tk, P_VKT[v][t, k])
                    obj.addConstant(ETA/2 * d_tk ** 2)

        assert len(OBJECTIVE_NODES) == len(MODEL_NODES)
        for node_obj, node_model in zip(OBJECTIVE_NODES, MODEL_NODES):
            node_model.setObjective(node_obj, GRB.MINIMIZE)
        self._objectives_nodes = OBJECTIVE_NODES
    
    def _add_objective(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        self._update_controller_objective()
        self._update_node_objective()
    
    def _update_Y_kt(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        T = self._T
        K = len(self._commodity_list)
        M = len(self._graph.nodes())
        P_VKT = self._P_vkt
        MU_VKT = self._mu

        matrix_P_vkt = [np.zeros(shape=(T, K)) for v in range(M)]
        for v in range(M):
            for t in range(T):
                for k in range(K):
                    matrix_P_vkt[v][t, k] = P_VKT[v][t, k].X
        
        self._Y_kt = np.average([
            matrix_P_vkt[v] + MU_VKT[v] for v in range(M)
        ], axis=0)
    
    def _update_re(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        """
        The update rule for r_e is:
            r_e \gets r_e + (X_oe - \sum_k X_ke)/2
        """

        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        X_OE = self._X_oe
        X_KE_SUM_GETTER = self._get_X_ke_sums
        RE = self._re

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                RE[v][i] += (
                    X_OE[v][i].X - X_KE_SUM_GETTER(v, i).getValue()
                ) / 2
    
    def _update_mu(self):
        T = self._T
        K = len(self._commodity_list)
        M = len(self._graph.nodes())
        P_VKT = self._P_vkt
        Y_KT = self._Y_kt

        matrix_P_vkt = [np.zeros(shape=(T, K)) for v in range(M)]
        for v in range(M):
            for t in range(T):
                for k in range(K):
                    matrix_P_vkt[v][t, k] = P_VKT[v][t, k].X

        for v in range(M):
            self._mu[v] += (matrix_P_vkt[v] - Y_KT)
    
    def _update_Z_oe(self):
        assert self._model_controller is not None and \
               self._model_nodes is not None

        """
        The update rule for lambda_e is:
            Z_oe \gets (X_oe + \sum_k X_ke)/2
        """

        M = len(self._graph.nodes)
        OUT_DEGREES = self._out_degrees
        X_KE_SUM_GETTER = self._get_X_ke_sums
        X_OE = self._X_oe
        Z_OE = self._Z_oe

        for v in range(M):
            for i in range(OUT_DEGREES[v]):
                Z_OE[v][i] = (
                    X_OE[v][i].X + X_KE_SUM_GETTER(v, i).getValue()
                ) / 2
    
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

    def _build_X_ke(self):
        K = len(self._commodity_list)
        M = len(self._graph.nodes)
        OUT_DEGREE = self._out_degrees
        X_KE = [
            np.zeros(shape=(OUT_DEGREE[v], K)) for v in range(M)
        ]
        X_KE_GETTER = self._get_X_ke
        
        for v in range(M):
            for e_out in range(OUT_DEGREE[v]):
                for k in range(K):
                    X_KE[v][e_out, k] = X_KE_GETTER(v, k, e_out).getValue()
        
        self._X_ke = X_KE
    
    def solve(self, params: SolverParams = None) -> float:
        assert params is None
        
        MODEL_CONTROLLER = self._model_controller
        MODEL_NODES = self._model_nodes
        M = len(MODEL_NODES)
        PARAMS = self._solver_params

        total_runtime = 0

        try:
            for _ in tqdm.tqdm(range(PARAMS.NumberOfEpochs)):
                t_nodes: Dict[int, List] = defaultdict(list)

                # First, let the controller decide what the utilization is
                optimize_or_scream(MODEL_CONTROLLER)

                # Now, do in-network optimization
                for _ in range(PARAMS.NumberOfNetworkUpdates):
                    for v, node_model in enumerate(MODEL_NODES):
                        optimize_or_scream(node_model)
                        t_nodes[v].append(node_model.Runtime)
                    self._update_Y_kt()
                    self._update_mu()
                    self._update_node_objective()

                # Now that we have non-zero flow assignments, inform the controller
                self._update_Z_oe()
                self._update_re()

                # Update the objectives and start again
                self._update_controller_objective()
                self._update_node_objective()

                # Houskeeping
                self._objective_trace.append(self._utility.X)
                total_runtime += MODEL_CONTROLLER.Runtime + max(sum(t_nodes[v]) for v in range(M))
            
            # Build flow assignments
            self._build_X_ke()

            return total_runtime
        except GurobiError as e:
            print(f'Error code {e.errno}: {e}')
            return -1
    
    def check(self, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None):
        assert (feasibility_tol is None) ^ (feasibility_ratio is None)
        X_OE = self._X_oe
        Z_OE = self._Z_oe
        X_KE = self._X_ke
        OUT_DEGREE = self._out_degrees
        PARAMS = self._solver_params

        for v in range(len(X_OE)):
            for i in range(OUT_DEGREE[v]):
                primal = X_OE[v][i].X
                pair = Z_OE[v][i]
                primal_str = str(np.round(primal, 4))
                pair_str = str(np.round(pair, 4))
                assert abs(primal - pair) < 2*PARAMS.FeasibilityTol, \
                    f"Node {v}: Edge {i} --> ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"
        
        check_distributed_flow_conservation(
            X_KE, self._graph, self._edge_out_indexing, self._commodity_list,
            PARAMS.FeasibilityTol
        )
        check_capacity_constraint(
            X_KE, self._graph, self._commodity_list,
            feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio
        )

    def get_solution_commodity_list(self) -> List[Commodity]:
        COMMODITIES = self._commodity_list
        X_KE = self._X_ke
        OUT_INDEX_MAPPING = self._edge_out_indexing
        GRAPH = self._graph

        out_and_in: Dict[Tuple[int, int], Tuple[int, int]] = dict()

        for k, commodity in enumerate(COMMODITIES):
            SOURCE = commodity.source
            DESTINATION = commodity.destination

            flow_out = []
            flow_in = []
            for s, d in GRAPH.edges(data=False):
                i = OUT_INDEX_MAPPING[(s, d)]
                if s == SOURCE:
                    flow_out.append(X_KE[s][i, k])
                if d == DESTINATION:
                    flow_in.append(X_KE[s][i, k])
            
            out_and_in[(SOURCE, DESTINATION)] = (sum(flow_out), sum(flow_in))

        return [
            (
                Commodity(
                    source=source_dest[0],
                    destination=source_dest[1],
                    demand=fout_fin[0]
                ),
                Commodity(
                    source=source_dest[0],
                    destination=source_dest[1],
                    demand=fout_fin[1]
                )
            )
            for source_dest, fout_fin in out_and_in.items()
        ]
