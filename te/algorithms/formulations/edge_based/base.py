import time
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from te.algorithms.base import TrafficEngineeringLP, TrafficEngineeringLPCheckResult
from te.traffic_models.base import Commodity
from te.algorithms.sub_algorithms.link_capacity_test import check_capacity_constraint
from te.algorithms.sub_algorithms.loop_free_test import check_loop_free_assignment
from te.algorithms.sub_algorithms.flow_conservation_test import check_flow_conservation
from utils.logging import as_info


class EdgeBasedTEBase(TrafficEngineeringLP):
    """
    A trivial base class for edge-based TE that implements minimal checks
    for the output solution.
    """
    def _report_problem_size(self):
        M = len(self.graph.nodes)
        N = len(self.graph.edges)
        K = len(self.commodity_list)

        print(as_info(f"Graph Size: {M} nodes | {N} edges"))
        print(as_info(f"Number of commodities: {K}"))
    
    def make_lp(self):
        t_start = time.time()
        self._make_variables()
        self._add_constraints()
        self._add_objective()
        print(as_info(f"Built model in {str(np.round(time.time() - t_start, 2))} seconds."))

    def check(self):
        """
        For edge-based TE, the default checks include:
        - Check the assignment is loop-free
        - Check how much demand satisfaction we have, optionally check flow-conservation
          explicitly.
        - Check how many congested links we have
        - Check the density (ratio of non-zero entries) in the final solution

        This function returns nothing. The check result is instead stored in the 
        `check_result` property.
        """
        assignments = self.assignments
        graph = self.graph
        commodity_list = self.commodity_list
        eval_params = self.problem_description.EvalParams
        loop_free = check_loop_free_assignment(assignments, graph, commodity_list, eval_params)
        unsat_ratio, unsat_commodities, total_satisfcation = check_flow_conservation(
            assignments, graph, commodity_list, eval_params
        )
        congested_ratio, congested_links = check_capacity_constraint(
            assignments, graph, commodity_list, eval_params
        )
        self.check_result = TrafficEngineeringLPCheckResult(
            unsat_ratio=unsat_ratio,
            congested_ratio=congested_ratio,
            unsat_commodities=unsat_commodities,
            congested_links=congested_links,
            density=np.count_nonzero(np.clip(assignments)) / assignments.size,
            total_satisfcation=total_satisfcation,
            loop_free=loop_free
        )
    
    def get_solution_commodity_list(self) -> List[Tuple[Commodity, Commodity]]:
        """
        A handy function that takes the edge-based assignment and returns the list
        of _routed_ commodities. We need this to check if the amount a node sends
        or receives is correctly balanced.

        Returns
        -------
        ls: List[Tuple[Commodity, Commodity]]
            List of tuples, containing the amount sent and received respectively.
        """
        assert self.assignments is not None

        COMMODITIES = self.commodity_list
        GRAPH = self.graph
        X = self.assignments

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
