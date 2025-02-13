import gurobipy
import numpy as np
import seaborn as sns
import networkx as nx
import te.constants
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Dict, Union, Optional
from collections import defaultdict
from te.traffic_models.base import Commodity
from te.algorithms.base import TrafficEngineeringLP, SolverParams, GurobiSolverParams


class ANSIColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def make_model(name: str, params: SolverParams):
    assert issubclass(params.__class__, GurobiSolverParams)
    model = gurobipy.Model(name=name)
    model.Params.Method = params.Method
    model.Params.NumericFocus = params.NumericFocus
    model.Params.BarConvTol = params.BarConvTol
    model.Params.FeasibilityTol = params.FeasibilityTol
    model.Params.LogFile = params.LogFile

    return model


def optimize_or_scream(model: gurobipy.Model):
    """Solve a Gurobi model. Throw an error if the model ends up in any non-optimal state"""
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        raise RuntimeError(f"Optimizing model {model.ModelName} returned non-optimal status: {model.Status}")


def get_solution_confusion_matrix(lp: TrafficEngineeringLP, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None, 
                                  report: bool = False, show: bool = True, save_fig: bool = True) -> Tuple[float, np.ndarray]:
    """
    Check how many of the demands are not satisfied and report the solution.
    To check feasibility:
        - We can either check if the solution is within a particular _distance_ of the optimal
        - Or we can check if it has lower than a particular _error ratio_
    """
    assert (feasibility_ratio is None) ^ (feasibility_tol is None), "Exactly one of `feasibility_tol` or `feasibility_ratio` must be given"

    def is_satisfied(optim, actual):
        if feasibility_tol is not None:
            return abs(optim - actual) < feasibility_tol
        return abs((optim - actual) / optim) < feasibility_ratio

    def make_fig(_cm: np.ndarray, _lp: TrafficEngineeringLP) -> Figure:
        _objective_trace = _lp.objective_trace
        _solver_params = _lp.params
        rho_coeff_trace = None
        if hasattr(_solver_params, 'UseVariableRho'):
            if _solver_params.UseVariableRho:
                print(f"{ANSIColors.BOLD}{ANSIColors.OKBLUE}ADMM algorithm used variable step sizes. Will plot that too{ANSIColors.ENDC}")
                rho_coeff_trace = _lp.rho_coeff_trace
        if _objective_trace is None:
            print(f"{ANSIColors.BOLD}{ANSIColors.WARNING}WARNING: No trace of objective value is available{ANSIColors.ENDC}")
            sns.heatmap(cm)
        else:
            if rho_coeff_trace is None:
                _fig = plt.figure(figsize=(7, 3))
                plt.subplot(1, 2, 1)
                plt.plot(_objective_trace)
                plt.subplot(1, 2, 2)
                sns.heatmap(_cm, vmin=0, vmax=1)
            else:
                _fig = plt.figure(figsize=(10, 3))
                plt.subplot(1, 3, 1)
                plt.plot(_objective_trace)
                plt.subplot(1, 3, 2)
                sns.heatmap(_cm, vmin=0, vmax=1)
                ax = plt.subplot(1, 3, 3)
                plt.plot(rho_coeff_trace)
                ax.set_yscale('log')
            return _fig


    commodities = lp.commodity_list
    topology_size = len(lp.graph.nodes)
    solution = lp.get_solution_commodity_list()

    K = len(commodities)
    unsats = 0
    cm = np.zeros(shape=(topology_size, topology_size))

    if report:
        print(" "*12 + "{:^10}    {:^20}".format("DESIRED", "ALLOCATED"))
        print("-"*46)

    for actual, ideal in zip(solution, commodities):
        assert actual[0].source == ideal.source
        assert actual[0].destination == ideal.destination
        assert actual[1].source == ideal.source
        assert actual[1].destination == ideal.destination
        if report:
            print(
                "{:<4} -> {:<4}".format(ideal.source, ideal.destination) +
                "{:^10}    {:^7} <--> {:^7}".format(
                    str(np.round(ideal.demand, 2)),
                    str(np.round(actual[0].demand, 2)),
                    str(np.round(actual[1].demand, 2))
                )
            )
        if not is_satisfied(ideal.demand, actual[0].demand):
            cm[ideal.source, ideal.destination] = 1
            unsats += 1
    
    unsatisfied = unsats / K

    if unsats == 0:
        print(f"{ANSIColors.BOLD}{ANSIColors.OKGREEN}ALL DEMANDS WERE SATISFIED{ANSIColors.ENDC}")
    else:
        print(f"{ANSIColors.BOLD}{ANSIColors.FAIL}" + "{:.1f}% OF DEMANDS WERE NOT SATISFIED".format(unsatisfied*100) + f"{ANSIColors.ENDC}")
    
    if show or save_fig:
        fig = make_fig(cm, lp)
        if fig is not None:
            if show:
                plt.show()
            if save_fig:
                fig.savefig('res.png')

    return (unsatisfied, cm)


def check_centralized_flow_conservation(
        flows: Union[gurobipy.tupledict, np.ndarray], graph: nx.DiGraph, 
        commodities: List[Commodity], feasibility_tol: float
    ):
    """
    Check if solution satisfies all of the following constraints:
        - Transit nodes conserve flows                                    ( flow conservation )
        - A demand destined to a node, never flows out from that node     (  no demand leaks  )
        - A demand sourced from a node, never flows back into that node   (      no loops     )
    This constraint is enforced pretty rigidly, we don't check for 
    relative error, we just check the distance.
    """

    IS_GUROBI_VAR = isinstance(flows, gurobipy.tupledict)

    for k, commodity in enumerate(commodities):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand

        flow_out = defaultdict(list)
        flow_in = defaultdict(list)
        for e, edge in enumerate(graph.edges()):
            flow_out[edge[0]].append(flows[e, k].X if IS_GUROBI_VAR else flows[e, k])
            flow_in[edge[1]].append(flows[e, k].X if IS_GUROBI_VAR else flows[e, k])

        for v in graph.nodes():
            fout = sum(flow_out[v])
            fin  = sum(flow_in[v])

            demand_str = str(np.round(DEMAND, 4))
            fout_str = str(np.round(fout, 4))
            fin_str = str(np.round(fin, 4))

            if v == SOURCE:
                assert abs(fout - DEMAND) < feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand outflow does not hold at source: {fout_str} vs {demand_str}"
                assert abs(fin) < feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Source receives its own demand! {fin_str}"
            elif v == DESTINATION:
                assert abs(fout) < feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Destination is leaking demand! {fout_str}"
                assert abs(fin - DEMAND) < feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand inflow does not hold at destination: {fin_str} vs {demand_str}"
            else:
                assert abs(fout - fin) < feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Transit demand conservation does not hold: {fin_str} --> {fout_str}"


def check_capacity_constraint(
        flows: Union[gurobipy.tupledict, np.ndarray], graph: nx.DiGraph, commodities: List[Commodity], 
        feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None, report: bool = True
    ) -> float:
    """Check if solution honors link capacity constraints"""
    assert (feasibility_ratio is None) ^ (feasibility_tol is None), "Exactly one of `feasibility_tol` or `feasibility_ratio` must be given"

    def is_congested(capacity, demand):
        if feasibility_tol is not None:
            return demand - capacity > feasibility_tol
        return ((demand - capacity) / capacity) > feasibility_ratio

    K = len(commodities)
    N = len(graph.edges)

    if isinstance(flows, gurobipy.tupledict):
        X_EK = np.zeros((N, K))
        for e in range(N):
            for k in range(K):
                X_EK[e, k] = flows[e, k].X
    else:
        X_EK = flows
    
    X_O_E = np.sum(X_EK, axis=1)
    c = 0
    for e, (s, d, c_e) in enumerate(graph.edges(data='capacity')):
        demand = X_O_E[e]
        if is_congested(c_e, demand):
            if report:
                cap_str = str(np.round(c_e, 4))
                demand_str = str(np.round(demand, 4))
                print(f"Link {s} --> {d} Is Congested: {demand_str} > {cap_str}")
            c += 1
    
    congesteds = c / N
    if congesteds == 0:
        print(f"{ANSIColors.BOLD}{ANSIColors.OKGREEN}ALL LINK CAPCITIES WERE HONORED{ANSIColors.ENDC}")
    else:
        print(f"{ANSIColors.BOLD}{ANSIColors.FAIL}" + "{:.1f}% OF LINKS ARE CONGESTED".format(congesteds*100) + f"{ANSIColors.ENDC}")


def check_distributed_flow_conservation(
        flows: List[gurobipy.MVar], graph: nx.DiGraph, out_index_mapping: Dict[Tuple[int, int], int], 
        commodities: List[Commodity], feasibility_tol: float = te.constants.DEFAULT_FEASIBILITY_TOLERANCE
    ):

    for k, commodity in enumerate(commodities):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand

        flow_out = defaultdict(list)
        flow_in = defaultdict(list)
        for edge in graph.edges():
            v = edge[0]
            i = out_index_mapping[edge]
            flow_out[edge[0]].append(flows[v][i, k].X)
            flow_in[edge[1]].append(flows[v][i, k].X)
        
        for v in graph.nodes():
            fout = sum(flow_out[v])
            fin  = sum(flow_in[v])

            demand_str = str(np.round(DEMAND, 4))
            fout_str = str(np.round(fout, 4))
            fin_str = str(np.round(fin, 4))
        
            if v == SOURCE:
                assert abs(fout - DEMAND) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand outflow does not hold at source: {fout_str} vs {demand_str}"
                assert abs(fin) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Source receives its own demand! {fin_str}"
            elif v == DESTINATION:
                assert abs(fout) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Destination is leaking demand! {fout_str}"
                assert abs(fin - DEMAND) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand inflow does not hold at destination: {fin_str} vs {demand_str}"
            else:
                assert abs(fout - fin) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Transit demand conservation does not hold: {fin_str} --> {fout_str}"

