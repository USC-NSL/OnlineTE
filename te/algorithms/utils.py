import gurobipy
import contextlib
import numpy as np
import seaborn as sns
import networkx as nx
import te.constants
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Dict, Union, Optional, Type
from collections import defaultdict
from te.traffic_models.base import Commodity, TrafficMatrixBase
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


as_bold = lambda msg: f"{ANSIColors.BOLD}{msg}{ANSIColors.ENDC}"
as_warning = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.WARNING}{msg}{ANSIColors.ENDC}"
as_info = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.OKBLUE}{msg}{ANSIColors.ENDC}"
as_success = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.OKGREEN}{msg}{ANSIColors.ENDC}"
as_fail = lambda msg: f"{ANSIColors.BOLD}{ANSIColors.FAIL}{msg}{ANSIColors.ENDC}"


method_to_str = {
    gurobipy.GRB.METHOD_BARRIER: "BARRIER",
    gurobipy.GRB.METHOD_PRIMAL: "PRIMAL-SIMPLEX",
    gurobipy.GRB.METHOD_DUAL: "DUAL-SIMPLEX"
}


def make_model(name: str, params: SolverParams, env: Optional[gurobipy.Env], **kwargs):
    assert issubclass(params.__class__, GurobiSolverParams)
    model = gurobipy.Model(name=name, env=env)
    model.Params.Method = params.Method
    model.Params.Crossover = params.Crossover
    model.Params.NumericFocus = params.NumericFocus

    # We set _both_ of these to the same value to make sure that both
    # Barrier and Simplex converge to within the same tolerance.
    model.Params.BarConvTol = params.ConvTol
    model.Params.OptimalityTol = params.ConvTol

    model.Params.FeasibilityTol = params.FeasibilityTol
    model.Params.LogFile = params.LogFile
    model.Params.Presolve = params.Presolve

    if len(kwargs) > 0:
        for k, v in kwargs.items():
            setattr(model.Params, k, v)

    print(as_bold(
        "Created Gurobi Model With:\n"
        f"\tMethod: {method_to_str[params.Method]}\n"
        f"\tOptimality Tolerance (OptimalityTol/BarConvTol): {params.ConvTol}\n"
        f"\tCosntraint Feasibility Tolerance (FeasibilityTol): {params.FeasibilityTol}\n"
    ))

    return model


def optimize_or_scream(model: gurobipy.Model):
    """Solve a Gurobi model. Throw an error if the model ends up in any non-optimal state"""
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        raise RuntimeError(as_fail(f"Optimizing model {model.ModelName} returned non-optimal status: {model.Status}"))


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
        if optim < te.constants.FLOAT_RES:
            return abs(actual) < te.constants.FLOAT_RES
        return abs((optim - actual) / optim) < feasibility_ratio

    def make_fig(_cm: np.ndarray, _lp: TrafficEngineeringLP) -> Figure:
        _objective_trace = _lp.objective_trace
        _objective_gap_trace = _lp.objective_gap_trace
        _solver_params = _lp.params
        rho_coeff_trace = None
        if hasattr(_solver_params, 'UseVariableRho'):
            if _solver_params.UseVariableRho:
                print(as_info("ADMM algorithm used variable step sizes. Will plot that too"))
                rho_coeff_trace = _lp.rho_coeff_trace
                eta_coeff_trace = _lp.eta_coeff_trace
        if _objective_trace is None:
            print(as_warning("No trace of objective value is available"))
        if _objective_gap_trace is None:
            print(as_warning("No trace of primal/dual objective gap is available"))
        else:
            if rho_coeff_trace is None:
                if _objective_gap_trace is None and _objective_trace is None:
                    _fig = plt.figure(figsize=(4, 3))
                    sns.heatmap(cm)
                elif _objective_trace is not None and _objective_gap_trace is None:
                    _fig = plt.figure(figsize=(8, 3))
                    plt.subplot(1, 2, 1)
                    plt.plot(_objective_trace)
                    plt.subplot(1, 2, 2)
                    sns.heatmap(_cm, vmin=0, vmax=1)
                else:
                    _fig = plt.figure(figsize=(12, 3))
                    plt.subplot(1, 3, 1)
                    plt.plot(_objective_trace)
                    ax = plt.subplot(1, 3, 2)
                    plt.plot(_objective_gap_trace)
                    ax.set_yscale('log')
                    plt.subplot(1, 3, 3)
                    sns.heatmap(_cm, vmin=0, vmax=1)
            else:
                if _objective_gap_trace is None and _objective_trace is None:
                    _fig = plt.figure(figsize=(4, 6))
                    plt.subplot(2, 2, 1)
                    sns.heatmap(cm)
                    ax = plt.subplot(2, 2, 3)
                    ax.set_yscale('log')
                    plt.plot(rho_coeff_trace)
                    ax = plt.subplot(2, 2, 4)
                    ax.set_yscale('log')
                    plt.plot(eta_coeff_trace)
                elif _objective_trace is not None and _objective_gap_trace is None:
                    _fig = plt.figure(figsize=(8, 6))
                    plt.subplot(2, 2, 1)
                    plt.plot(_objective_trace)
                    plt.subplot(2, 2, 2)
                    sns.heatmap(cm)
                    ax = plt.subplot(2, 2, 3)
                    ax.set_yscale('log')
                    plt.plot(rho_coeff_trace)
                    ax = plt.subplot(2, 2, 4)
                    ax.set_yscale('log')
                    plt.plot(eta_coeff_trace)
                else:
                    _fig = plt.figure(figsize=(12, 6))
                    plt.subplot(2, 3, 1)
                    plt.plot(_objective_trace)
                    plt.subplot(2, 3, 2)
                    sns.heatmap(cm)
                    ax = plt.subplot(2, 3, 3)
                    ax.set_yscale('log')
                    plt.plot(_objective_gap_trace)
                    ax = plt.subplot(2, 3, 4)
                    ax.set_yscale('log')
                    plt.plot(rho_coeff_trace)
                    ax = plt.subplot(2, 3, 5)
                    ax.set_yscale('log')
                    plt.plot(eta_coeff_trace)
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
        is_not_satisfied = not is_satisfied(ideal.demand, actual[0].demand)
        if is_not_satisfied:
            cm[ideal.source, ideal.destination] = 1
            unsats += 1        
        
        if report:
            report_str = "{:<4} -> {:<4}".format(ideal.source, ideal.destination) + \
                "{:^10}    {:^7} <--> {:^7}".format(
                    str(np.round(ideal.demand, 2)),
                    str(np.round(actual[0].demand, 2)),
                    str(np.round(actual[1].demand, 2))
                )
            if is_not_satisfied:
                print(as_fail(report_str))
            else:
                print(report_str)
    
    unsatisfied = unsats / K

    if unsats == 0:
        print(as_success("ALL DEMANDS WERE SATISFIED"))
    else:
        print(as_fail("{:.1f}% OF DEMANDS WERE NOT SATISFIED".format(unsatisfied*100)))
    
    if show or save_fig:
        fig = make_fig(cm, lp)
        if fig is not None:
            if show:
                plt.show()
            if save_fig:
                fig.savefig('res.png')

    return (unsatisfied, cm)


def get_solution_maximum_utilization(assignments: np.ndarray, graph: nx.DiGraph) -> float:
    assert len(np.shape(assignments)) == 2
    flows = np.sum(assignments, axis=1)
    u = 0
    for e, (_, _, c_e) in enumerate(graph.edges(data='capacity')):
        this_u = flows[e] / c_e
        if u < this_u:
            u = this_u
    return u


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
        print(as_success("ALL LINK CAPCITIES WERE HONORED"))
    else:
        print(as_fail("{:.1f}% OF LINKS ARE CONGESTED".format(congesteds*100)))


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


def dykstra_proj(x_0: np.ndarray, A: np.ndarray, x: np.ndarray, feasibility_tol: float, 
                 max_iter: int = -1) -> Tuple[np.ndarray, bool]:
    """
    Use Dykstra' algorithm to project point `x` onto the nearest point `y` that
    satisfies:

        0 \leq x_0 + A @ y
    """
    assert len(np.shape(x)) == 1 or (len(np.shape(x)) == 2 and np.shape(x)[-1] == 1)

    # Should we even do anything?
    if np.all((x_0 + A @ x) > 0):
        return x, True
    
    number_of_sets = np.shape(A)[0]
    dim = np.shape(x)[0]
    
    # TODO: Need to add the robust stopping criterion
    At = A.T
    increments = np.zeros((dim, number_of_sets))
    iterates = np.zeros((dim, number_of_sets+1))
    iterates[:, -1] = x
    iterates[:, 0] = x
    
    while True:
        max_div = 0
        counter = 0
        for p in range(number_of_sets):
            a = At[:, p]
            column, did_nothing = half_space_proj(x_0[p], a, iterates[:, p-1] + increments[:, p], feasibility_tol)
            if did_nothing:
                continue
            increments[:, p] = (iterates[:, p-1] + increments[:, p]) - column
            div = np.linalg.norm(column - iterates[:, p])
            iterates[:, p] = column
            if div > max_div:
                max_div = div
        counter += 1

        if (max_iter > 0 and counter == max_iter) or (max_div <= feasibility_tol):
            out = iterates[:, -2]
            return out, False


def half_space_proj(x_0: float, a: np.ndarray, x: np.ndarray, feasibility_tol: float) -> Tuple[np.ndarray, bool]:
    """
    Return the orthogonal projection of point `x` onto the half space:

        0 \leq x_0 + a.T @ x
    
    If the condition is already satisfied, return `(x, True)`. If not, then it returns
    the `(projection, False)`
    """

    d = x_0 + np.dot(a, x)
    if d >= -feasibility_tol:
        return x, True
    return x - (d / np.linalg.norm(a)**2) * a, False


all_elements_within_threshold = lambda x, thresh: np.all(np.abs(x) < thresh)


def careful_norm(x: np.ndarray, scaled: bool = False, axis: Optional[int] = None) -> float:
    if scaled:
        scale_factor = np.sqrt(x.size)
        if all_elements_within_threshold(x, te.constants.MINIMUM_NORM / scale_factor):
            return 0
        return np.linalg.norm(x) / scale_factor
    if all_elements_within_threshold(x, te.constants.MINIMUM_NORM) and axis is None:
        return 0
    return np.linalg.norm(x, axis=axis)


def careful_norm_squared(x: np.ndarray, axis: Optional[int] = None) -> float:
    if all_elements_within_threshold(x, te.constants.MINIMUM_NORM) and axis is None:
        return 0
    if axis is None:
        return np.dot(x, x)
    return np.linalg.norm(x, axis=axis) ** 2


def test_mlu(lp_cls: Type[TrafficEngineeringLP], graph: nx.DiGraph, tm: TrafficMatrixBase, solver_params: SolverParams,
             feasibility_tol: float = None, feasibility_ratio: float = None, **kwargs):
    with contextlib.closing(lp_cls(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio)
            get_solution_confusion_matrix(lp, feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio, **kwargs)
            print(as_info(f"Solved in {str(round(t, 2))} seconds"))
            print(as_info(f"Final objective value: {str(round(lp.objective_value, 4))}"))
            print(as_info(f"Actual utilization: {str(round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4))}"))
