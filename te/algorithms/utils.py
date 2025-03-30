import math
import gurobipy
import contextlib
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp
    cp.get_array_module = lambda x: np
import seaborn as sns
import networkx as nx
import te.constants
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Dict, Union, Optional, Type
from collections import defaultdict
from utils.logging import as_bold, as_fail, as_info, as_success, as_warning, method_to_str, str_round
from te.traffic_models.base import Commodity, TrafficMatrixBase
from te.algorithms.base import TrafficEngineeringLP, SolverParams, GurobiSolverParams
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution, EdgeBasedMinimizeMaximumUtilitySolutionParams
from te.algorithms.statistics.base import stringify_collected_stats


def make_model(name: str, params: SolverParams, env: Optional[gurobipy.Env], verbose: bool = True, **kwargs):
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

    model.Params.Threads = params.Threads

    if len(kwargs) > 0:
        for k, v in kwargs.items():
            setattr(model.Params, k, v)

    if verbose:
        print(as_bold(
            "Created Gurobi Model With:\n"
            f"\tMethod: {method_to_str[params.Method]}\n"
            f"\tSimplex Optimality Tolerance (OptimalityTol): {model.Params.OptimalityTol}\n"
            f"\tBarrier Optimality Tolerance (BarConvTol): {model.Params.BarConvTol}\n"
            f"\tCosntraint Feasibility Tolerance (FeasibilityTol): {model.Params.FeasibilityTol}\n"
        ))
    
    if model.Params.OptimalityTol != model.Params.BarConvTol:
        print(as_warning(f'Simplex and Barrier have different convergence tolerances. Make sure this is actualy intended to be!!'))

    return model


def optimize_or_scream(model: gurobipy.Model):
    """Solve a Gurobi model. Throw an error if the model ends up in any non-optimal state"""
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        raise RuntimeError(as_fail(f"Optimizing model {model.ModelName} returned non-optimal status: {model.Status}"))


def is_satisfied(optim, actual, feasibility_tol: Optional[float], feasibility_ratio: Optional[float]):
    """
    Check if `actual` is close to `optim` assignment.
    The test can either absolute or relative tolerance (if both are present, only
    absolute tolerance is considered).
    """
    if feasibility_tol is not None:
        return math.isclose(optim, actual, abs_tol=feasibility_tol)
    if abs(optim) < te.constants.FLOAT_RES:
        return math.isclose(actual, 0, abs_tol=te.constants.FLOAT_RES)
    return math.isclose(optim, actual, rel_tol=feasibility_ratio)


def is_negligible(actual, baseline, feasibility_tol: Optional[float], feasibility_ratio: Optional[float]):
    """
    If `feasibility_tol` is given, the it checks if absolute value of `actual` is
    within `min(feasibility_tol, baseline)`.
    If `feasibility_ratio` is present, it checks if the absolute value is within
    `baseline * feasibility_ratio` tolerance.
    """
    if abs(actual) < te.constants.FLOAT_RES:
        return True
    if feasibility_tol is not None:
        return abs(actual) < min(baseline, feasibility_tol)
    return abs(actual) < abs(baseline * feasibility_ratio)


def get_unsatisfied_demands(commodities: List[Commodity], solution: List[Tuple[Commodity, Commodity]],
                            feasibility_tol: Optional[float], feasibility_ratio: Optional[float]) -> \
                                List[Tuple[Commodity, Tuple[Commodity, Commodity]]]:
    """
    Check for demands that are not satisfied.
    To do this, it accepts a list of commodities as target (i.e. triple of `(src, dst, demand)`), and
    list of pairs of triplets (i.e. `(src, dst, demand-out)` and `(src, dst, demand-in)`).
    Of these triples, `src` and `dst` MUST always agree with the target, but demands may be different.
    
    Demands are checked to be within absolute/relative tolerance as given in arguments. Demands which
    were not satisfied are returned.
    """
    unsats: List[Tuple[Commodity, Tuple[Commodity, Commodity]]] = []
    for actual, ideal in zip(solution, commodities):
        assert actual[0].source == ideal.source
        assert actual[0].destination == ideal.destination
        assert actual[1].source == ideal.source
        assert actual[1].destination == ideal.destination
        if not is_satisfied(ideal.demand, actual[0].demand, feasibility_tol, feasibility_ratio):
            unsats.append((ideal, actual))
    return unsats


def get_solution_confusion_matrix(lp: TrafficEngineeringLP, feasibility_tol: Optional[float] = None, feasibility_ratio: Optional[float] = None, 
                                  report: bool = False, show: bool = True, save_fig: bool = True,
                                  trace_out_path: Optional[str] = 'res.txt') -> Tuple[float, np.ndarray]:
    """
    Check how many of the demands are not satisfied and report the solution.
    To check feasibility:
        - We can either check if the solution is within a particular _distance_ of the optimal
        - Or we can check if it has lower than a particular _error ratio_
    """
    assert (feasibility_ratio is None) ^ (feasibility_tol is None), "Exactly one of `feasibility_tol` or `feasibility_ratio` must be given"

    def write_traces(_lp: TrafficEngineeringLP):
        _solver_params = _lp.params
        _objective_trace = _lp.objective_trace
        _objective_gap_trace = _lp.objective_gap_trace
        if _objective_trace is None:
            _objective_trace = []
        if _objective_gap_trace is None:
            _objective_gap_trace = []
        _rho_coeff_trace = []
        _eta_coeff_trace = []
        if hasattr(_solver_params, 'UseVariableRho'):
            if _solver_params.UseVariableRho:
                _rho_coeff_trace = _lp.rho_coeff_trace
                _eta_coeff_trace = _lp.eta_coeff_trace
        with open(trace_out_path, 'w') as traces:
            traces.writelines([
                f'objective_value: {",".join([str(item) for item in _objective_trace])}\n',
                f'duality_gap: {",".join(str(item) for item in _objective_gap_trace)}\n',
                f'admm_step_coeff_1: {",".join(str(item) for item in _rho_coeff_trace)}\n',
                f'admm_step_coeff_2: {",".join(str(item) for item in _eta_coeff_trace)}\n'
            ])

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
                    traces = []
                    if len(_objective_trace) > 0:
                        item0 = _objective_trace[0]
                        if isinstance(item0, float):
                            traces.append(_objective_trace)
                        elif isinstance(item0, (tuple, list)):
                            traces.extend(list(zip(*_objective_trace)))
                        else:
                            as_warning(f'No idea how to plot objective samples of type: {type(item0)}')
                    plt.subplot(1, 2, 1)
                    for trace in traces:
                        plt.plot(trace)
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
    unsatisfied_demands = get_unsatisfied_demands(commodities, solution, feasibility_tol, feasibility_ratio)

    K = len(commodities)
    unsats = len(unsatisfied_demands)
    cm = np.zeros(shape=(topology_size, topology_size))

    if report:
        print(as_fail(" "*12 + "{:^10}    {:^20}".format("DESIRED", "ALLOCATED")))
        print(as_fail("-"*46))

    for ideal, actual in unsatisfied_demands:
        assert actual[0].source == ideal.source
        assert actual[0].destination == ideal.destination
        assert actual[1].source == ideal.source
        assert actual[1].destination == ideal.destination

        cm[ideal.source, ideal.destination] = 1
        
        if report:
            report_str = "{:<4} -> {:<4}".format(ideal.source, ideal.destination) + \
                "{:^10}    {:^7} <--> {:^7}".format(
                    str_round(ideal.demand, 2),
                    str_round(actual[0].demand, 2),
                    str_round(actual[1].demand, 2)
                )
            print(as_fail(report_str))
    
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
    
    if trace_out_path is not None:
        write_traces(lp)

    return (unsatisfied, cm)


def get_solution_maximum_utilization(assignments: np.ndarray, graph: nx.DiGraph) -> float:
    if len(np.shape(assignments)) == 1:
        flows = assignments
    else:
        flows = np.sum(assignments, axis=1)
    u = 0
    for e, (_, _, c_e) in enumerate(graph.edges(data='capacity')):
        this_u = flows[e] / c_e
        if u < this_u:
            u = this_u
    return u


def check_centralized_flow_conservation(
        flows: Union[gurobipy.tupledict, np.ndarray], graph: nx.DiGraph, 
        commodities: List[Commodity], feasibility_tol: Optional[float],
        feasibility_ratio: Optional[float] = None
    ):
    """
    Check if solution satisfies all of the following constraints:
        - Transit nodes conserve flows                                    ( flow conservation )
        - A demand destined to a node, never flows out from that node     (  no demand leaks  )
        - A demand sourced from a node, never flows back into that node   (      no loops     )
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

            demand_str = str_round(DEMAND, 4)
            fout_str = str_round(fout, 4)
            fin_str = str_round(fin, 4)

            if v == SOURCE:
                if not is_satisfied(fout, DEMAND, feasibility_tol, feasibility_ratio):
                    print(as_fail(f"Commodity {k}: Node {v} --> Demand outflow does not hold at source: {fout_str} vs {demand_str}"))
                if not is_negligible(fin, DEMAND, feasibility_tol, feasibility_ratio):
                    print(as_fail(f"Commodity {k}: Node {v} --> Source receives its own demand! {fin_str}"))
            elif v == DESTINATION:
                if not is_negligible(fout, DEMAND, feasibility_tol, feasibility_ratio):
                    print(as_fail(f"Commodity {k}: Node {v} --> Destination is leaking demand! {fout_str}"))
                if not is_satisfied(fin, DEMAND, feasibility_tol, feasibility_ratio):
                    print(as_fail(f"Commodity {k}: Node {v} --> Demand inflow does not hold at destination: {fin_str} vs {demand_str}"))
            else:
                if not is_satisfied(fout, fin, feasibility_tol, feasibility_ratio):
                    print(as_fail(f"Commodity {k}: Node {v} --> Transit demand conservation does not hold: {fin_str} --> {fout_str}"))


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
                cap_str = str_round(c_e, 4)
                demand_str = str_round(demand, 4)
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

            demand_str = str_round(DEMAND, 4)
            fout_str = str_round(fout, 4)
            fin_str = str_round(fin, 4)
        
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


all_elements_within_threshold = lambda x, thresh, mod: mod.all(mod.abs(x) < thresh)


def careful_norm(x: np.ndarray, scaled: bool = False, axis: Optional[int] = None) -> float:
    mod = cp.get_array_module(x)
    if scaled:
        scale_factor = np.sqrt(x.size)
        if all_elements_within_threshold(x, te.constants.MINIMUM_NORM / scale_factor, mod):
            return 0
        return mod.linalg.norm(x) / scale_factor
    if all_elements_within_threshold(x, te.constants.MINIMUM_NORM, mod) and axis is None:
        return 0
    return mod.linalg.norm(x, axis=axis)


def careful_norm_squared(x: np.ndarray, axis: Optional[int] = None) -> float:
    mod = cp.get_array_module(x)
    if all_elements_within_threshold(x, te.constants.MINIMUM_NORM, mod) and axis is None:
        return 0
    if axis is None:
        return mod.dot(x, x)
    return mod.linalg.norm(x, axis=axis) ** 2


def test_mlu(lp_cls: Type[TrafficEngineeringLP], graph: nx.DiGraph, tm: TrafficMatrixBase, solver_params: SolverParams,
             feasibility_tol: float = None, feasibility_ratio: float = None,
             solution_params: Optional[EdgeBasedMinimizeMaximumUtilitySolutionParams] = None, 
             **kwargs):
    print(as_info("="*60))
    print(as_info("="*23 + " MLU PROBLEM " + "="*24))
    print(as_info("="*60))
    with contextlib.closing(lp_cls(graph, tm, solver_params)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Solving With Parameters:\n{solver_params}"))
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio)
            get_solution_confusion_matrix(lp, feasibility_tol=feasibility_tol, feasibility_ratio=feasibility_ratio, **kwargs)
            print(as_info(f"Solved in {str_round(t, 2)} seconds"))
            print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
            print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
        stats = stringify_collected_stats()
        if stats is not None:
            print(as_info(stats))
        if solution_params:
            solution = EdgeBasedMinimizeMaximumUtilitySolution(params=solution_params)
            lp.add_solution_elements(solution)
            solution.dump_elements()
            solution.dump(name=solution_params.sol_name)
