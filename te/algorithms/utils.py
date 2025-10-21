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
from typing import Optional, Type
from utils.exceptions import SolutionInterrupted
from utils.logging import (as_bold, as_fail, as_info, as_warning, method_to_str, 
                           str_round, log_section_title, log_subsection_title)
from te.traffic_models.base import TrafficMatrixBase
from te.algorithms.base import TrafficEngineeringLP, SolverParams, GurobiSolverParams, TrafficEngineeringLPEvaluationParams
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
        print(as_info(as_bold(
            "Created Gurobi Model With:\n"
            f"\tMethod: {method_to_str[params.Method]}\n"
            f"\tSimplex Optimality Tolerance (OptimalityTol): {model.Params.OptimalityTol}\n"
            f"\tBarrier Optimality Tolerance (BarConvTol): {model.Params.BarConvTol}\n"
            f"\tCosntraint Feasibility Tolerance (FeasibilityTol): {model.Params.FeasibilityTol}\n"
        )))
    
    if model.Params.OptimalityTol != model.Params.BarConvTol:
        print(as_warning(f'Simplex and Barrier have different convergence tolerances. Make sure this is actualy intended to be!!'))

    return model


def optimize_or_scream(model: gurobipy.Model):
    """Solve a Gurobi model. Throw an error if the model ends up in any non-optimal state"""
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        if model.Status == gurobipy.GRB.INTERRUPTED:
            raise SolutionInterrupted
        else:
            raise RuntimeError(as_fail(f"Optimizing model {model.ModelName} returned non-optimal status: {model.Status}"))


def get_solution_confusion_matrix(lp: TrafficEngineeringLP, eval_params: TrafficEngineeringLPEvaluationParams):
    """Check how many of the demands are not satisfied and report the solution"""
    def write_traces(_lp: TrafficEngineeringLP):
        solver_params = _lp.params
        objective_trace = _lp.objective_trace
        objective_gap_trace = _lp.objective_gap_trace
        if objective_trace is None:
            objective_trace = []
        else:
            objective_trace = objective_trace.trace
        if objective_gap_trace is None:
            objective_gap_trace = []
        rho_coeff_trace = []
        eta_coeff_trace = []
        if hasattr(solver_params, 'UseVariableRho'):
            if solver_params.UseVariableRho:
                rho_coeff_trace = _lp.rho_coeff_trace
                eta_coeff_trace = _lp.eta_coeff_trace
        with open(eval_params.TraceOutputPath, 'w') as traces:
            traces.writelines([
                f'objective_value: {",".join([str(item) for item in objective_trace])}\n',
                f'duality_gap: {",".join(str(item) for item in objective_gap_trace)}\n',
                f'admm_step_coeff_1: {",".join(str(item) for item in rho_coeff_trace)}\n',
                f'admm_step_coeff_2: {",".join(str(item) for item in eta_coeff_trace)}\n'
            ])
    
    def get_cm(_lp: TrafficEngineeringLP):
        try:
            check_result = _lp.check_result
        except ValueError:
            _lp.check(eval_params)
            check_result = _lp.check_result
        
        commodities = lp.commodity_list
        topology_size = len(lp.graph.nodes)
        cm = np.zeros(shape=(topology_size, topology_size))
        for index in check_result.unsat_commodities:
            commodity = commodities[index]
            cm[commodity.source, commodity.destination] = 1
        return cm

    def make_fig(_lp: TrafficEngineeringLP) -> Figure:
        cm = get_cm(_lp)
        objective_trace = _lp.objective_trace
        objective_gap_trace = _lp.objective_gap_trace
        solver_params = _lp.params
        rho_coeff_trace = None
        if hasattr(solver_params, 'UseVariableRho'):
            if solver_params.UseVariableRho:
                print(as_info("ADMM algorithm used variable step sizes. Will plot that too"))
                rho_coeff_trace = _lp.rho_coeff_trace
                eta_coeff_trace = _lp.eta_coeff_trace
        if objective_trace is None:
            print(as_warning("No trace of objective value is available"))
        if objective_gap_trace is None:
            print(as_warning("No trace of primal/dual objective gap is available"))
        else:
            if rho_coeff_trace is None:
                if objective_gap_trace is None and objective_trace is None:
                    fig = plt.figure(figsize=(4, 3))
                    sns.heatmap(cm)
                elif objective_trace is not None and objective_gap_trace is None:
                    fig = plt.figure(figsize=(8, 3))
                    plt.subplot(1, 2, 1)
                    objective_trace.plot()
                    plt.subplot(1, 2, 2)
                    sns.heatmap(cm, vmin=0, vmax=1)
                else:
                    fig = plt.figure(figsize=(12, 3))
                    plt.subplot(1, 3, 1)
                    # plt.xlim([100, 300])
                    objective_trace.plot()
                    ax = plt.subplot(1, 3, 2)
                    plt.plot(objective_gap_trace)
                    ax.set_yscale('log')
                    plt.subplot(1, 3, 3)
                    sns.heatmap(cm, vmin=0, vmax=1)
            else:
                if objective_gap_trace is None and objective_trace is None:
                    fig = plt.figure(figsize=(4, 6))
                    plt.subplot(2, 2, 1)
                    sns.heatmap(cm)
                    ax = plt.subplot(2, 2, 3)
                    ax.set_yscale('log')
                    plt.plot(rho_coeff_trace)
                    ax = plt.subplot(2, 2, 4)
                    ax.set_yscale('log')
                    plt.plot(eta_coeff_trace)
                elif objective_trace is not None and objective_gap_trace is None:
                    fig = plt.figure(figsize=(8, 6))
                    plt.subplot(2, 2, 1)
                    plt.plot(objective_trace)
                    plt.subplot(2, 2, 2)
                    sns.heatmap(cm)
                    ax = plt.subplot(2, 2, 3)
                    ax.set_yscale('log')
                    plt.plot(rho_coeff_trace)
                    ax = plt.subplot(2, 2, 4)
                    ax.set_yscale('log')
                    plt.plot(eta_coeff_trace)
                else:
                    fig = plt.figure(figsize=(12, 6))
                    plt.subplot(2, 3, 1)
                    objective_trace.plot()
                    plt.subplot(2, 3, 2)
                    sns.heatmap(cm)
                    ax = plt.subplot(2, 3, 3)
                    ax.set_yscale('log')
                    plt.plot(objective_gap_trace)
                    ax = plt.subplot(2, 3, 4)
                    ax.set_yscale('log')
                    plt.plot(rho_coeff_trace)
                    ax = plt.subplot(2, 3, 5)
                    ax.set_yscale('log')
                    plt.plot(eta_coeff_trace)
            return fig
    
    if eval_params.ShowPLT or eval_params.SavePLT:
        fig = make_fig(lp)
        if fig is not None:
            if eval_params.ShowPLT:
                plt.show()
            if eval_params.SavePLT:
                fig.savefig(eval_params.PLTOutputPath)
    
    if eval_params.TraceOutputPath is not None:
        write_traces(lp)


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


def test_mlu(lp_cls: Type[TrafficEngineeringLP], graph: nx.DiGraph, tm: TrafficMatrixBase, 
             solver_params: SolverParams,
             eval_params: TrafficEngineeringLPEvaluationParams,
             solution_params: Optional[EdgeBasedMinimizeMaximumUtilitySolutionParams] = None, ):
    print(as_info(log_section_title("MLU PROBLEM")))
    with contextlib.closing(lp_cls(graph, tm, solver_params)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Evaluating With Parameters:\n{eval_params}"))
        print(as_info(f"Solving With Parameters:\n{solver_params}"))
        print(as_info(log_subsection_title("MAKING TE LP")))
        lp.make_lp()
        print(as_info(log_subsection_title(f"SOLVING WITH: {lp.alg_name}")))
        t = lp.solve()
        print(as_info(log_subsection_title("CHECKING SOLUTION")))
        if t >= 0:
            lp.check(eval_params)
            print(lp.check_result)
            get_solution_confusion_matrix(lp, eval_params)
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
