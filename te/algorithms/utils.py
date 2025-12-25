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
from utils.logging import as_info, as_warning, str_round, log_section_title, log_subsection_title
from te.traffic_models.base import TrafficMatrixBase
from te.algorithms.base import TrafficEngineeringLP, SolverParams, TrafficEngineeringLPEvaluationParams
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolution, EdgeBasedMinimizeMaximumUtilitySolutionParams
from te.algorithms.statistics.base import stringify_collected_stats
from te.algorithms.sub_algorithms.stretch import get_average_stretch



def get_solution_confusion_matrix(lp: TrafficEngineeringLP, eval_params: TrafficEngineeringLPEvaluationParams):
    """
    Plot the solution and output the objective trace.
    """
    def write_traces(_lp: TrafficEngineeringLP):
        objective_trace = _lp.objective_trace
        objective_gap_trace = _lp.objective_gap_trace
        average_stretch: np.ndarray = get_average_stretch(
            _lp.commodity_list,
            _lp.assignments,
            _lp.graph
        )
        if objective_trace is None:
            objective_trace = []
        else:
            objective_trace = objective_trace.trace
        if objective_gap_trace is None:
            objective_gap_trace = []
        with open(eval_params.TraceOutputPath, 'w') as traces:
            traces.writelines([
                f'objective_value: {",".join([str(item) for item in objective_trace])}\n',
                f'duality_gap: {",".join(str(item) for item in objective_gap_trace)}\n',
                f'average_stretch: {",".join(str(item) for item in average_stretch.tolist())}'
            ])

    def make_fig(_lp: TrafficEngineeringLP) -> Figure:
        avg_stretch = get_average_stretch(
            lp.commodity_list,
            lp.assignments,
            lp.graph
        )
        objective_trace = _lp.objective_trace
        objective_gap_trace = _lp.objective_gap_trace
        if objective_trace is None:
            print(as_warning("No trace of objective value is available"))
        if objective_gap_trace is None:
            print(as_warning("No trace of primal/dual objective gap is available"))
        if objective_gap_trace is None and objective_trace is None:
            fig = plt.figure(figsize=(4, 3))
            plt.subplot(1, 1, 1)
            sns.ecdfplot(avg_stretch)
            plt.title('Average Stretch (Hops)')
        elif objective_trace is not None and objective_gap_trace is None:
            fig = plt.figure(figsize=(8, 3))
            plt.subplot(1, 2, 1)
            objective_trace.plot()
            plt.title('Objective Trace')
            plt.subplot(1, 2, 2)
            sns.ecdfplot(avg_stretch)
            plt.title('Average Stretch (Hops)')
        else:
            fig = plt.figure(figsize=(12, 3))
            plt.subplot(1, 3, 1)
            objective_trace.plot()
            plt.title('Objective Trace')
            ax = plt.subplot(1, 3, 2)
            plt.plot(objective_gap_trace)
            ax.set_yscale('log')
            plt.title('Objective Gap Trace')
            plt.subplot(1, 3, 3)
            sns.ecdfplot(avg_stretch)
            plt.title('Average Stretch (Hops)')
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
