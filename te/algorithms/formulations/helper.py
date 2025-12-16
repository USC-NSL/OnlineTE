import contextlib
from te.algorithms.base import *
from te.algorithms.utils import get_solution_confusion_matrix, get_solution_maximum_utilization, stringify_collected_stats
from utils.logging import as_info, as_fail, log_subsection_title, str_round


def solve_lp_and_report(lp: TrafficEngineeringLP):
    """
    Helper method that receives the LP object and the evaluation parameters and solves the MLU problem.

    Arguments
    ---------
    lp: type[TrafficEngineeringLP]
        The full LP object that we can use to solve the problem
    """
    t = lp.solve()
    if t > -1:
        print(as_info(log_subsection_title("CHECKING SOLUTION")))
        lp.check()
        print(lp.check_result)
        print(as_info(f"Solved in {str_round(t, 2)} seconds"))
        print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
        print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
    else:
        print(as_fail("TE problem couldn't be solved as expected"))


def solve_te_and_check(
    problem: TrafficEngineeringProblemDescription, 
    solver_cls: type[TrafficEngineeringLP], 
    solver_params: SolverParams, 
    *args, **kwargs
):
    """
    Create the TE LP instance, solve it, and finally check it.
    
    Arguments
    ---------
    problem: TrafficEngineeringProblemDescription
        Full TE problem input and evaluation description
    solver_cls: type[TrafficEngineeringLP]
        TE solver class to instantiate
    solver_params: SolverParams
        TE solver parameters
    args, kwargs:
        Extra parameters to pass to `solver_cls` constructor.
    """
    with contextlib.closing(solver_cls(problem, solver_params, *args, **kwargs)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Evaluating With Parameters:\n{problem.EvalParams}"))
        print(as_info(f"Solving With Parameters:\n{solver_params}"))
        print(as_info(log_subsection_title("MAKING TE LP")))
        lp.make_lp()
        print(as_info(log_subsection_title(f"SOLVING WITH: {lp.alg_name}")))
        solve_lp_and_report(lp)

        # TODO: Handle the solution save case for warm-starts
        
        if problem.Solution:
            lp.add_and_dump_lp_solutions(problem.Solution)
        
        if problem.Converter is not None:
            converted_tm = problem.TM
            for i in range(problem.WarmStartParams.WarmIters):
                print(as_info(log_subsection_title(f"WARM-START ITERATION {i}")))
                converted_tm = problem.Converter.convert(converted_tm)
                lp.update_traffic_matrix(converted_tm)

                solve_lp_and_report(lp)
        
        get_solution_confusion_matrix(lp, problem.EvalParams)
        
        stats = stringify_collected_stats()
        if stats is not None:
            print(as_info(stats))


__all__ = [
    'solve_lp_and_report', 'solve_te_and_check'
]