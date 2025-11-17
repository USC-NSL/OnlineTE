import argparse
import contextlib
from typing import Optional, Tuple
from te.algorithms.base import *
from te.algorithms.utils import get_solution_confusion_matrix, get_solution_maximum_utilization, stringify_collected_stats
from te.traffic_models.converters import SampledConverter, SampledTrafficMatrixConverterParams
from te.algorithms.solution import (EdgeBasedMinimizeMaximumUtilitySolutionParams, 
                                    EdgeBasedMinimizeMaximumUtilitySolution)
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic
from utils.logging import as_info, as_fail, log_subsection_title, log_section_title, str_round


# @dataclass
# class MLUHelperParams:
#     """
#     The generic dataclass for any MLU problem.
    
#     Arguments
#     ---------
#     TELPCLS: type[TrafficEngineeringLP]
#         The class extending `TrafficEngineeringLP` that implements the TE problem.
#     AlgorithmSolverParams: SolverParams
#         Solver parameters for the algorithm to run. Must subclass `SolverParams`.
#     EvalParams: TrafficEngineeringLPEvaluationParams
#         Evaluation parameters for checking convergence, feasibility, etc.
#     WarmstartParams: Optional[TrafficEngineeringLPWarmStartParams]
#         Warm start parameters. Can be `None`, in which we just run the algorithm from scratch
#         and stop when it converges.
#     SolutionParams: Optional[TrafficEngineeringLPSolutionParams]
#         Parameters for saving solutions. If `None`, then solutions will not be saved.
#     """
#     TELPCLS: type[TrafficEngineeringLP]
#     AlgorithmSolverParams: SolverParams
#     EvalParams: TrafficEngineeringLPEvaluationParams
#     WarmstartParams: Optional[TrafficEngineeringLPWarmStartParams]
#     SolutionParams: Optional[TrafficEngineeringLPSolutionParams]

# @dataclass
# class MLUProblemDescription:
#     EvalParams: TrafficEngineeringLPEvaluationParams
#     Graph: nx.DiGraph
#     TM: TrafficMatrixBase
#     Converter: Optional[TrafficMatrixConverterBase]
#     WarmStartParams: Optional[TrafficEngineeringLPWarmStartParams]
#     Solution: Optional[TrafficEngineeringLPSolution]


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


def edge_based_mlu_input_helper(
    eval_params: TrafficEngineeringLPEvaluationParams,
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams] = None,
    solution_params: Optional[TrafficEngineeringLPSolutionParams] = None
) -> TrafficEngineeringProblemDescription:
    """
    A helper for quickly creating some (Edge-Based) MLU problem and its inputs.
    
    Arguments
    ---------
    eval_params: TrafficEngineeringLPEvaluationParams
        The set of evaluation parameters to use
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams]
        Optional set of warm-start parameters
    solution_params: Optional[TrafficEngineeringLPSolutionParams]
        Optional set of solution output parameters
    
    Returns
    -------
    mlu_problem: TrafficEngineeringProblemDescription
        Full description of some MLU problem.
    """
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(
        eval_params.TopologyName, 
        eval_params.Seed, 
        scale_factor=eval_params.ScaleFactor
    )

    if warmstart_params is not None:
        print(as_info(log_section_title("MLU PROBLEM (WITH WARM-START)")))
        # TODO: Implement different converter passing here ...
        converter = SampledConverter(
            seed=warmstart_params.ConverterSeed,
            params=warmstart_params.ConverterParams
        )
    else:
        print(as_info(log_section_title("MLU PROBLEM")))
        converter = None
    
    print(as_info(f"Network link capacity is: {str(round(c, 2))}"))

    if eval_params.SaveSol:
        if converter is None:
            solution = EdgeBasedMinimizeMaximumUtilitySolution(
                EdgeBasedMinimizeMaximumUtilitySolutionParams(
                seed=eval_params.Seed, 
                topology_name=eval_params.TopologyName, 
                capacity=c,
                tm_model_name=tm.type(), 
                tm_model_params=tm.params,
                path=solution_params.Path, 
                sol_name=solution_params.Name
            ))
        else:
            raise NotImplementedError('Will not save solution for warm-tests for now ... (takes too much space!)')
    else:
        solution = None
    
    return TrafficEngineeringProblemDescription(
        EvalParams=eval_params, Graph=graph, TM=tm, Converter=converter, 
        WarmStartParams=warmstart_params, Solution= solution
    )


def mlu_argparser(prog_name: str) -> argparse.ArgumentParser:
    """
    Helper utility for building the argument parser of an MLU problem.

    Arguments
    ---------
    prog_name: str
        Problem name to see when `help` is issued.
    
    Returns
    -------
    parser: argparse.ArgumentParser
        A partially completed argument parser. The fields that it contains 
        are described below.

    General Parameters
    ------------------
    `topo`: str
        The topology name
    `tm-seed`: int
        The RNG seed used to generate the TM
    Runtime Parameters
    ------------------
    `feas-tol`: float
        Contraint feasibility absolute tolerance
    `conv-tol`: float
        Objective value relative convergance tolerance
    `scale-factor`: float
        Link capacity scaling factor
    `report-unsat`: bool
        Whether or not to output the details of unsatisfied demands or
        congested links.
    Warm-Start Parameters
    ---------------------
    `converter-seed`: int
        RNG seed for the TM converter
    `warm-iters`: int
        Number of warm-start iterations
    Sampled TM Converter Parameters
    -------------------------------
    `delta-max`: float
        Maximum value of change to a single pertrubed demand
    `delta-min`: float
        Minimum value of change to a single perturbed demand
    `num-samples`: int
        Number of samples to perturb for each iteration
    Solution Output Parameters
    --------------------------
    `save-sol`: bool
        Whether or not to save the solution output
    `path-sol`: str
        Path to the directory to output the solution
    `name-sol`: str
        Name of teh output solution file
    """
    parser = argparse.ArgumentParser(prog_name)
    
    # Topology name and TM seed are _ALWAYS_ needed
    parser.add_argument('--topo', help='Topology name', required=True)
    parser.add_argument('--tm-seed', type=int, help='TM RNG seed', required=True)

    # These runtime parameters are also always needed
    runtime_params_group = parser.add_argument_group('Runtime Parameters')
    runtime_params_group.add_argument('--feas-tol', help='Feasibility tolerance', type=float, default=1e-3)
    runtime_params_group.add_argument('--conv-tol', help='Optimality tolerance', type=float, default=1e-3)
    runtime_params_group.add_argument('--scale-factor', type=float, default=10.0, 
                                      help='Link capacity scaling factor.')
    runtime_params_group.add_argument('--report-unsat', action='store_true', 
                                      help='Report unsatisfied commodity assignments.')

    # Parameters for warm-start tests
    warm_start_params_group = parser.add_argument_group('Warm Start Parameters')
    warm_start_params_group.add_argument('--converter-seed', type=int, help='RNG seed for TM converter')
    warm_start_params_group.add_argument('--warm-iters', type=int, help='Number of warm-start iterations')

    sampled_tm_converter_params_group = parser.add_argument_group('Sampled TM Converter Parameters')
    sampled_tm_converter_params_group.add_argument('--delta-max', type=float, default=0.4, 
                                                   help='Maximum change of demand value')
    sampled_tm_converter_params_group.add_argument('--delta-min', type=float, default=0.2, 
                                                   help='Minimum change of demand value')
    sampled_tm_converter_params_group.add_argument('--num-samples', type=int, default=10, 
                                                   help='Number of concurrent demand changes')

    # Parameters for recording solutions
    solution_params_group = parser.add_argument_group('Solution Handling Parameters')
    solution_params_group.add_argument('--save-sol', action='store_true', help='Save the final solution')
    solution_params_group.add_argument('--path-sol', help='Directory path to store the solution(s) in')
    solution_params_group.add_argument('--name-sol', help='Name (prefix) for solution files')
    
    return parser


def mlu_parse_args(parser: argparse.ArgumentParser) -> Tuple[
    TrafficEngineeringLPEvaluationParams, 
    Optional[TrafficEngineeringLPSolutionParams],
    Optional[TrafficEngineeringLPWarmStartParams],
    argparse.Namespace]:
    """
    Parse all the default arguments needed for the MLU problem.

    Arguments
    ---------
    parser: `argparse.ArgumentParser`
        The argument parser (assumed produced with `mlu_argparser`)
    
    Returns
    -------
    eval_params: TrafficEngineeringLPEvaluationParams
        The TE problem evaluation parameters
    solution_params: Optional[TrafficEngineeringLPSolutionParams]
        Solution output parameters
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams]
        Warm-start parameters
    args: argparse.Namespace
        The namespace object of parsed arguments to further process
    """
    args = parser.parse_args()
    eval_params = TrafficEngineeringLPEvaluationParams(
        TopologyName=args.topo, 
        Seed=args.tm_seed, 
        SaveSol=args.save_sol,
        ScaleFactor=args.scale_factor,
        FeasibilityTolerance=args.feas_tol, 
        FeasibilityRatio=None,
        PrintReports=args.report_unsat
    )

    converter_seed = args.converter_seed
    warm_iters = args.warm_iters
    assert (converter_seed is None and warm_iters is None) or (converter_seed is not None and warm_iters is not None)
    if warm_iters is not None:
        warm_start_params = TrafficEngineeringLPWarmStartParams(
            ConverterSeed=converter_seed,
            WarmIters=warm_iters,
            ConverterParams=SampledTrafficMatrixConverterParams(
                delta_max=args.delta_max,
                delta_min=args.delta_min,
                number_of_samples=args.number_samples
            )
        )
    else:
        warm_start_params = None
    
    if args.save_sol:
        solution_params = TrafficEngineeringLPSolutionParams(
            Name=args.name_sol, Path=args.path_sol
        )
    else:
        solution_params = None
    
    return eval_params, solution_params, warm_start_params, args


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