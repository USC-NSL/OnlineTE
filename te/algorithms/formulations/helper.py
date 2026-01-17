import contextlib
import jsonargparse
from typing import Optional, Tuple
from te.algorithms.base import *
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic, load_topology
from te.algorithms.utils import get_solution_confusion_matrix, get_solution_maximum_utilization, stringify_collected_stats
from te.traffic_models.models import FilebackedTrafficMatrix, FilebackedTrafficMatrixParams
from te.traffic_models.converters import SampledConverter, SampledTrafficMatrixConverterParams
from te.algorithms.solution import (EdgeBasedMinimizeMaximumUtilitySolutionParams, 
                                    EdgeBasedMinimizeMaximumUtilitySolution)
from utils.logging import as_info, as_fail, log_subsection_title, str_round, log_section_title


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
) -> Optional[TrafficEngineeringLPObjectiveTrace]:
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
    
    Returns
    -------
    trace: Optional[TrafficEngineeringLPObjectiveTrace]
        The objective trace (along with any debug info) recorded as
        the algorithm was executed.
    """
    with contextlib.closing(solver_cls(problem, solver_params, *args, **kwargs)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Evaluating With Parameters:\n{problem.EvalParams}"))
        print(as_info(f"Solving With Parameters:\n{solver_params.str_all()}"))
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
        
        return lp.objective_trace


def te_input_helper(
    eval_params: TrafficEngineeringLPEvaluationParams,
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams] = None,
    solution_params: Optional[TrafficEngineeringLPSolutionParams] = None
) -> TrafficEngineeringProblemDescription:
    """
    A helper for quickly creating some TE problem and its inputs.
    
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
    te_problem: TrafficEngineeringProblemDescription
        Full description of some TE problem.
    """
    if eval_params.Seed is not None and eval_params.TMPath is None:
        as_info(f"No TM path given. Generating matrix from seed {eval_params.Seed}")
        c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(
            eval_params.TopologyName, 
            eval_params.Seed, 
            scale_factor=eval_params.ScaleFactor
        )
    elif eval_params.Seed is None and eval_params.TMPath is not None:
        as_info(f"Loading matrix from path {eval_params.TMPath}")
        graph, has_cap = load_topology(eval_params.TopologyName)
        assert has_cap
        tm_params = FilebackedTrafficMatrixParams(eval_params.TMPath)
        caps = set([d['capacity'] for _, _, d in graph.edges(data=True)])
        # Check if all values are the same
        assert len(caps) == 1
        c = list(caps)[0]
        tm = FilebackedTrafficMatrix(params=tm_params)
    else:
        raise ValueError('Specifying both or neither of TM Seed and TM Path is most likely a mistake. Aborting ...')

    if warmstart_params is not None:
        print(as_info(log_section_title(f"{eval_params.Objective} PROBLEM (WITH WARM-START)")))
        # TODO: Implement different converter passing here ...
        converter = SampledConverter(
            seed=warmstart_params.ConverterSeed,
            params=warmstart_params.ConverterParams
        )
    else:
        print(as_info(log_section_title(f"{eval_params.Objective} PROBLEM")))
        converter = None
    
    print(as_info(f"Network link capacity is: {str(round(c, 2))}"))

    if eval_params.SaveSol:
        if converter is None:
            solution = EdgeBasedMinimizeMaximumUtilitySolution(
                EdgeBasedMinimizeMaximumUtilitySolutionParams(
                    TMSeed=eval_params.Seed, 
                    TopologyName=eval_params.TopologyName, 
                    Capacity=c,
                    TMModelName=tm.type(), 
                    TMModelParams=tm.params,
                    Path=solution_params.Path, 
                    Name=solution_params.Name
                )
            )
        else:
            raise NotImplementedError('Will not save solution for warm-tests for now ... (takes too much space!)')
    else:
        solution = None
    
    return TrafficEngineeringProblemDescription(
        EvalParams=eval_params, Graph=graph, TM=tm, Converter=converter, 
        WarmStartParams=warmstart_params, Solution= solution
    )


def te_problem_description_parser(prog_name: str) -> jsonargparse.ArgumentParser:
    """
    Helper utility that creates an argument parser for defining a random TE problem.

    Arguments
    ---------
    prog_name: str
        Problem name to see when `help` is issued.
    
    Returns
    -------
    parser: jsonargparse.ArgumentParser
        A partially completed argument parser. The fields that it contains 
        are described below.

    General Parameters
    ------------------
    `topo`: str
        The topology name (must in the Internet Topology Zoo)
    `tm-seed`: int
        The RNG seed used to generate the TM
    `objective`: TEObjective
        TE objective to solve for
    `tm-path`: Optional[str]
        Path to a TM file that will be loaded as a file-backed matrix
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
    parser = jsonargparse.ArgumentParser(prog_name)

    parser.add_argument('--config', action='config')
    
    # Topology name and TM seed are _ALWAYS_ needed
    parser.add_argument('--topo', help='Topology name', required=True)
    parser.add_argument('--tm-path', help='Path to a TM', type=Optional[str])
    parser.add_argument('--tm-seed', type=Optional[int], help='TM RNG seed')
    parser.add_argument('--objective', help='TE Objective', type=TEObjective, required=True)

    # These runtime parameters are also always needed
    runtime_params_group = parser.add_argument_group('Runtime Parameters')
    runtime_params_group.add_argument('--feas-tol', help='Feasibility tolerance', type=Optional[float], default=1e-3)
    runtime_params_group.add_argument('--conv-tol', help='Optimality tolerance', type=Optional[float], default=1e-3)
    runtime_params_group.add_argument('--scale-factor', type=float, default=10.0, 
                                      help='Link capacity scaling factor.')
    runtime_params_group.add_argument('--report-unsat', action='store_true', 
                                      help='Fully report unsatisfied commodity assignments (if False, only gives a summary)')

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
    
    trace_params_group = parser.add_argument_group('Runtime Trace Parameters')
    trace_params_group.add_argument('--path-trace', help='Path to store the runtime trace record file',
                                    default='res.txt')
    trace_params_group.add_argument('--path-plt', help='Path to store the runtime trace plot file',
                                    default='res.png')
    
    return parser


def parse_te_problem_description_args(parser: jsonargparse.ArgumentParser) -> Tuple[
    TrafficEngineeringProblemDescription,
    jsonargparse.Namespace]:
    """
    Parse all the default arguments needed for the TE problem.

    Arguments
    ---------
    parser: `jsonargparse.ArgumentParser`
        The argument parser (assumed produced with `te_argparser`)
    
    Returns
    -------
    problem_description: TrafficEngineeringProblemDescription
        Full description of our TE problem to pass to our solvers
    args: jsonargparse.Namespace
        The namespace object of parsed arguments to further process
    """
    args = parser.parse_args()
    eval_params = TrafficEngineeringLPEvaluationParams(
        TopologyName=args.topo, 
        Seed=args.tm_seed, 
        TMPath=args.tm_path,
        Objective=args.objective,
        SaveSol=args.save_sol,
        ScaleFactor=args.scale_factor,
        FeasibilityTolerance=args.feas_tol, 
        FeasibilityRatio=None,
        PrintReports=args.report_unsat,
        TraceOutputPath=args.path_trace,
        PLTOutputPath=args.path_plt
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
    
    problem_description = te_input_helper(
        eval_params=eval_params, warmstart_params=warm_start_params,
        solution_params=solution_params
    )
    
    return problem_description, args


__all__ = [
    'solve_lp_and_report', 'solve_te_and_check',
    'te_input_helper', 'parse_te_problem_description_args', 
    'te_problem_description_parser'
]