import jsonargparse
from typing import Optional, Tuple
from te.algorithms.base import *
from te.traffic_models.converters import SampledConverter, SampledTrafficMatrixConverterParams
from te.algorithms.solution import (EdgeBasedMinimizeMaximumUtilitySolutionParams, 
                                    EdgeBasedMinimizeMaximumUtilitySolution)
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic
from utils.logging import as_info, log_section_title


def edge_based_te_input_helper(
    eval_params: TrafficEngineeringLPEvaluationParams,
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams] = None,
    solution_params: Optional[TrafficEngineeringLPSolutionParams] = None
) -> TrafficEngineeringProblemDescription:
    """
    A helper for quickly creating some (Edge-Based) TE problem and its inputs.
    
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
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(
        eval_params.TopologyName, 
        eval_params.Seed, 
        scale_factor=eval_params.ScaleFactor
    )

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
    parser.add_argument('--tm-seed', type=int, help='TM RNG seed', required=True)
    parser.add_argument('--objective', help='TE Objective', type=TEObjective, required=True)

    # These runtime parameters are also always needed
    runtime_params_group = parser.add_argument_group('Runtime Parameters')
    runtime_params_group.add_argument('--feas-tol', help='Feasibility tolerance', type=float, default=1e-3)
    runtime_params_group.add_argument('--conv-tol', help='Optimality tolerance', type=float, default=1e-3)
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
    
    problem_description = edge_based_te_input_helper(
        eval_params=eval_params, warmstart_params=warm_start_params,
        solution_params=solution_params
    )
    
    return problem_description, args


__all__ = [
    'edge_based_te_input_helper', 'parse_te_problem_description_args', 
    'te_problem_description_parser'
]