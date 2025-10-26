import argparse
import contextlib
from typing import Optional, Tuple
from te.traffic_models.converters import SampledConverter, SampledTrafficMatrixConverterParams
from te.algorithms.base import (SolverParams, TrafficEngineeringLP, TrafficEngineeringLPEvaluationParams, 
                                TrafficEngineeringLPSolutionParams, TrafficEngineeringLPWarmStartParams)
from te.algorithms.utils import get_solution_confusion_matrix, get_solution_maximum_utilization, stringify_collected_stats
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolutionParams, EdgeBasedMinimizeMaximumUtilitySolution
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic
from utils.logging import as_info, as_fail, log_subsection_title, log_section_title, str_round


def mlu_solve_and_check(lp: TrafficEngineeringLP, eval_params: TrafficEngineeringLPEvaluationParams):
    """
    Helper method that receives the LP object and the evaluation parameters and solves the MLU problem.

    Arguments
    ---------
    lp: type[TrafficEngineeringLP]
        The full LP object that we can use to solve the problem
    eval_params: type[TrafficEngineeringLPEvaluationParams]
        TE evaluation parameters.
    """
    t = lp.solve()
    if t > -1:
        print(as_info(log_subsection_title("CHECKING SOLUTION")))
        lp.check(eval_params)
        print(lp.check_result)
        print(as_info(f"Solved in {str_round(t, 2)} seconds"))
        print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
        print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
    else:
        print(as_fail("MLU problem couldn't be solved as expected"))


def mlu_helper(
    cls: type[TrafficEngineeringLP],
    solver_params: SolverParams, 
    eval_params: TrafficEngineeringLPEvaluationParams,
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams] = None,
    solution_params: Optional[TrafficEngineeringLPSolutionParams] = None
):
    """
    A helper for quickly creating and solving MLU problems.
    
    Arguments
    ---------
    cls: type[TrafficEngineeringLP]
        The TE LP _class_ to use
    solver_params: SolverParams
        The solver parameters for the given `cls` (it will not check if the two match
        though, you may or may not get an exception if there is a mismatch between the
        solver class and the parameters)
    eval_params: TrafficEngineeringLPEvaluationParams
        The set of evaluation parameters to use
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams]
        Optional set of warm-start parameters
    solution_params: Optional[TrafficEngineeringLPSolutionParams]
        Optional set of solution output parameters
    """
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)

    if eval_params.SaveSol:
        mlu_solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
            seed=eval_params.Seed, 
            topology_name=eval_params.TopologyName, 
            capacity=c,
            tm_model_name=tm.type(), 
            tm_model_params=tm.params,
            path=solution_params.Path, 
            sol_name=solution_params.Name
        )
    else:
        mlu_solution_params = None

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
    
    with contextlib.closing(cls(graph, tm, solver_params)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Evaluating With Parameters:\n{eval_params}"))
        print(as_info(f"Solving With Parameters:\n{solver_params}"))
        print(as_info(log_subsection_title("MAKING TE LP")))
        lp.make_lp()
        print(as_info(log_subsection_title(f"SOLVING WITH: {lp.alg_name}")))
        mlu_solve_and_check(lp, eval_params)
        
        if mlu_solution_params:
            if converter is None:
                solution = EdgeBasedMinimizeMaximumUtilitySolution(params=mlu_solution_params)
                lp.add_solution_elements(solution)
                solution.dump_elements()
                solution.dump(name=mlu_solution_params.Name)
            else:
                raise NotImplementedError('Will not save solution for warm-tests for now ... (takes too much space!)')
        
        if converter is not None:
            converted_tm = tm
            for i in range(warmstart_params.WarmIters):
                print(as_info(log_subsection_title(f"WARM-START ITERATION {i}")))
                converted_tm = converter.convert(tm)
                lp.update_traffic_matrix(converted_tm)
                mlu_solve_and_check(lp, eval_params)
        
        get_solution_confusion_matrix(lp, eval_params)
        
        stats = stringify_collected_stats()
        if stats is not None:
            print(as_info(stats))


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
