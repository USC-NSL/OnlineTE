import argparse
import contextlib
from typing import Optional
from gurobipy import GRB
from te.algorithms.formulations.aggregate import CentralizedEdgeBasedLP, GurobiSolverParams, DualCentralizedEdgeBasedLP
from te.algorithms.base import TrafficEngineeringLPEvaluationParams
from te.algorithms.utils import get_solution_confusion_matrix, get_solution_maximum_utilization
from te.algorithms.solution import EdgeBasedMinimizeMaximumUtilitySolutionParams, EdgeBasedMinimizeMaximumUtilitySolution, default_solution_name
from te.traffic_models.converters import SampledConverter, SampledTrafficMatrixConverterParams, TrafficMatrixConverterParamsBase
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic
from utils.logging import as_info, as_warning, log_subsection_title, log_section_title, str_round

import warnings
warnings.filterwarnings("error")


SOLVER_PARAMS = GurobiSolverParams()
CONVERTER_PARAMS: Optional[SampledTrafficMatrixConverterParams] = None

CONVERTER_SEED: Optional[int] = None
CONVERTER_ITERS: Optional[int] = None


def centralized_test(solver_params: GurobiSolverParams, eval_params: TrafficEngineeringLPEvaluationParams,
                     converter_params: Optional[TrafficMatrixConverterParamsBase] = None,
                     converter_seed: Optional[int] = None,
                     converter_iters: Optional[int] = None,
                     dual: bool = False):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(eval_params.TopologyName, eval_params.Seed, scale_factor=eval_params.ScaleFactor)
    print(f"Network link capacity is: {str(round(c, 2))}")

    if converter_params is not None and eval_params.SaveSol:
        raise NotImplementedError("Save sol for warm start not implemented yet ...")

    solution_params = None
    if eval_params.SaveSol:
        solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
            seed=eval_params.Seed, topology_name=eval_params.TopologyName, capacity=c,
            tm_model_name=tm.type(), tm_model_params=tm.params,
            path=None, sol_name=default_solution_name(
                topology_name=eval_params.TopologyName, rng_seed=eval_params.Seed, tm_type=tm.type(),
                method=solver_params.Method, crossover=solver_params.Crossover
            )
        )

    if converter_params is not None:
        assert converter_iters is not None and converter_seed is not None
        print(as_info(log_section_title("MLU PROBLEM (WITH WARM-START)")))
        converter = SampledConverter(converter_seed, converter_params)
    else:
        print(as_info(log_section_title("MLU PROBLEM")))
        converter = None
    
    cls = DualCentralizedEdgeBasedLP if dual else CentralizedEdgeBasedLP
    
    with contextlib.closing(cls(graph, tm, SOLVER_PARAMS)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Evaluating With Parameters:\n{eval_params}"))
        print(as_info(f"Solving With Parameters:\n{solver_params}"))

        if converter is not None:
            print(as_info(log_subsection_title("BASELINE SOLUTION")))
        
        print(as_info(log_subsection_title("MAKING TE LP")))
        lp.make_lp()
        print(as_info(log_subsection_title(f"SOLVING WITH: {lp.alg_name}")))
        t = lp.solve()
        print(as_info(log_subsection_title("CHECKING SOLUTION")))
        if t > 0:
            lp.check(eval_params)
            print(lp.check_result)
            if converter is None:
                get_solution_confusion_matrix(lp, eval_params)
            print(as_info(f"Solved in {str_round(t, 2)} seconds"))
            print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
            print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
        if solution_params:
            if converter is None:
                solution = EdgeBasedMinimizeMaximumUtilitySolution(params=solution_params)
                lp.add_solution_elements(solution)
                solution.dump_elements()
                solution.dump(name=solution_params.sol_name)
            else:
                print(as_warning('Will not save solution for warm-tests for now ... (takes too much space!)'))
        
        if converter is not None:
            converted_tm = tm
            for i in range(converter_iters):
                print(as_info(log_subsection_title(f"WARM-START ITERATION {i}")))
                converted_tm = converter.convert(tm)
                lp.update_traffic_matrix(converted_tm)
                t = lp.solve()
                if t > 0:
                    lp.check(eval_params)
                    print(lp.check_result)
                    print(as_info(f"Solved in {str_round(t, 2)} seconds"))
                    print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
                    print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
            get_solution_confusion_matrix(lp, eval_params)
        
    # test_mlu(CentralizedEdgeBasedLP, graph, tm, solver_params, eval_params, solution_params=solution_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simple distributed test')
    
    parser.add_argument('topo', help='Topology name')
    parser.add_argument('seed', type=int, help='RNG seed')
    
    solver_params_group = parser.add_argument_group('Solver Parameters', description='Gurobi solver parameters')
    solver_params_group.add_argument('--method', help='Gurobi method to use', choices=['BARRIER', 'SIM-PRIMAL', 'SIM-DUAL'], default='BARRIER')
    solver_params_group.add_argument('--focus', help='Gurobi numeric focus', type=int, choices=[0, 1, 2, 3], default=SOLVER_PARAMS.NumericFocus)
    solver_params_group.add_argument('--feas-tol', help='Feasibility tolerance', type=float, default=SOLVER_PARAMS.FeasibilityTol)
    solver_params_group.add_argument('--conv-tol', help='Optimality tolerance', type=float, default=SOLVER_PARAMS.ConvTol)
    solver_params_group.add_argument('--presolve', help='Perform presolve', action='store_true')
    solver_params_group.add_argument('--crossover', action='store_true', help='(BARRIER only) perform crossover after barrier solver ends')
    solver_params_group.add_argument('--threads', help='(BARRIER only) Max number of threads', type=int, default=SOLVER_PARAMS.Threads)
    solver_params_group.add_argument('--log-to', help='Log file path', default=SOLVER_PARAMS.LogFile)

    runtime_params_group = parser.add_argument_group('Runtime Parameters')
    runtime_params_group.add_argument('--save-sol', action='store_true', help='Save the final solution')
    runtime_params_group.add_argument('--scale-factor', type=float, default=10.0, 
                                      help='Link capacity scaling factor.')
    runtime_params_group.add_argument('--report-unsat', action='store_true', 
                                      help='Report unsatisfied commodity assignments.')

    warm_start_params_group = parser.add_argument_group('Warm Start Parameters')
    warm_start_params_group.add_argument('--converter-seed', type=int, help='RNG seed for TM converter')
    warm_start_params_group.add_argument('--warm-iters', type=int, help='Number of warm-start iterations')
    warm_start_params_group.add_argument('--delta-max', type=float, default=0.4, 
                                         help='Maximum change of demand value')
    warm_start_params_group.add_argument('--delta-min', type=float, default=0.2, 
                                         help='Minimum change of demand value')
    warm_start_params_group.add_argument('--num-samples', type=int, default=10, 
                                         help='Number of concurrent demand changes')

    dual_solver_params_group = parser.add_argument_group('Dual Solver')
    dual_solver_params_group.add_argument('--dual', action='store_true', help='Calculate all dual variables and verify dual feasibility')

    args = parser.parse_args()

    method_map = {
        'BARRIER': GRB.METHOD_BARRIER,
        'SIM-PRIMAL': GRB.METHOD_PRIMAL,
        'SIM-DUAL': GRB.METHOD_DUAL
    }

    SOLVER_PARAMS = GurobiSolverParams(
        Method=method_map[args.method],
        NumericFocus=args.focus,
        FeasibilityTol=args.feas_tol,
        ConvTol=args.conv_tol,
        Presolve=args.presolve,
        Crossover=args.crossover,
        Threads=args.threads,
        LogFile=args.log_to
    )

    EVAL_PARAMS = TrafficEngineeringLPEvaluationParams(
        TopologyName=args.topo, Seed=args.seed, ScaleFactor=args.scale_factor,
        FeasibilityTolerance=args.feas_tol, 
        FeasibilityRatio=None,
        PrintReports=args.report_unsat,
        SaveSol=args.save_sol
    )

    CONVERTER_SEED = args.converter_seed
    if CONVERTER_SEED is not None:
        CONVERTER_PARAMS = SampledTrafficMatrixConverterParams(
            delta_max=args.delta_max, delta_min=args.delta_min, 
            number_of_samples=args.num_samples
        )
        CONVERTER_ITERS = args.warm_iters
    
    centralized_test(SOLVER_PARAMS, EVAL_PARAMS, CONVERTER_PARAMS, CONVERTER_SEED, CONVERTER_ITERS, dual=args.dual)
