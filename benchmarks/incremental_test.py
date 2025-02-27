import time
import gurobipy
import contextlib
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt
from te.traffic_models import get_traffic_model, get_traffic_converter, get_traffic_converter_params
from te.algorithms.solution import (
    GurobiEdgeBasedMinimizeMaximumUtilitySolution, GurobiEdgeBasedMinimizeMaximumUtilityShiftedSolution)
from te.traffic_models.models import UniformTrafficMatrixParams
from te.traffic_models.base import TrafficMatrixConverterParamsBase
from te.traffic_models.converters import UniformConverter
from te.algorithms.base import GurobiSolverParams
from te.algorithms.formulations.edge_based_centralized import CentralizedEdgeBasedLP
from te.algorithms.formulations.edge_based_unregulated_admm import UnregulatedADMMLP, UnregulatedADMMSolverParams
from te.algorithms.utils import get_solution_confusion_matrix, get_solution_maximum_utilization
from topologies.utils import load_zoo_topology, get_capacity_lower_bound, set_edge_capacity_to


def get_baseline_solution(base_seed: int, topology_name: str, tm_model: str, 
                          convergence_tol: float, feasibility_tol: float, barrier: bool = False):
    """This function generates the optimal basis for the base TM and all shifts"""
    gurobi_sol_name = f'{topology_name}_{base_seed}_{tm_model}'
    solution_name = f'{gurobi_sol_name}.tesol'
    graph = load_zoo_topology(topology_name)
    tm_params = UniformTrafficMatrixParams(n = len(graph.nodes), min = 0.0, max = 1.0)
    tm = get_traffic_model(tm_model)(seed=base_seed, params=tm_params)
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)

    solver_params = GurobiSolverParams()

    if not barrier:
        solver_params.Method = gurobipy.GRB.METHOD_PRIMAL
        solver_params.Presolve = 1
    else:
        solver_params.Method = gurobipy.GRB.METHOD_BARRIER
        # We need a basic solution, thus, we need to enable crossover
        solver_params.Crossover = 1
    solver_params.ConvTol = convergence_tol
    solver_params.FeasibilityTol = feasibility_tol
    
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol, report=False, show=False, save_fig=False)
            print(f"Solved in {str(round(t, 4))} seconds. Final objective value: {str(round(lp.objective_value, 4))}")
            print(f"Actual utilization: {get_solution_maximum_utilization(lp.assignments, lp.graph)}")
            solution = GurobiEdgeBasedMinimizeMaximumUtilitySolution(
                seed=base_seed, topology_name=topology_name, capacity=c_min*10, tm_model_name=tm_model,
                tm_model_params=tm_params, gurobi_sol_name=gurobi_sol_name, runtime=t
            )
            solution.dump(model=lp._model, name=solution_name)


def get_baseline_shifted_solutions(base_seed: int, topology_name: str, tm_model: str,
                                   converter_model: str, converter_params: TrafficMatrixConverterParamsBase,
                                   converter_seed: int, number_of_shifts: int, 
                                   convergence_tol: float, feasibility_tol: float,
                                   warm_start: bool = True, 
                                   save_solutions: bool = True, 
                                   sol_name_postfix: Optional[str] = None,
                                   check_solution: bool = False):
    """This function generates baseline solutions for shifted demands, by default it warm starts Gurobi"""
    base_solution_name = f'{topology_name}_{base_seed}_{tm_model}.tesol'
    if warm_start:
        base_solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution = GurobiEdgeBasedMinimizeMaximumUtilitySolution.load(name=base_solution_name)
        graph, tm = base_solution.regenerate()
    else:
        # TODO: Add the option to refuse warm start ...
        raise NotImplementedError
    converter = get_traffic_converter(name=converter_model)(seed=converter_seed, params=converter_params)
    solver_params = GurobiSolverParams()
    solver_params.Presolve = 1
    solver_params.ConvTol = convergence_tol
    solver_params.FeasibilityTol = feasibility_tol
    solver_params.Method = gurobipy.GRB.METHOD_PRIMAL
    solution_times: List[float] = []
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        # First, initiate to baseline solution
        lp.initialize_to(base_solution)
        # Now, start shifting the demands
        shifted_tm = tm
        for i in range(number_of_shifts):
            print("="*10 + f" ITERATION {i+1} / {number_of_shifts} " + "="*10)
            shifted_tm = converter.convert(shifted_tm)
            lp.update_traffic_matrix(shifted_tm)
            t = lp.solve()
            if t > 0:
                if check_solution:
                    lp.check(feasibility_ratio=1e-2)
                solution_times.append(t)
                get_solution_confusion_matrix(lp, solver_params.FeasibilityTol, report=False, show=False, save_fig=False)
                print(f"Solved in {str(round(t, 4))} seconds. Final objective value: {str(round(lp.objective_value, 4))}")
                print(f"Actual utilization: {get_solution_maximum_utilization(lp.assignments, lp.graph)}")
                if save_solutions:
                    gurobi_sol_name = f'{topology_name}_{base_seed}_{tm_model}_shifted_{converter_seed}_{converter_model}_{i}'
                    if sol_name_postfix is not None:
                        gurobi_sol_name = f'{gurobi_sol_name}_{sol_name_postfix}'
                    solution_name = f'{gurobi_sol_name}.tesol'
                    shifted_solution = GurobiEdgeBasedMinimizeMaximumUtilityShiftedSolution(
                        seed=base_seed, topology_name=topology_name, capacity=base_solution.capacity, 
                        tm_model_name=tm_model, tm_model_params=base_solution.tm_model_params,
                        tm_converter_name=converter_model, tm_converter_params=converter_params,
                        converter_seed=converter_seed, iteration=i,
                        gurobi_sol_name=gurobi_sol_name, 
                        runtime=t
                    )
                    shifted_solution.dump(
                        model=lp._model,
                        name=solution_name
                    )
    print(
        f"MEDIAN: {str(round(np.median(solution_times), 4))} | "
        f"MIN: {str(round(np.min(solution_times), 4))} | "
        f"MAX: {str(round(np.max(solution_times), 4))}"
    )
    return solution_times


def test_warm_start(base_seed: int, topology_name: str, tm_model: str,
                    converter_model: str, converter_params: TrafficMatrixConverterParamsBase,
                    converter_seed: int, number_of_shifts: int, 
                    convergence_tol: float, feasibility_tol: float,
                    warm_start: bool = True, 
                    save_solutions: bool = True, 
                    sol_name_postfix: Optional[str] = None,
                    check_solution: bool = False):
    base_solution_name = f'{topology_name}_{base_seed}_{tm_model}.tesol'
    if warm_start:
        base_solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution = GurobiEdgeBasedMinimizeMaximumUtilitySolution.load(name=base_solution_name)
        graph, tm = base_solution.regenerate()
    else:
        # TODO: Add the option to refuse warm start ...
        raise NotImplementedError
    converter = get_traffic_converter(name=converter_model)(seed=converter_seed, params=converter_params)
    n = len(graph.edges)
    solver_params = UnregulatedADMMSolverParams(
        NumberOfEpochs=100,
        NumberOfNetworkUpdates=2,
        PGDIterations=25,
        Gamma=None,
        Eta=10/(n**2),
        Rho=10/(n**2),
        NumWorkers=8,
        UseVariableRho=False,
        BigTheta=1e-2,
        BigGamma=1e-4
    )
    solver_params.ConvTol = convergence_tol
    solver_params.FeasibilityTol = feasibility_tol
    solution_times: List[float] = []
    with contextlib.closing(UnregulatedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        # First, initiate to baseline solution
        lp.initialize_to(base_solution)
        lp.set_target(base_solution)
        # Now, start shifting the demands
        shifted_tm = tm
        for i in range(number_of_shifts):
            print("="*10 + f" ITERATION {i+1} / {number_of_shifts} " + "="*10)
            shifted_tm = converter.convert(shifted_tm)
            lp.update_traffic_matrix(shifted_tm)
            t = lp.solve()
            get_solution_confusion_matrix(lp, feasibility_ratio=1e-2, report=True, show=False, save_fig=False)
            print(f"Solved in {str(round(t, 4))} seconds. Final objective value: {str(round(lp.objective_value, 4))}")
            print(f"Actual utilization: {get_solution_maximum_utilization(lp.assignments, lp.graph)}")
            if t > 0:
                if check_solution:
                    lp.check(feasibility_ratio=1e-2)
                solution_times.append(t)
                get_solution_confusion_matrix(lp, feasibility_ratio=1e-2, report=False, show=False, save_fig=False)
                print(f"Solved in {str(round(t, 4))} seconds. Final objective value: {str(round(lp.objective_value, 4))}")
                print(f"Actual utilization: {get_solution_maximum_utilization(lp.assignments, lp.graph)}")
                # if save_solutions:
                #     gurobi_sol_name = f'{topology_name}_{base_seed}_{tm_model}_shifted_{converter_seed}_{converter_model}_{i}'
                #     if sol_name_postfix is not None:
                #         gurobi_sol_name = f'{gurobi_sol_name}_{sol_name_postfix}'
                #     solution_name = f'{gurobi_sol_name}.tesol'
                #     shifted_solution = GurobiEdgeBasedMinimizeMaximumUtilityShiftedSolution(
                #         seed=base_seed, topology_name=topology_name, capacity=base_solution.capacity, 
                #         tm_model_name=tm_model, tm_model_params=base_solution.tm_model_params,
                #         tm_converter_name=converter_model, tm_converter_params=converter_params,
                #         converter_seed=converter_seed, iteration=i,
                #         gurobi_sol_name=gurobi_sol_name, 
                #         runtime=t
                #     )
                #     shifted_solution.dump(
                #         model=lp._model,
                #         name=solution_name
                #     )
    print(
        f"MEDIAN: {str(round(np.median(solution_times), 4))} | "
        f"MIN: {str(round(np.min(solution_times), 4))} | "
        f"MAX: {str(round(np.max(solution_times), 4))}"
    )
    return solution_times


if __name__ == '__main__':
    # get_baseline_solution(12345, 'Interoute', 'Uniform', convergence_tol=1e-4, feasibility_tol=1e-6)
    # get_baseline_solution(12345, 'Interoute', 'Uniform', convergence_tol=1e-4, feasibility_tol=1e-6)
    DELTA = 0.08
    # get_baseline_shifted_solutions(
    #     base_seed=12345, topology_name='Interoute', tm_model='Uniform',
    #     converter_model='Uniform', 
    #     converter_params=get_traffic_converter_params('Uniform')(delta_min=-DELTA, delta_max=DELTA),
    #     converter_seed=6789, convergence_tol=1e-4, feasibility_tol=1e-6, number_of_shifts=30,
    #     sol_name_postfix=f'{DELTA}'
    # )
    test_warm_start(
        base_seed=12345, topology_name='Claranet', tm_model='Uniform',
        converter_model='Uniform', 
        converter_params=get_traffic_converter_params('Uniform')(delta_min=-DELTA, delta_max=DELTA),
        converter_seed=6789, convergence_tol=1e-4, feasibility_tol=1e-6, number_of_shifts=1,
        sol_name_postfix=f'{DELTA}'
    )
