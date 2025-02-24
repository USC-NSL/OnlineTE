import time
import gurobipy
import contextlib
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from te.traffic_models import get_traffic_model
from te.algorithms.solution import GurobiEdgeBasedMinimizeMaximumUtilitySolution
from te.traffic_models.models import UniformTrafficMatrixParams
from te.traffic_models.converters import UniformConverter
from te.algorithms.base import GurobiSolverParams
from te.algorithms.formulations.edge_based_centralized import CentralizedEdgeBasedLP
from te.algorithms.formulations.edge_based_unregulated_admm import UnregulatedADMMLP, UnregulatedADMMSolverParams
from te.algorithms.utils import get_solution_confusion_matrix, get_solution_maximum_utilization
from topologies.utils import load_zoo_topology, get_capacity_lower_bound, set_edge_capacity_to


def get_baseline_solutions(base_seed: int, topology_name: str, tm_model: str, delta_min: float, 
                           delta_max: float, converter_seed: int, number_of_shifts: int):
    """This function generates the optimal basis for the base TM and all shifts"""
    gurobi_sol_name = f'{topology_name}_{base_seed}_{tm_model}'
    solution_name = f'{gurobi_sol_name}.tesol'
    graph = load_zoo_topology(topology_name)
    tm_params = UniformTrafficMatrixParams(n = len(graph.nodes), min = 0.0, max = 1.0)
    tm = get_traffic_model(tm_model)(seed=base_seed, params=tm_params)
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)

    solver_params = GurobiSolverParams()
    solver_params.Method = gurobipy.GRB.METHOD_DUAL
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


def centralized_test_small() -> List[float]:
    SEED = 12345
    CONVERTER_SEED = 6789
    TOPOLOGY_NAME = 'Forthnet'
    DELTA_MIN = -0.3
    DELTA_MAX = 0.3
    TM_MODEL = 'Uniform'
    SOLUTION_NAME = f'{TOPOLOGY_NAME}_{SEED}_{TM_MODEL}.tesol'

    solution: GurobiEdgeBasedMinimizeMaximumUtilitySolution = GurobiEdgeBasedMinimizeMaximumUtilitySolution.load(name=SOLUTION_NAME)
    graph, tm = solution.regenerate()

    converter = UniformConverter(seed=CONVERTER_SEED, delta_min=DELTA_MIN, delta_max=DELTA_MAX)

    solver_params = GurobiSolverParams()
    solver_params.Method = gurobipy.GRB.METHOD_DUAL
    solution_times: List[float] = []
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        lp.initialize_to(solution)
        # First, get the base solution
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol, report=False, show=False, save_fig=False)
            print(f"Solved in {str(round(t, 4))} seconds. Final objective value: {str(round(lp.objective_value, 4))}")
            print(f"Actual utilization: {get_solution_maximum_utilization(lp.assignments, lp.graph)}")
            # solution = GurobiEdgeBasedMinimizeMaximumUtilitySolution(
            #     seed=SEED, topology_name=TOPOLOGY_NAME, capacity=c_min*10, tm_model_name=TM_MODEL,
            #     tm_model_params=TM_PARAMS, gurobi_sol_name=f'{TOPOLOGY_NAME}_{SEED}_{TM_MODEL}'
            # )
            # solution.dump(model=lp._model, name=SOLUTION_NAME)
    #     # Now, start shifting the demands
    #     shifted_tm = tm
    #     NUM_TESTS = 10
    #     for i in range(NUM_TESTS):
    #         print("="*10 + f" ITERATION {i+1} / {NUM_TESTS} " + "="*10)
    #         shifted_tm = converter.convert(shifted_tm)
    #         lp.update_traffic_matrix(shifted_tm)
    #         t = lp.solve()
    #         if t > 0:
    #             lp.check(feasibility_ratio=1e-2)
    #             solution_times.append(t)
    #             get_solution_confusion_matrix(lp, solver_params.FeasibilityTol, report=False, show=False, save_fig=False)
    #             print(f"Solved in {str(round(t, 4))} seconds. Final objective value: {str(round(lp.objective_value, 4))}")
    #             print(f"Actual utilization: {get_solution_maximum_utilization(lp.assignments, lp.graph)}")
    #         time.sleep(1)
    # print(
    #     f"MEDIAN: {str(round(np.median(solution_times), 4))} | "
    #     f"MIN: {str(round(np.min(solution_times), 4))} | "
    #     f"MAX: {str(round(np.max(solution_times), 4))}"
    # )
    # return solution_times


def unregulated_admm_test_small():
    graph = load_zoo_topology('Claranet')
    # graph = load_zoo_topology('Forthnet')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = UnregulatedADMMSolverParams(
        NumberOfEpochs=100,
        NumberOfNetworkUpdates=1,
        PGDIterations=1000,
        Gamma=1e-1,
        Eta=1e-4,
        Rho=1e-5,
        NumWorkers=8,
        UseVariableRho=True
    )
    with contextlib.closing(UnregulatedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, feasibility_ratio=1e-2, report=True)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")
            print(f"Actual utilization: {get_solution_maximum_utilization(lp.assignments, lp.graph)}")


if __name__ == '__main__':
    # get_baseline_solutions(12345, 'Forthnet', 'Uniform', -0.3, 0.3, 6789, 10)
    centralized_test_small()
    # unregulated_admm_test_small()
