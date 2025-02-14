import contextlib
import numpy as np
import matplotlib.pyplot as plt
from te.traffic_models import get_traffic_model
from te.traffic_models.models import UniformTrafficMatrixParams, CustomTrafficMatrix
from te.algorithms.base import GurobiSolverParams
from te.algorithms.formulations.edge_based_centralized import CentralizedEdgeBasedLP
from te.algorithms.formulations.edge_based_distributed import DistributedEdgeBasedLP, DistributedSolverParams
from te.algorithms.formulations.edge_based_distributed_parallel import (
    DistributedParallelEdgeBasedLP, DistributedParallelSolverParams,
    DistributedParallelSolverNodeParams, DistributedParallelSolverControllerParams
)
from te.algorithms.formulations.edge_based_distributed_admm import (
    DistributedEdgeBasedADMMLP, DistributedADMMSolverParams
)
from te.algorithms.formulations.edge_based_regularized_admm import RegularizedADMMLP, RegularizedADMMSolverParams
from te.algorithms.formulations.edge_based_unregulated_admm import UnregulatedADMMLP, UnregulatedADMMSolverParams
from te.algorithms.formulations.mp_edge_based_regularized_admm import (
    MultiProcessorRegularizedADMMLP, MultiProcessesorRegularizedADMMSolverParams,
    RegularizedADMMRPCParams
)
from te.algorithms.utils import check_centralized_flow_conservation, get_solution_confusion_matrix
from topologies.utils import (
    load_zoo_topology, get_capacity_lower_bound,
    set_edge_capacity_to, make_graph_from_dict,
)


def toy_test_1():
    graph_dict = {
        (0, 1): 10,
        (1, 2): 5,
        (1, 3): 5
    }

    graph = make_graph_from_dict(graph_n=4, graph_dict=graph_dict)
    tm = CustomTrafficMatrix(
        np.array([
            [0.0, 0.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
    )

    solver_params = GurobiSolverParams()
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def toy_test_2():
    graph_dict = {
        (0, 1): 10,
        (1, 2): 5,
        (1, 3): 5
    }

    graph = make_graph_from_dict(graph_n=4, graph_dict=graph_dict)
    tm = CustomTrafficMatrix(
        np.array([
            [0.0, 0.0, 3.0, 4.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
    )

    solver_params = DistributedSolverParams()
    with contextlib.closing(DistributedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def zoo_test_1():
    graph = load_zoo_topology('Twaren')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*3)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = GurobiSolverParams()
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def zoo_test_1_dist():
    graph = load_zoo_topology('Twaren')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*3)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = DistributedSolverParams(
        NumberOfEpochs=2000,
        Alpha=1e-1, Beta=1e-1
    )
    with contextlib.closing(DistributedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def zoo_test_1_dist_parallel():
    graph = load_zoo_topology('Twaren')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*3)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = DistributedParallelSolverParams(NumberOfEpochs=1000)
    controller_params = DistributedParallelSolverControllerParams()
    node_params = DistributedParallelSolverNodeParams()
    with contextlib.closing(DistributedParallelEdgeBasedLP(graph, tm, solver_params, controller_params, node_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def zoo_test_1_admm():
    graph = load_zoo_topology('Twaren')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*3)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = DistributedADMMSolverParams(
        NumberOfEpochs=20,
        NumberOfNetworkUpdates=3
    )
    with contextlib.closing(DistributedEdgeBasedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def zoo_test_2():
    graph = load_zoo_topology('Sanet')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*5)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = GurobiSolverParams()
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def zoo_test_3():
    # graph = load_zoo_topology('UsSignal')
    graph = load_zoo_topology('Internode')
    # graph = load_zoo_topology('Uninett2011')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*8)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = GurobiSolverParams()
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def centralized_test_small():
    graph = load_zoo_topology('Claranet')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = GurobiSolverParams()
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {str(round(t, 4))} seconds. Final objective value: {str(round(lp.objective_value, 4))}")


def centralized_test_medium():
    graph = load_zoo_topology('Interoute')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = GurobiSolverParams()
    with contextlib.closing(CentralizedEdgeBasedLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def admm_test_medium():
    graph = load_zoo_topology('Interoute')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = DistributedADMMSolverParams(
        NumberOfEpochs=20,
        NumberOfNetworkUpdates=3
    )
    with contextlib.closing(DistributedEdgeBasedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def regularized_admm_test_small():
    graph = load_zoo_topology('Claranet')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = RegularizedADMMSolverParams(
        NumberOfEpochs=5,
        NumberOfNetworkUpdates=2,
        Epsilon=1e-4,
        Eta=1e-3,
        Rho=1e-3
    )
    with contextlib.closing(RegularizedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def mp_regularized_admm_test_small():
    graph = load_zoo_topology('Claranet')
    # graph = load_zoo_topology('Interoute')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = MultiProcessesorRegularizedADMMSolverParams(
        NumberOfNodeProcesses=5,
        NumberOfEpochs=50,
        NumberOfNetworkUpdates=10,
        Epsilon=1e-5,
        Eta=1e-3,
        Rho=1e-3
    )
    rpc_params = RegularizedADMMRPCParams(number_of_controller_workers=5)
    with contextlib.closing(MultiProcessorRegularizedADMMLP(graph, tm, solver_params, rpc_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, feasibility_ratio=1e-2)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def regularized_admm_test_medium():
    graph = load_zoo_topology('Interoute')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = RegularizedADMMSolverParams(
        NumberOfEpochs=20,
        NumberOfNetworkUpdates=10,
        Epsilon=1e-3,
        Rho=1e-3
    )
    with contextlib.closing(RegularizedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def unregulated_admm_test_small():
    graph = load_zoo_topology('Claranet')
    # graph = load_zoo_topology('Interoute')
    
    tm_params = UniformTrafficMatrixParams(
        n = len(graph.nodes), min = 0.0, max = 1.0
    )
    tm = get_traffic_model('Uniform')(seed=12345, params=tm_params)
    
    c_min = get_capacity_lower_bound(graph, tm)
    set_edge_capacity_to(graph, c_min*10)
    print(f"Capacity lower bound is: {c_min}")

    solver_params = UnregulatedADMMSolverParams(
        NumberOfEpochs=40,
        NumberOfNetworkUpdates=2,
        PGDIterations=1000,
        Gamma=1e-1,
        Eta=1e-4,
        Rho=1e-4,
        NumWorkers=8,
        UseVariableRho=True
    )
    with contextlib.closing(UnregulatedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_ratio=1e-2)
            get_solution_confusion_matrix(lp, solver_params.FeasibilityTol)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


if __name__ == '__main__':
    # toy_test_1()
    # zoo_test_1()
    # zoo_test_2()
    
    # zoo_test_3()

    # toy_test_1()
    # toy_test_2()
    # zoo_test_1()
    # zoo_test_1_dist()
    # zoo_test_1_admm()
    # zoo_test_1_dist_parallel()
    # centralized_test_small()
    # centralized_test_medium()
    # admm_test_medium()
    # regularized_admm_test_small()
    # regularized_admm_test_medium()
    # mp_regularized_admm_test_small()
    unregulated_admm_test_small()
