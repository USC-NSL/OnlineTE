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
    DistributedEdgeBasedADMMLP, DistributedADMMSolverParams,
    SemiDistributedEdgeBasedADMMLP
)
from te.algorithms.utils import report_commodity_assignments, check_centralized_flow_conservation
from topologies.utils import (
    load_zoo_topology, get_capacity_lower_bound,
    set_edge_capacity_to, make_graph_from_dict
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
            expected = lp.commodity_list
            result = lp.get_solution_commodity_list()
            unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
            report_commodity_assignments(expected, result, unsatisfied)
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
        # if t > 0:
        expected = lp.commodity_list
        result = lp.get_solution_commodity_list()
        # unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
        report_commodity_assignments(expected, result, 0)
        print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")
        plt.plot(lp.objective_trace)
        plt.show()


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
        if t >= 0:
            lp.check()
            expected = lp.commodity_list
            result = lp.get_solution_commodity_list()
            unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
            report_commodity_assignments(expected, result, unsatisfied)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")
            print(f"Final utilization value: {lp._utility.X}")


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
        expected = lp.commodity_list
        result = lp.get_solution_commodity_list()
        # unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
        report_commodity_assignments(expected, result, 0.0)
        print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")
        print(f"Final utilization value: {lp._utility.X}")
        plt.plot(lp.objective_trace)
        plt.show()


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
        expected = lp.commodity_list
        result = lp.get_solution_commodity_list()
        # unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
        report_commodity_assignments(expected, result, 0.0)
        print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")
        plt.plot(lp.objective_trace)
        plt.show()


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
        NumberOfNetworkUpdates=5
    )
    with contextlib.closing(DistributedEdgeBasedADMMLP(graph, tm, solver_params)) as lp:
    # with contextlib.closing(SemiDistributedEdgeBasedADMMLP(graph, tm, solver_params)) as lp:
        lp.make_lp()
        t = lp.solve()
        expected = lp.commodity_list
        result = lp.get_solution_commodity_list()
        # unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
        report_commodity_assignments(expected, result, 0.0)
        print(f"Solved in {t} seconds.")
        print(f"Final utilization value: {lp._utility.X}")
        plt.plot(lp.objective_trace)
        plt.show()


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
            expected = lp.commodity_list
            result = lp.get_solution_commodity_list()
            unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
            report_commodity_assignments(expected, result, unsatisfied)
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
            expected = lp.commodity_list
            result = lp.get_solution_commodity_list()
            unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
            report_commodity_assignments(expected, result, unsatisfied, verbose=False)
            print(f"Solved in {t} seconds. Final objective value: {lp.objective_value}")


def zoo_test_4():
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
            expected = lp.commodity_list
            result = lp.get_solution_commodity_list()
            unsatisfied = lp.get_ratio_of_unsatisfied_demands(solver_params)
            report_commodity_assignments(expected, result, unsatisfied)
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
    zoo_test_1_admm()
    # zoo_test_1_dist_parallel()
