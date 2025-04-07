import time
import argparse
import contextlib
import concurrent.futures
from te.algorithms.formulations.aggregate import (
    NetworkWorkerNode, ControllerNode,
    DistributedADMMSolverParams, DistributedADMMWorkerRPCParams, 
    DistributedADMMControllerRPCParams
)
from te.algorithms.formulations.edge_based_distributed_admm.controller_backends import list_backends
from te.algorithms.solution import (EdgeBasedMinimizeMaximumUtilitySolution, 
                                    EdgeBasedMinimizeMaximumUtilitySolutionParams, 
                                    default_solution_name)
from utils.logging import as_info, as_success, log_section_title
from te.algorithms.utils import (get_solution_confusion_matrix, stringify_collected_stats, 
                                 str_round, get_solution_maximum_utilization)
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic


RNG_SEED = 12345

FEASIBILITY_TOL = None
FEASIBILITY_RATIO = 5e-2

SMALL_TOPOLOGY = 'Claranet'
SMALL_MEDIUM_TOPOLOGY = 'Forthnet'
MEDIUM_TOPOLOGY = 'Interoute'
HUGE_TOPOLOGY = 'Kdl'


LOCAL_HOST = "localhost"
BASE_PORT = 13000


SOLVER_PARAMS: DistributedADMMSolverParams = DistributedADMMSolverParams()
CONTROLLER_RPC_PARAMS: DistributedADMMControllerRPCParams = DistributedADMMControllerRPCParams()


def local_distributed_admm_test(topology: str, seed: int, scale_factor: float = 10.0,
                                save_solution: bool = False, **kwargs):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed, scale_factor=scale_factor)
    print(f"Network link capacity is: {str(round(c, 2))}")

    solution_params = None
    if save_solution:
        solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
            seed=seed, topology_name=topology, capacity=c,
            tm_model_name=tm.type(), tm_model_params=tm.params,
            path=None, sol_name=default_solution_name(
                topology_name=topology, rng_seed=seed, tm_type=tm.type(),
                postfix='ours'
            )
        )

    print(as_info(log_section_title("MLU PROBLEM")))
    
    # show_addrs(worker_addrs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONTROLLER_RPC_PARAMS.NumWorkers) as network_pool:
        for worker_id, worker_addr in enumerate(CONTROLLER_RPC_PARAMS.AddressList):
            network_pool.submit(NetworkWorkerNode.spawn_and_wait, 
                                worker_id, DistributedADMMWorkerRPCParams(IP=worker_addr[0], Port=worker_addr[1]))
        
        with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, CONTROLLER_RPC_PARAMS)) as lp:
            print(as_info(f"Solving With: {lp.alg_name}"))
            print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS}"))
            print(as_info(f"Communication Backend Parameters:\n{CONTROLLER_RPC_PARAMS}"))
            print(as_info("Waiting For Network Nodes ..."))
            while True:
                time.sleep(1)
                if lp.are_network_nodes_ready():
                    break
            print(as_success("All Network Nodes Ready"))

            lp.make_lp()
            t = lp.solve()
            if t > 0:
                lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
                get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
                print(as_info(f"Solved in {str_round(t, 2)} seconds"))
                print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
                print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
            if solution_params:
                solution = EdgeBasedMinimizeMaximumUtilitySolution(params=solution_params)
                lp.add_solution_elements(solution)
                solution.dump_elements()
                solution.dump(name=solution_params.sol_name)
            stats = stringify_collected_stats()
            if stats is not None:
                print(as_info(stats))


def remote_distributed_admm_test(topology: str, seed: int, scale_factor: float = 10.0,
                                 save_solution: bool = False, **kwargs):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed, scale_factor=scale_factor)
    print(f"Network link capacity is: {str(round(c, 2))}")

    solution_params = None
    if save_solution:
        solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
            seed=seed, topology_name=topology, capacity=c,
            tm_model_name=tm.type(), tm_model_params=tm.params,
            path=None, sol_name=default_solution_name(
                topology_name=topology, rng_seed=seed, tm_type=tm.type(),
                postfix='ours'
            )
        )

    print(as_info(log_section_title("MLU PROBLEM")))

    with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, CONTROLLER_RPC_PARAMS)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS}"))
        print(as_info(f"Communication Backend Parameters:\n{CONTROLLER_RPC_PARAMS}"))
        print(as_info("Waiting For Network Nodes ..."))
        while True:
            time.sleep(1)
            if lp.are_network_nodes_ready():
                break
        print(as_success("All Network Nodes Ready"))

        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
            get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
            print(as_info(f"Solved in {str_round(t, 2)} seconds"))
            print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
            print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
        if solution_params:
            solution = EdgeBasedMinimizeMaximumUtilitySolution(params=solution_params)
            lp.add_solution_elements(solution)
            solution.dump_elements()
            solution.dump(name=solution_params.sol_name)
        stats = stringify_collected_stats()
        if stats is not None:
            print(as_info(stats))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simple distributed test')
    
    parser.add_argument('topo', help='Topology name')
    parser.add_argument('seed', type=int, help='RNG seed')
    parser.add_argument('num_workers', type=int, help='Number of workers to invoke')
    parser.add_argument('--local', action='store_true', help='Perform the test on local network')
    
    solver_params_group = parser.add_argument_group('Solver Parameters', description='ADMM solver parameters')
    solver_params_group.add_argument('--epochs', type=int, default=SOLVER_PARAMS.NumberOfEpochs, 
                                     help='Number of epochs')
    solver_params_group.add_argument('--updates', type=int, default=SOLVER_PARAMS.NumberOfNetworkUpdates, 
                                     help='Number of consecutive network updates')
    solver_params_group.add_argument('--pgd-iters', type=int, default=SOLVER_PARAMS.PGDIterations, 
                                     help='Number of PGD iterations at each step')
    solver_params_group.add_argument('--pgd-step', type=float, default=SOLVER_PARAMS.Gamma, 
                                     help='PGD step size')
    solver_params_group.add_argument('--pgd-reduction', type=float, default=SOLVER_PARAMS.Kappa, 
                                     help='PGD step size reduction factor')
    solver_params_group.add_argument('--admm-outer', type=float, default=SOLVER_PARAMS.Rho, 
                                     help='Outer ADMM step size')
    solver_params_group.add_argument('--admm-inner', type=float, default=SOLVER_PARAMS.Eta, 
                                     help='Inner ADMM step size')
    solver_params_group.add_argument('--controller-opt-tol', type=float, default=SOLVER_PARAMS.BigGamma, 
                                     help='Barrier method convergence tolerance')
    solver_params_group.add_argument('--precision', choices=['half', 'single', 'double'], default=SOLVER_PARAMS.Precision,
                                     help='Floating point operation precision')

    runtime_params_group = parser.add_argument_group('Runtime Parameters')
    runtime_params_group.add_argument('--save-sol', action='store_true', help='Save the final solution')
    runtime_params_group.add_argument('--scale-factor', type=float, default=10.0, 
                                      help='Link capacity scaling factor.')
    runtime_params_group.add_argument('--report-unsat', action='store_true', 
                                      help='Report unsatisfied commodity assignments.')
    
    rpc_params_group = parser.add_argument_group('Communication Backend Parameters')
    rpc_params_group.add_argument('--backend-name', choices=list_backends(), default=CONTROLLER_RPC_PARAMS.Backends,
                                  help='Communication backend name to use')
    
    args = parser.parse_args()

    SOLVER_PARAMS = DistributedADMMSolverParams(
        NumberOfEpochs=args.epochs,
        NumberOfNetworkUpdates=args.updates,
        PGDIterations=args.pgd_iters,
        Gamma=args.pgd_step,
        Eta=args.admm_inner,
        Rho=args.admm_outer,
        Kappa=args.pgd_reduction,
        Seed=args.seed,
        BigGamma=args.controller_opt_tol,
        Precision=args.precision,
    )

    if args.local:
        CONTROLLER_RPC_PARAMS = DistributedADMMControllerRPCParams(
            NumWorkers=args.num_workers,
            AddressList=tuple([(LOCAL_HOST, BASE_PORT + worker_id) for worker_id in range(args.num_workers)]),
            NumThreads=args.num_workers,
            Backends=args.backend_name
        )
        local_distributed_admm_test(args.topo, args.seed, args.scale_factor, 
                                    save_solution=args.save_sol, report=args.report_unsat)
    else:
        CONTROLLER_RPC_PARAMS = DistributedADMMControllerRPCParams(
            NumWorkers=args.num_workers,
            AddressList=tuple([(f'n{worker_id}.infra.v0.unregulatedadmm.distte', BASE_PORT + worker_id) 
                               for worker_id in range(args.num_workers)]),
            # AddressList=tuple([(LOCAL_HOST, BASE_PORT + worker_id) 
            #                    for worker_id in range(args.num_workers)]),
            NumThreads=args.num_workers,
            Backends=args.backend_name
        )
        remote_distributed_admm_test(args.topo, args.seed, args.scale_factor, 
                                     save_solution=args.save_sol, report=args.report_unsat)
    # local_distributed_admm_test(SMALL_TOPOLOGY, RNG_SEED, save_solution=True)
    # local_distributed_admm_test(SMALL_MEDIUM_TOPOLOGY, RNG_SEED)
    # local_distributed_admm_test(MEDIUM_TOPOLOGY, RNG_SEED)
    # remote_distributed_admm_test([f'n{i}.infra.v0.unregulatedadmm.distte' for i in range(SOLVER_PARAMS.NumWorkers)], 
    #                               SMALL_TOPOLOGY, RNG_SEED, save_solution=True)
