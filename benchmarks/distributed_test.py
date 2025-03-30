import time
import argparse
import contextlib
import concurrent.futures
from typing import List, Tuple
from te.algorithms.formulations.edge_based_distributed_admm.worker import NetworkWorkerNode
from te.algorithms.formulations.edge_based_distributed_admm.controller import ControllerNode
from te.algorithms.formulations.edge_based_distributed_admm import (DistributedADMMSolverParams,
                                                                    DistributedADMMWorkerRPCParams,
                                                                    DistributedADMMControllerRPCParams)
from te.algorithms.solution import (EdgeBasedMinimizeMaximumUtilitySolution, 
                                    EdgeBasedMinimizeMaximumUtilitySolutionParams, 
                                    default_solution_name)
from utils.logging import as_info, as_success
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

SOLVER_PARAMS = DistributedADMMSolverParams(
    NumberOfEpochs=150,
    NumberOfNetworkUpdates=2,
    PGDIterations=2,
    Gamma=0.9,
    Eta=8,
    Rho=1,
    Kappa=0.1,
    Seed=RNG_SEED,
    BigGamma=1e-7,
    Precision='double',
    NumWorkers=2
)


def show_addrs(addrs: List[Tuple[str, int]]):
    print(as_info("Worker Nodes:"))
    for host, port in addrs:
        print(as_info("\t{:^32} : {:^10}".format(host, str(port))))


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

    print(as_info("="*60))
    print(as_info("="*23 + " MLU PROBLEM " + "="*24))
    print(as_info("="*60))

    worker_addrs = tuple([(LOCAL_HOST, BASE_PORT + worker_id) for worker_id in range(SOLVER_PARAMS.NumWorkers)])
    show_addrs(worker_addrs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=SOLVER_PARAMS.NumWorkers) as network_pool:
        for worker_id, worker_addr in enumerate(worker_addrs):
            network_pool.submit(NetworkWorkerNode.spawn_and_wait, 
                                worker_id, DistributedADMMWorkerRPCParams(ip=worker_addr[0], port=worker_addr[1]))
        
        with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, 
                                            DistributedADMMControllerRPCParams(
                                                tuple(worker_addrs),
                                                num_threads=min(SOLVER_PARAMS.NumWorkers, 8)
                                            ))) as lp:
            print(as_info(f"Solving With: {lp.alg_name}"))
            print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS}"))
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


def remote_distributed_admm_test(hosts: List[str], topology: str, seed: int, scale_factor: float = 10.0,
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

    print(as_info("="*60))
    print(as_info("="*23 + " MLU PROBLEM " + "="*24))
    print(as_info("="*60))

    worker_addrs = tuple([(hosts[worker_id], BASE_PORT + worker_id) 
                          for worker_id in range(min(SOLVER_PARAMS.NumWorkers, len(hosts)))])
    show_addrs(worker_addrs)
    with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, 
                                        DistributedADMMControllerRPCParams(
                                            tuple(worker_addrs),
                                            num_threads=min(SOLVER_PARAMS.NumWorkers, 8)
                                        ))) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS}"))
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
    solver_params_group.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    solver_params_group.add_argument('--updates', type=int, default=2, help='Number of consecutive network updates')
    solver_params_group.add_argument('--pgd-iters', type=int, default=2, help='Number of PGD iterations at each step')
    solver_params_group.add_argument('--pgd-step', type=float, default=0.9, help='PGD step size')
    solver_params_group.add_argument('--pgd-reduction', type=float, default=0.1, help='PGD step size reduction factor')
    solver_params_group.add_argument('--admm-outer', type=float, default=1.0, help='Outer ADMM step size')
    solver_params_group.add_argument('--admm-inner', type=float, default=8.0, help='Inner ADMM step size')
    solver_params_group.add_argument('--controller-opt-tol', type=float, default=1e-7, 
                                     help='Barrier method convergence tolerance')
    solver_params_group.add_argument('--precision', choices=['half', 'single', 'double'], default='single',
                                     help='Floating point operation precision')

    runtime_params_group = parser.add_argument_group('Runtime Parameters')
    runtime_params_group.add_argument('--save-sol', action='store_true', help='Save the final solution')
    runtime_params_group.add_argument('--scale-factor', type=float, default=10.0, 
                                      help='Link capacity scaling factor.')
    runtime_params_group.add_argument('--report-unsat', action='store_true', 
                                      help='Report unsatisfied commodity assignments.')
    
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
        NumWorkers=args.num_workers
    )

    if args.local:
        local_distributed_admm_test(args.topo, args.seed, args.scale_factor, 
                                    save_solution=args.save_sol, report=args.report_unsat)
    else:
        remote_distributed_admm_test([f'n{i}.infra.v0.unregulatedadmm.distte' for i in range(SOLVER_PARAMS.NumWorkers)], 
                                     args.topo, args.seed, args.scale_factor, 
                                     save_solution=args.save_sol, report=args.report_unsat)
    # local_distributed_admm_test(SMALL_TOPOLOGY, RNG_SEED, save_solution=True)
    # local_distributed_admm_test(SMALL_MEDIUM_TOPOLOGY, RNG_SEED)
    # local_distributed_admm_test(MEDIUM_TOPOLOGY, RNG_SEED)
    # remote_distributed_admm_test([f'n{i}.infra.v0.unregulatedadmm.distte' for i in range(SOLVER_PARAMS.NumWorkers)], 
    #                               SMALL_TOPOLOGY, RNG_SEED, save_solution=True)
