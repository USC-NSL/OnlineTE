import time
import argparse
import contextlib
import concurrent.futures
from typing import List
from te.algorithms.formulations.edge_based_asynchronous_admm import AsynchronousADMMSolverParams
from te.algorithms.formulations.edge_based_asynchronous_admm.controller_backends.udp_multicast_backend import MulticastControllerBackendParams
from te.algorithms.formulations.edge_based_asynchronous_admm.worker_backends.udp_multicast_backend import MulticastWorkerBackendParams
from te.algorithms.formulations.edge_based_asynchronous_admm.controller import ControllerNode
from te.algorithms.formulations.edge_based_asynchronous_admm.worker import NetworkWorkerNode
from te.algorithms.solution import (EdgeBasedMinimizeMaximumUtilitySolution, 
                                    EdgeBasedMinimizeMaximumUtilitySolutionParams, 
                                    default_solution_name)
from utils.logging import as_info, as_warning, as_success, log_section_title
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


SOLVER_PARAMS: AsynchronousADMMSolverParams = AsynchronousADMMSolverParams()
CONTROLLER_RPC_PARAMS: MulticastControllerBackendParams = MulticastControllerBackendParams()
WORKER_RPC_PARAMS: List[MulticastWorkerBackendParams] = []


def local_asynchronous_admm_test(topology: str, seed: int, scale_factor: float = 10.0,
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
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONTROLLER_RPC_PARAMS.NumWorkers) as network_pool:
        display_param = MulticastWorkerBackendParams(
            IP=[addr[0] for addr in CONTROLLER_RPC_PARAMS.AddressList],
            Port=[addr[1] for addr in CONTROLLER_RPC_PARAMS.AddressList],
            WorkerID=[i for i in range(len(CONTROLLER_RPC_PARAMS.AddressList))],
            ScatterAddress=CONTROLLER_RPC_PARAMS.ScatterAddress,
            ScatterPort=CONTROLLER_RPC_PARAMS.ScatterPort,
            ControllerHost=CONTROLLER_RPC_PARAMS.Hostname,
            ControllerPort=CONTROLLER_RPC_PARAMS.ListenPort,
            TTL=CONTROLLER_RPC_PARAMS.TTL
        )
        print(as_info(f"Local Worker Backend Parameters:\n{display_param}"))
        for worker_param in WORKER_RPC_PARAMS:
            network_pool.submit(NetworkWorkerNode.spawn_and_wait, worker_param)
        
        with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, CONTROLLER_RPC_PARAMS)) as lp:
            lp.initialize()
            print(as_info(f"Solving With: {lp.alg_name}"))
            print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS.stringify_up_to_level(1)}"))
            print(as_info(f"Controller backend parameters:\n{CONTROLLER_RPC_PARAMS.stringify_up_to_level(1)}"))
            print(as_info("Waiting For Network Nodes ..."))
            while True:
                time.sleep(1)
                ready = lp.are_network_nodes_ready()
                if not lp.is_active:
                    print(as_warning("Aborting"))
                    return
                if ready is True:
                    print(as_success("All Network Nodes Ready"))
                    break
                
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


# def remote_distributed_admm_test(topology: str, seed: int, scale_factor: float = 10.0,
#                                  save_solution: bool = False, **kwargs):
#     c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed, scale_factor=scale_factor)
#     print(f"Network link capacity is: {str(round(c, 2))}")

#     solution_params = None
#     if save_solution:
#         solution_params = EdgeBasedMinimizeMaximumUtilitySolutionParams(
#             seed=seed, topology_name=topology, capacity=c,
#             tm_model_name=tm.type(), tm_model_params=tm.params,
#             path=None, sol_name=default_solution_name(
#                 topology_name=topology, rng_seed=seed, tm_type=tm.type(),
#                 postfix='ours'
#             )
#         )

#     if CONVERTER_PARAMS is not None:
#         assert CONVERTER_ITERS is not None and CONVERTER_SEED is not None
#         print(as_info(log_section_title("MLU PROBLEM (WITH WARM-START)")))
#         converter = NCFlowTrafficMatrixConverter(CONVERTER_SEED, CONVERTER_PARAMS)
#     else:
#         print(as_info(log_section_title("MLU PROBLEM")))
#         converter = None

#     with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, CONTROLLER_RPC_PARAMS)) as lp:
#         print(as_info(f"Solving With: {lp.alg_name}"))
#         print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS}"))
#         print(as_info(f"Communication Backend `{CONTROLLER_RPC_PARAMS.Backend}` With Parameters:\n{CONTROLLER_RPC_PARAMS.stringify_up_to_level(1)}"))
#         print(as_info("Waiting For Network Nodes ..."))
#         while True:
#             time.sleep(1)
#             ready = lp.are_network_nodes_ready()
#             if ready is True:
#                 print(as_success("All Network Nodes Ready"))
#                 break
#             elif ready is None:
#                 print(as_warning("Aborting"))
#                 return

#         if converter is not None:
#             print(as_info(log_subsection_title("BASELINE SOLUTION")))

#         lp.make_lp()
#         t = lp.solve()
#         if t > 0:
#             lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
#             get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
#             print(as_info(f"Solved in {str_round(t, 2)} seconds"))
#             print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
#             print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
#         if solution_params:
#             if converter is None:
#                 solution = EdgeBasedMinimizeMaximumUtilitySolution(params=solution_params)
#                 lp.add_solution_elements(solution)
#                 solution.dump_elements()
#                 solution.dump(name=solution_params.sol_name)
#             else:
#                 print(as_warning('Will not save solution for warm-tests for now ... (takes too much space!)'))

#         if converter is not None:
#             converted_tm = tm
#             for i in range(CONVERTER_ITERS):
#                 print(as_info(log_subsection_title(f"WARM-START ITERATION {i}")))
#                 converted_tm = converter.convert(tm)
#                 lp.update_traffic_matrix(converted_tm)
#                 t = lp.solve(params=args.warm_epochs)
#                 if t > 0:
#                     lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
#                     get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
#                     print(as_info(f"Solved in {str_round(t, 2)} seconds"))
#                     print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
#                     print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
        
#         stats = stringify_collected_stats()
#         if stats is not None:
#             print(as_info(stats))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simple asynchronous test')
    
    parser.add_argument('topo', help='Topology name')
    parser.add_argument('seed', type=int, help='RNG seed')
    parser.add_argument('num_workers', type=int, help='Number of workers to invoke')
    parser.add_argument('--local', action='store_true', help='Perform the test on local network')
    
    solver_params_group = parser.add_argument_group('Solver Parameters', description='ADMM solver parameters')
    solver_params_group.add_argument('--epochs', type=int, default=SOLVER_PARAMS.NumberOfEpochs, 
                                     help='Number of epochs')
    solver_params_group.add_argument('--updates', type=int, default=SOLVER_PARAMS.NumberOfNetworkUpdates, 
                                     help='Number of consecutive network updates')
    solver_params_group.add_argument('--local-updates', type=int, default=SOLVER_PARAMS.NumberOfLocalUpdates,
                                     help='Number of local updates per network update')
    solver_params_group.add_argument('--qp-method', default=SOLVER_PARAMS.QPMethod, 
                                     help='Solver method for the QP at each node', choices=['ADMM', 'PGD'])
    solver_params_group.add_argument('--qp-iters', type=int, default=SOLVER_PARAMS.QPIterations, 
                                     help='Number of iterations to solve the node QP for each local update')
    solver_params_group.add_argument('--qp-step', type=float, default=SOLVER_PARAMS.Gamma, 
                                     help='Step size for solving the node QP')
    solver_params_group.add_argument('--pgd-reduction', type=float, default=SOLVER_PARAMS.Kappa, 
                                     help='PGD step size reduction factor (specific to the `PGD` solver for the node QP)')
    solver_params_group.add_argument('--admm-outer', type=float, default=SOLVER_PARAMS.Rho, 
                                     help='Outer ADMM step size')
    solver_params_group.add_argument('--admm-inner', type=float, default=SOLVER_PARAMS.Eta, 
                                     help='Inner ADMM step size')
    solver_params_group.add_argument('--controller-opt-tol', type=float, default=SOLVER_PARAMS.BigGamma, 
                                     help='Barrier method convergence tolerance')
    solver_params_group.add_argument('--precision', choices=['half', 'single', 'double'], default=SOLVER_PARAMS.Precision,
                                     help='Floating point operation precision')
    solver_params_group.add_argument('--upsilon', type=int, default=SOLVER_PARAMS.Upsilon,
                                     help='Minimum number of updateable switches for the controller loop')
    solver_params_group.add_argument('--sigma', type=int, default=SOLVER_PARAMS.Sigma,
                                     help='Maximum number of local iterations on a switch without controller input')
    solver_params_group.add_argument('--batch-size', type=int, default=SOLVER_PARAMS.WorkerBatchSize,
                                     help='maximum number of consecutive controller updates to consume at once on a switch')
    
    # TODO: Param for ADMM convergence tolerance

    runtime_params_group = parser.add_argument_group('Runtime Parameters')
    runtime_params_group.add_argument('--save-sol', action='store_true', help='Save the final solution')
    runtime_params_group.add_argument('--scale-factor', type=float, default=10.0, 
                                      help='Link capacity scaling factor.')
    runtime_params_group.add_argument('--report-unsat', action='store_true', 
                                      help='Report unsatisfied commodity assignments.')
    
    multicast_backend_params_group = parser.add_argument_group('Backend Parameters')
    multicast_backend_params_group.add_argument('--group', default='224.0.0.10',
                                                help='Multicast group address for all worker nodes')
    multicast_backend_params_group.add_argument('--mport', type=int, default=12000,
                                                help='Multicast listen port on the switches')
    multicast_backend_params_group.add_argument('--cport', type=int, default=11000,
                                                help='UDP port to listen for switch updates on the controller')
    multicast_backend_params_group.add_argument('--gport-base', type=int, default=13000,
                                                help='Base gRPC listening port on switches for RPCs that use large messages')
    multicast_backend_params_group.add_argument('--ttl', type=int, default=2,
                                                help='Multicast packet TTL (should be at least 2)')
    
    # TODO: Param for queue timeout, socket timeout and remote hosts
    
    args = parser.parse_args()

    SOLVER_PARAMS = AsynchronousADMMSolverParams(
        NumberOfEpochs=args.epochs,
        NumberOfNetworkUpdates=args.updates,
        NumberOfLocalUpdates=args.local_updates,
        QPIterations=args.qp_iters,
        QPMethod=args.qp_method,
        Gamma=args.qp_step,
        Eta=args.admm_inner,
        Rho=args.admm_outer,
        Kappa=args.pgd_reduction,
        Seed=args.seed,
        BigGamma=args.controller_opt_tol,
        Precision=args.precision,
        Upsilon=args.upsilon,
        Sigma=args.sigma,
        WorkerBatchSize=args.batch_size
    )

    if args.local:
        CONTROLLER_RPC_PARAMS = MulticastControllerBackendParams(
            AddressList=tuple([(LOCAL_HOST, BASE_PORT + worker_id) for worker_id in range(args.num_workers)]),
            NumWorkers=args.num_workers, ScatterAddress=args.group, ScatterPort=args.mport,
            Hostname=LOCAL_HOST, ListenPort=args.cport, TTL=args.ttl
        )
        WORKER_RPC_PARAMS = [
            MulticastWorkerBackendParams(
                IP=LOCAL_HOST, Port=args.gport_base + worker_id, WorkerID=worker_id,
                ScatterAddress=args.group, ControllerHost=LOCAL_HOST,
                ControllerPort=args.cport, TTL=args.ttl, 
                ScatterPort=args.mport
            ) for worker_id in range(args.num_workers)
        ]
        local_asynchronous_admm_test(
            topology=args.topo, seed=args.seed, scale_factor=args.scale_factor,
            save_solution=args.save_sol, report=args.report_unsat
        )
    else:
        raise NotImplementedError('Not yet implement remote case!')
