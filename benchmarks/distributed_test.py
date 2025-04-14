import time
import argparse
import contextlib
import concurrent.futures
from typing import Optional
from te.algorithms.formulations.aggregate import (
    NetworkWorkerNode, ControllerNode,
    DistributedADMMSolverParams, DistributedADMMWorkerRPCParams, 
    DistributedADMMControllerRPCParams
)
from te.algorithms.formulations.edge_based_distributed_admm.controller_backends import list_backends, get_backend_params
from te.algorithms.solution import (EdgeBasedMinimizeMaximumUtilitySolution, 
                                    EdgeBasedMinimizeMaximumUtilitySolutionParams, 
                                    default_solution_name)
from utils.logging import as_info, as_warning, as_success, log_section_title, log_subsection_title
from te.algorithms.utils import (get_solution_confusion_matrix, stringify_collected_stats, 
                                 str_round, get_solution_maximum_utilization)
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic
from te.traffic_models.converters import NCFlowTrafficMatrixConverter, NCFlowTrafficMatrixConverterParams


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

CONVERTER_SEED: Optional[int] = None
CONVERTER_PARAMS: Optional[NCFlowTrafficMatrixConverterParams] = None
CONVERTER_ITERS: Optional[int] = None
WARM_EPOCHS: Optional[int] = None


def local_distributed_admm_test(topology: str, seed: int, scale_factor: float = 10.0,
                                save_solution: bool = False, multicast: bool = False, **kwargs):
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

    if CONVERTER_PARAMS is not None:
        assert CONVERTER_ITERS is not None and CONVERTER_SEED is not None
        print(as_info(log_section_title("MLU PROBLEM (WITH WARM-START)")))
        converter = NCFlowTrafficMatrixConverter(CONVERTER_SEED, CONVERTER_PARAMS)
    else:
        print(as_info(log_section_title("MLU PROBLEM")))
        converter = None
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONTROLLER_RPC_PARAMS.NumWorkers) as network_pool:
        display_param = DistributedADMMWorkerRPCParams(
            IP=[addr[0] for addr in CONTROLLER_RPC_PARAMS.AddressList],
            Port=[addr[1] for addr in CONTROLLER_RPC_PARAMS.AddressList],
            WorkerID=[i for i in range(len(CONTROLLER_RPC_PARAMS.AddressList))],
            Multicast=multicast
        )
        print(as_info(f"Local Worker Backend Parameters:\n{display_param}"))
        for worker_id, worker_addr in enumerate(CONTROLLER_RPC_PARAMS.AddressList):
            network_pool.submit(
                NetworkWorkerNode.spawn_and_wait, 
                DistributedADMMWorkerRPCParams(
                    IP=worker_addr[0], Port=worker_addr[1], 
                    WorkerID=worker_id, Multicast=multicast
                )
            )
        
        with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, CONTROLLER_RPC_PARAMS)) as lp:
            print(as_info(f"Solving With: {lp.alg_name}"))
            print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS}"))
            print(as_info(f"Communication Backend `{CONTROLLER_RPC_PARAMS.Backend}` With Parameters:\n{CONTROLLER_RPC_PARAMS.stringify_up_to_level(1)}"))
            print(as_info("Waiting For Network Nodes ..."))
            while True:
                time.sleep(1)
                ready = lp.are_network_nodes_ready()
                if ready is True:
                    print(as_success("All Network Nodes Ready"))
                    break
                elif ready is None:
                    print(as_warning("Aborting"))
                    return
            
            if converter is not None:
                print(as_info(log_subsection_title("BASELINE SOLUTION")))
                
            lp.make_lp()
            t = lp.solve()
            if t > 0:
                lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
                get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
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
                for i in range(CONVERTER_ITERS):
                    print(as_info(log_subsection_title(f"WARM-START ITERATION {i}")))
                    converted_tm = converter.convert(tm)
                    lp.update_traffic_matrix(converted_tm)
                    t = lp.solve(params=args.warm_epochs)
                    if t > 0:
                        lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
                        get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
                        print(as_info(f"Solved in {str_round(t, 2)} seconds"))
                        print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
                        print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
            
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

    if CONVERTER_PARAMS is not None:
        assert CONVERTER_ITERS is not None and CONVERTER_SEED is not None
        print(as_info(log_section_title("MLU PROBLEM (WITH WARM-START)")))
        converter = NCFlowTrafficMatrixConverter(CONVERTER_SEED, CONVERTER_PARAMS)
    else:
        print(as_info(log_section_title("MLU PROBLEM")))
        converter = None

    with contextlib.closing(ControllerNode(graph, tm, SOLVER_PARAMS, CONTROLLER_RPC_PARAMS)) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Solving With Parameters:\n{SOLVER_PARAMS}"))
        print(as_info(f"Communication Backend `{CONTROLLER_RPC_PARAMS.Backend}` With Parameters:\n{CONTROLLER_RPC_PARAMS.stringify_up_to_level(1)}"))
        print(as_info("Waiting For Network Nodes ..."))
        while True:
            time.sleep(1)
            ready = lp.are_network_nodes_ready()
            if ready is True:
                print(as_success("All Network Nodes Ready"))
                break
            elif ready is None:
                print(as_warning("Aborting"))
                return

        if converter is not None:
            print(as_info(log_subsection_title("BASELINE SOLUTION")))

        lp.make_lp()
        t = lp.solve()
        if t > 0:
            lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
            get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
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
            for i in range(CONVERTER_ITERS):
                print(as_info(log_subsection_title(f"WARM-START ITERATION {i}")))
                converted_tm = converter.convert(tm)
                lp.update_traffic_matrix(converted_tm)
                t = lp.solve(params=args.warm_epochs)
                if t > 0:
                    lp.check(feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO)
                    get_solution_confusion_matrix(lp, feasibility_tol=FEASIBILITY_TOL, feasibility_ratio=FEASIBILITY_RATIO, **kwargs)
                    print(as_info(f"Solved in {str_round(t, 2)} seconds"))
                    print(as_info(f"Final objective value: {str_round(lp.objective_value, 4)}"))
                    print(as_info(f"Actual utilization: {str_round(get_solution_maximum_utilization(lp.assignments, lp.graph), 4)}"))
        
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
    
    rpc_params_group = parser.add_argument_group('General Communication Backend Parameters')
    rpc_params_group.add_argument('--backend-name', choices=list_backends(), default='gRPC-asynchronous',
                                  help='Communication backend name to use')
    
    sync_rcp_params_group = parser.add_argument_group('Synchronous gRPC Backend Parameters')
    sync_rcp_params_group.add_argument('--num-threads', type=int, default=None,
                                       help='Number of threads in backend thread pool. Defaults to number of workers.')
    
    async_rpc_params_group = parser.add_argument_group('Asynchronous gRPC Backend Parameters')
    async_rpc_params_group.add_argument('--timeout', type=float, default=5.0,
                                        help='Future `get` timeout for `asyncio`')
    
    multicast_backend_params_group = parser.add_argument_group('UDP Multicast Backend Parameters')
    multicast_backend_params_group.add_argument('--group', default='224.0.0.10',
                                                help='Multicast group address for all worker nodes')
    multicast_backend_params_group.add_argument('--mport', type=int, default=12000,
                                                help='UDP port to listen for responses')
    multicast_backend_params_group.add_argument('--ttl', type=int, default=2,
                                                help='Multicast packet TTL (should be at least 2)')
    
    host_params_group = parser.add_argument_group('Remote Host Parameters')
    host_params_group.add_argument('--hosts', nargs='*', default=[], help='List of remote hosts to connect to')

    warm_start_params_group = parser.add_argument_group('Remote Host Parameters')
    warm_start_params_group.add_argument('--converter-seed', type=int, help='RNG seed for TM converter')
    warm_start_params_group.add_argument('--warm-iters', type=int, help='Number of warm-start iterations')
    warm_start_params_group.add_argument('--warm-epochs', type=int, help='Number of epochs per warm-start iteration')
    warm_start_params_group.add_argument('--scale-mean', type=float, default=0.1, 
                                         help='Relative scaling of TM mean between conversions')
    warm_start_params_group.add_argument('--scale-std', type=float, default=0.2, 
                                         help='Relative scaling of TM standard deviation between conversions')
    
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

    def set_backend_specific_params(rpc_params: DistributedADMMControllerRPCParams):
        if args.backend_name == 'gRPC-synchronous':
            rpc_params.NumThreads = args.num_threads if args.num_threads is not None else args.num_workers
        elif args.backend_name == 'gRPC-asynchronous':
            rpc_params.Timeout = args.timeout
        elif args.backend_name == 'multicast':
            rpc_params.Timeout = args.timeout
            rpc_params.ScatterAddress = args.group
            rpc_params.ScatterPort = args.mport
            rpc_params.TTL = args.ttl
        else:
            raise ValueError
    
    CONVERTER_ITERS = args.warm_iters
    if CONVERTER_ITERS is not None:
        CONVERTER_PARAMS = NCFlowTrafficMatrixConverterParams(args.scale_mean, args.scale_std)
        CONVERTER_SEED = args.converter_seed
        WARM_EPOCHS = args.warm_epochs

    if args.local:
        CONTROLLER_RPC_PARAMS = get_backend_params(args.backend_name)(
            NumWorkers=args.num_workers,
            AddressList=tuple([(LOCAL_HOST, BASE_PORT + worker_id) for worker_id in range(args.num_workers)])
        )
        set_backend_specific_params(CONTROLLER_RPC_PARAMS)
        local_distributed_admm_test(args.topo, args.seed, args.scale_factor, 
                                    save_solution=args.save_sol, 
                                    multicast=(args.backend_name == 'multicast'),
                                    report=args.report_unsat)
    else:
        if len(args.hosts) > 0:
            assert len(args.hosts) == args.num_workers
            hosts = [(host, BASE_PORT + worker_id) for worker_id, host in enumerate(args.hosts)]
        else:
            hosts = [(f'n{worker_id}', BASE_PORT + worker_id) 
                        for worker_id in range(args.num_workers)]
        CONTROLLER_RPC_PARAMS = get_backend_params(args.backend_name)(
            NumWorkers=args.num_workers,
            AddressList=tuple(hosts)
        )
        set_backend_specific_params(CONTROLLER_RPC_PARAMS)
        remote_distributed_admm_test(args.topo, args.seed, args.scale_factor, 
                                     save_solution=args.save_sol, report=args.report_unsat)
    # local_distributed_admm_test(SMALL_TOPOLOGY, RNG_SEED, save_solution=True)
    # local_distributed_admm_test(SMALL_MEDIUM_TOPOLOGY, RNG_SEED)
    # local_distributed_admm_test(MEDIUM_TOPOLOGY, RNG_SEED)
    # remote_distributed_admm_test([f'n{i}.infra.v0.unregulatedadmm.distte' for i in range(SOLVER_PARAMS.NumWorkers)], 
    #                               SMALL_TOPOLOGY, RNG_SEED, save_solution=True)
