import time
import contextlib
import concurrent.futures
from te.algorithms.formulations.edge_based_distributed_admm.worker import NetworkWorkerNode
from te.algorithms.formulations.edge_based_distributed_admm.controller import ControllerNode
from te.algorithms.formulations.edge_based_distributed_admm import (DistributedADMMSolverParams,
                                                                    DistributedADMMWorkerRPCParams,
                                                                    DistributedADMMControllerRPCParams)
from te.algorithms.utils import as_info, get_solution_confusion_matrix, stringify_collected_stats, str_round, get_solution_maximum_utilization
from topologies.utils import get_uniform_tm_problem_with_capacity_heuristic

import warnings
warnings.filterwarnings("error")

RNG_SEED = 12345

FEASIBILITY_TOL = None
FEASIBILITY_RATIO = 1e-2

SMALL_TOPOLOGY = 'Claranet'
SMALL_MEDIUM_TOPOLOGY = 'Forthnet'
MEDIUM_TOPOLOGY = 'Interoute'
HUGE_TOPOLOGY = 'Kdl'


HOST = "localhost"
BASE_PORT = 13000


def distributed_admm_test(topology: str, seed: int, scale_factor: float = 10.0, **kwargs):
    c, graph, tm = get_uniform_tm_problem_with_capacity_heuristic(topology, seed, scale_factor=scale_factor)
    print(f"Network link capacity is: {str(round(c, 2))}")

    solver_params = DistributedADMMSolverParams(
        NumberOfEpochs=2,
        NumberOfNetworkUpdates=2,
        PGDIterations=2,
        Gamma=1,
        Eta=8,
        Rho=1,
        Kappa=0.1,
        Seed=RNG_SEED,
        NumWorkers=1
    )

    print(as_info("="*60))
    print(as_info("="*23 + " MLU PROBLEM " + "="*24))
    print(as_info("="*60))

    worker_addrs = tuple([(HOST, BASE_PORT + worker_id) for worker_id in range(solver_params.NumWorkers)])
    # with concurrent.futures.ProcessPoolExecutor(max_workers=solver_params.NumWorkers) as network_pool:
    #     for worker_id, worker_addr in enumerate(worker_addrs):
    #         network_pool.submit(NetworkWorkerNode.spawn_and_wait, 
    #                             worker_id, solver_params, DistributedADMMWorkerRPCParams(ip=worker_addr[0], port=worker_addr[1]))
        
    with contextlib.closing(ControllerNode(graph, tm, solver_params, 
                                        DistributedADMMControllerRPCParams(tuple(worker_addrs)))) as lp:
        print(as_info(f"Solving With: {lp.alg_name}"))
        print(as_info(f"Solving With Parameters:\n{solver_params}"))
        # print(as_info("Waiting For Network Nodes ..."))
        # while True:
        #     time.sleep(1)
        #     if lp.are_network_nodes_ready():
        #         break
        # print(as_info("All Network Nodes Ready"))

        lp.make_lp()
        t = lp.solve()
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
    distributed_admm_test(SMALL_TOPOLOGY, RNG_SEED)
