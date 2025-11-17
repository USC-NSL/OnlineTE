import argparse
import multiprocessing
import concurrent.futures
import te.constants
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .base import *
from te.algorithms.base import *
from utils.logging import as_info, as_warning, log_section_title
from te.algorithms.formulations.helper_base import solve_te_and_check, mlu_argparser, mlu_parse_args
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver


@dataclass
class DistributedMLUHelperParams:
    """
    Helper Dataclass for a distributed MLU solver.
    Adds attributes for MLU solver backend and its parameters to the usual
    `DistributedSolverNodeParams` class.
    """
    MasterCLS: type[TrafficEngineeringLP]
    MasterBackendCLS: type[CommunicationBackendBase]
    MasterRPCParams: RPCParams
    MLUCLS: type[ControllerMLUSolver]
    MLUParams: SolverParams


@dataclass
class SingleControllerMultiprocessMLUHelperParams(DistributedMLUHelperParams):
    """
    Helper Dataclass for a multi-process MLU solver with a single controller.
    In this setting, individual processes are spawned, where exactly one of
    them runs the cetnral controller node, and all other processes are worker
    nodes that interact with the controller.
    Worker nodes are distinguished using their RPC parameters.
    """
    WorkerCLS: type[DistributedSolverNodeBase]
    WorkerBackendCLS: type[CommunicationBackendBase]
    WorkerRPCParamList: List[RPCParams]


@dataclass
class HierarchicalMLUHelperParams:
    """
    Helper Dataclass for a hierarchical MLU solver, where domain
    controllers interact with a master node in a peer-to-peer fashion.
    Very similar to `DistributedMLUHelperParams`, but accepts a list
    of integers, that define how large each domain partition is.
    """
    MasterCLS: type[TrafficEngineeringLP]
    MasterBackendCLS: type[CommunicationBackendBase]
    MasterRPCParams: RPCParams
    MLUCLS: type[ControllerMLUSolver]
    MLUParams: SolverParams
    DomainPartitions: List[int]


@dataclass
class HierarchicalMultiprocessMLUHelperParams(HierarchicalMLUHelperParams):
    """
    Helper Dataclass for a multi-process, hierarchical MLU solver.
    Similar to `SingleControllerMultiprocessMLUHelperParams`, we accept extra
    parameters to spawn processes that implement domain worker nodes, but we also
    accept parameters for spawning domain controller nodes as well.
    """
    DomainCLS: type[DistributedSolverNodeBase]
    DomainBackendCLS: type[CommunicationBackendBase]
    DomainControllerRPCParamList: List[RPCParams]
    WorkerCLS: type[DistributedSolverNodeBase]
    WorkerBackendCLS: type[CommunicationBackendBase]
    WorkerRPCParamList: List[RPCParams]

    def __post_init__(self):
        # Number of partitions must agree with number of domains
        assert len(self.DomainPartitions) == len(self.DomainControllerRPCParamList)
        # Total number of nodes across partitions must agree with number of workers
        assert len(self.WorkerRPCParamList) == sum(self.DomainPartitions)


@dataclass
class PrettyAddressList(SolverParams):
    Addresses: Tuple[Tuple[str, int]]
    
    def __post_init__(self):
        self._left_column_share = 0.2


def distributed_mlu_helper(
    problem: TrafficEngineeringProblemDescription, 
    distributed_solver: DistributedMLUHelperParams,
    solver_params: SolverParams
):
    print(as_info(
        f'Using master node communication backend `{distributed_solver.MasterBackendCLS.backend_name()}` with parameters:\n'+
        distributed_solver.MasterRPCParams.str_all()
    ))
    node_params = DistributedSolverNodeParams(
        CommunicationBackendCLS=distributed_solver.MasterBackendCLS,
        RPCParams_=distributed_solver.MasterRPCParams
    )
    solve_te_and_check(
        problem, 
        distributed_solver.MasterCLS,
        solver_params,
        node_params,
        distributed_solver.MLUCLS,
        distributed_solver.MLUParams
    )


def single_controller_multiprocess_mlu_helper(
    problem: TrafficEngineeringProblemDescription, 
    distributed_solver: SingleControllerMultiprocessMLUHelperParams,
    solver_params: SolverParams
):
    num_workers = len(distributed_solver.MasterRPCParams.Workers)
    # Number of workers that we want to spawn must match the number of workers
    # controlled by our controller node.
    assert len(distributed_solver.WorkerRPCParamList) == num_workers
    print(as_warning(log_section_title("LOCAL EXPERIMENT")))
    print(as_info(
        f'A total of {num_workers} worker nodes will be spawned with addresses:\n'+
        PrettyAddressList(tuple([p.Peers[0] for p in distributed_solver.WorkerRPCParamList])).str_all()
    ))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        mp_context=multiprocessing.get_context(method='spawn')
    ) as network_pool:
        # Worker nodes need no problem description and solver inputs, the
        # central controller will tell them all they need.
        for worker_rpc_params in distributed_solver.WorkerRPCParamList:
            network_pool.submit(
                distributed_solver.WorkerCLS.spawn_and_run, 
                DistributedSolverNodeParams(
                    CommunicationBackendCLS=distributed_solver.WorkerBackendCLS,
                    RPCParams_=worker_rpc_params
                )
            )
        distributed_mlu_helper(problem, distributed_solver, solver_params)


def hierarchical_multiprocess_mlu_helper(
    problem: TrafficEngineeringProblemDescription, 
    distributed_solver: HierarchicalMultiprocessMLUHelperParams,
    solver_params: SolverParams
):
    print(as_warning(log_section_title("LOCAL EXPERIMENT")))
    NUM_DOMAIN_CONTROLLERS = len(distributed_solver.DomainControllerRPCParamList)
    NUM_DOMAIN_WORKERS = len(distributed_solver.WorkerRPCParamList)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_DOMAIN_WORKERS, 
        mp_context=multiprocessing.get_context(method='spawn')
    ) as domain_worker_pool:
        for worker_rpc_params in distributed_solver.WorkerRPCParamList:
            domain_worker_pool.submit(
                distributed_solver.WorkerCLS.spawn_and_run, 
                DistributedSolverNodeParams(
                    CommunicationBackendCLS=distributed_solver.WorkerBackendCLS,
                    RPCParams_=worker_rpc_params
                )
            )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=NUM_DOMAIN_CONTROLLERS,
            mp_context=multiprocessing.get_context(method='spawn')
        ) as domain_controller_pool:
            # Domain controller nodes must stand-by until the master is up.
            # As such, they will not get any extra problem description.
            for domain_controller_rpc_params in distributed_solver.DomainControllerRPCParamList:
                domain_controller_pool.submit(
                    distributed_solver.DomainCLS.spawn_and_run, 
                    DistributedSolverNodeParams(
                        communication_backend=distributed_solver.DomainBackendCLS,
                        rpc_params=domain_controller_rpc_params
                    ),
                    distributed_solver.MLUCLS, distributed_solver.MLUParams
                )
            distributed_mlu_helper(problem, distributed_solver, solver_params)


def distributed_mlu_argparser(prog_name: str) -> argparse.ArgumentParser:
    parser = mlu_argparser(prog_name)
    parser.add_argument('--num-workers', type=int, help='Number of workers to invoke', required=True)
    
    host_params_group = parser.add_argument_group('Remote Host Parameters')
    host_params_group.add_argument('--hosts', nargs='*', default=[], 
                                   help='List of remote hosts to connect to. If empty, defaults to `n0`, `n1`, `n2`, etc.')
    host_params_group.add_argument('--local', action='store_true', 
                                   help='Perform the test on local network. Overrides `hosts` option value')

    return parser


def distributed_mlu_parse_args(parser: argparse.ArgumentParser) -> Tuple[
    int, Tuple[Tuple[str, int], ...],
    TrafficEngineeringLPEvaluationParams, 
    Optional[TrafficEngineeringLPSolutionParams],
    Optional[TrafficEngineeringLPWarmStartParams],
    argparse.Namespace]:
    """
    Parse all the default arguments needed for the distributed MLU problem.

    Arguments
    ---------
    parser: `argparse.ArgumentParser`
        The argument parser (assumed produced with `distributed_mlu_argparser`)
    
    Returns
    -------
    num_workers: int
        Number of distributed workers involved in the problem
    addr_list: Tuple[Tuple[str, int], ...]
        A tuple (yes, it is a named a list because of reasons ...) of all worker
        node addresses. If empty, hostnames of the form `n0`, `n1`, etc. will be
        used, each bound to port `DEFAULT_RPC_PORT`.
        
        If run locally, the ports will be incremented by one for each host.
    eval_params: TrafficEngineeringLPEvaluationParams
        The TE problem evaluation parameters
    solution_params: Optional[TrafficEngineeringLPSolutionParams]
        Solution output parameters
    warmstart_params: Optional[TrafficEngineeringLPWarmStartParams]
        Warm-start parameters
    args: argparse.Namespace
        The namespace object of parsed arguments to further process
    """
    eval_params, solution_params, warm_start_params, args = mlu_parse_args(parser)
    num_workers = args.num_workers
    if args.local:
        addr_list = tuple([('localhost', te.constants.DEFAULT_RPC_PORT + i + 1) for i in range(num_workers)])
    else:
        if len(args.hosts) == 0:
            # Use `ni` as hosts ...
            addr_list = tuple([(f'n{i}', te.constants.DEFAULT_RPC_PORT) for i in range(num_workers)])
        else:
            assert len(args.hosts) == num_workers
            addr_list = tuple([(host, te.constants.DEFAULT_RPC_PORT) for host in args.hosts])
    
    return num_workers, addr_list, eval_params, solution_params, warm_start_params, args
