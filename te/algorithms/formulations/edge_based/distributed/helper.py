import argparse
import multiprocessing
import concurrent.futures
import te.constants
from dataclasses import dataclass
from typing import Optional, List, Tuple
from .base import *
from te.algorithms.base import *
from utils.logging import as_warning, log_section_title
from te.algorithms.formulations.helper_base import solve_te_and_check, mlu_argparser, mlu_parse_args
from te.algorithms.sub_algorithms.mlu_backends.base import ControllerMLUSolver


@dataclass
class DistributedMLUHelperParams(DistributedSolverNodeParams):
    """
    Helper Dataclass for a distributed MLU solver.
    Adds attributes for MLU solver backend and its parameters to the usual
    `DistributedSolverNodeParams` class.
    """
    MasterCLS: type[TrafficEngineeringLP]
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
class HierarchicalMLUHelperParams(DistributedSolverNodeParams):
    """
    Helper Dataclass for a hierarchical MLU solver, where domain
    controllers interact with a master node in a peer-to-peer fashion.
    Very similar to `DistributedMLUHelperParams`, but accepts a list
    of integers, that define how large each domain partition is.
    """
    MasterCLS: type[TrafficEngineeringLP]
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


def distributed_mlu_helper(params: DistributedMLUHelperParams):
    solve_te_and_check(
        params.ProblemDescription, 
        params.MasterCLS, 
        params.SolverParams_,
        params.MLUCLS,
        params.MLUParams
    )


def single_controller_multiprocess_mlu_helper(params: SingleControllerMultiprocessMLUHelperParams):
    num_workers = len(params.RPCParams_.Workers)
    # Number of workers that we want to spawn must match the number of workers
    # controlled by our controller node.
    assert len(params.WorkerRPCParamList) == num_workers
    print(as_warning(log_section_title("LOCAL EXPERIMENT")))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        mp_context=multiprocessing.get_context(method='spawn')
    ) as network_pool:
        # Worker nodes need no problem description and solver inputs, the
        # central controller will tell them all they need.
        for worker_rpc_params in params.WorkerRPCParamList:
            network_pool.submit(
                params.WorkerCLS.spawn_and_run, 
                DistributedSolverNodeParams(
                    CommunicationBackendCLS=params.WorkerBackendCLS,
                    RPCParams_=worker_rpc_params
                )
            )
        distributed_mlu_helper(params)


def hierarchical_multiprocess_mlu_helper(params: HierarchicalMultiprocessMLUHelperParams):
    print(as_warning(log_section_title("LOCAL EXPERIMENT")))
    NUM_DOMAIN_CONTROLLERS = len(params.DomainControllerRPCParamList)
    NUM_DOMAIN_WORKERS = len(params.WorkerRPCParamList)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_DOMAIN_WORKERS, 
        mp_context=multiprocessing.get_context(method='spawn')
    ) as domain_worker_pool:
        for worker_rpc_params in params.WorkerRPCParamList:
            domain_worker_pool.submit(
                params.WorkerCLS.spawn_and_run, 
                DistributedSolverNodeParams(
                    CommunicationBackendCLS=params.WorkerBackendCLS,
                    RPCParams_=worker_rpc_params
                )
            )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=NUM_DOMAIN_CONTROLLERS,
            mp_context=multiprocessing.get_context(method='spawn')
        ) as domain_controller_pool:
            # Domain controller nodes must stand-by until the master is up.
            # As such, they will not get any extra problem description.
            for domain_controller_rpc_params in params.DomainControllerRPCParamList:
                domain_controller_pool.submit(
                    params.DomainCLS.spawn_and_run, 
                    DistributedSolverNodeParams(
                        mlu_backend=params.MLUCLS, mlu_params=params.MLUParams,
                        communication_backend=params.DomainBackendCLS,
                        rpc_params=domain_controller_rpc_params
                    )
                )
            distributed_mlu_helper(params)


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
        addr_list = tuple([('localhost', te.constants.DEFAULT_RPC_PORT + i) for i in range(num_workers)])
    else:
        if len(args.hosts) == 0:
            # Use `ni` as hosts ...
            addr_list = tuple([(f'n{i}', te.constants.DEFAULT_RPC_PORT) for i in range(num_workers)])
        else:
            assert len(args.hosts) == num_workers
            addr_list = tuple([(host, te.constants.DEFAULT_RPC_PORT) for host in args.hosts])
    
    return num_workers, addr_list, eval_params, solution_params, warm_start_params, args
