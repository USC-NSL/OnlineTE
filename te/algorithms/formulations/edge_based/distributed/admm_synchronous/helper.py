import jsonargparse
import te.constants
import multiprocessing
import concurrent.futures
from dataclasses import dataclass
from typing import Tuple, List, Optional
from utils.logging import as_info, as_warning, log_section_title
from te.algorithms.base import *
from te.algorithms.communication.base import *
from te.algorithms.formulations.helper import *
from te.algorithms.sub_algorithms.mlu_backends.aggregate import *
from . import add_synch_solver_params_parser, parse_synch_solver_params
from .controller_backends.aggregate import *
from .controller import SynchADMMControllerNode
from .worker import SynchADMMWorkerNode

"""
The parameters that define the synchronous solver are:
- Algorithm Parameters
- MLU solver backend parameters
- Communication backend parameters
"""

@dataclass
class DistributedMLUSolverDescription:
    """
    Helper Dataclass for a distributed MLU solver.
    This completely describes the distributed MLU solver, and once combined
    with the problem description, can be used to spawn a full instance
    of the solver and see how it works.
    """
    AlgorithmParams: SolverParams
    MasterCLS: type[TELP]
    MasterBackendCLS: type[CommunicationBackendBase]
    MasterRPCParams: RPCParams
    MLUCLS: type[ControllerMLUSolver]
    MLUParams: SolverParams
    WorkerCLS: type[DistributedSolverNodeBase]
    WorkerBackendCLS: type[CommunicationBackendBase]
    WorkerRPCParamList: List[RPCParams]


def single_controller_topology_address_parser(parser: jsonargparse.ArgumentParser):
    addr_group = parser.add_argument_group(name='Node Addresses')
    addr_group.add_argument('--num-workers', type=int, required=True,
                            help='Number of worker nodes')
    addr_group.add_argument('--master-addr', type=Tuple[str, int],
                            help='Controller node address')
    addr_group.add_argument('--worker-addr', type=List[Tuple[str, int]],
                            help='List of worker node addresses')
    addr_group.add_argument('--local', action='store_true',
                            help='Assume everything must run locally')


def parse_single_controller_topology_address_parser(
    args: jsonargparse.Namespace
) -> Tuple[Tuple[str, int], List[Tuple[str, int]]]:
    n: int = args.num_workers
    master_addr: Optional[Tuple[str, int]] = args.master_addr
    worker_addr_list: Optional[List[Tuple[str, int]]] = args.worker_addr
    if args.local:
        master_addr = ("localhost", te.constants.DEFAULT_RPC_PORT)
        worker_addr_list = [("localhost", te.constants.DEFAULT_RPC_PORT + 1 + i) for i in range(n)]
    else:
        if master_addr is None:
            master_addr = ("controller", te.constants.DEFAULT_RPC_PORT)
        if worker_addr_list is None or len(worker_addr_list) == 0:
            worker_addr_list = [(f"n{i}", te.constants.DEFAULT_RPC_PORT) for i in range(n)]
    return master_addr, worker_addr_list


def distributed_synchronous_admm_parser() -> jsonargparse.ArgumentParser:
    parser = jsonargparse.ArgumentParser('Distributed Synchronous ADMM Solver')
    add_synch_solver_params_parser(parser)
    add_mlu_backend_parser(parser)
    single_controller_topology_address_parser(parser)
    add_communication_backend_params_parser(parser)
    return parser


def parse_distributed_synchronous_admm(args: jsonargparse.Namespace) -> DistributedMLUSolverDescription:
    solver_params = parse_synch_solver_params(args)
    mlu_backend_params, mlu_backend_cls = parse_mlu_backend_params(args)
    master_addr, worker_addr_list = parse_single_controller_topology_address_parser(args)
    comm_backend_description = parse_communication_backend_params(master_addr, worker_addr_list, args)
    return DistributedMLUSolverDescription(
        AlgorithmParams=solver_params,
        MasterCLS=SynchADMMControllerNode,
        MasterBackendCLS=comm_backend_description.ControllerBackendCLS,
        MasterRPCParams=comm_backend_description.ControllerBackendParams,
        MLUCLS=mlu_backend_cls,
        MLUParams=mlu_backend_params,
        WorkerCLS=SynchADMMWorkerNode,
        WorkerBackendCLS=comm_backend_description.WorkerBackendCLS,
        WorkerRPCParamList=comm_backend_description.WorkerBackendParams
    )


def spawn_distributed_synchronous_solver(
    problem: TEProblemDescription, 
    solver: DistributedMLUSolverDescription,
    is_local: bool = False
) -> Optional[TETracer]:
    def _spawn_distributed_synchronous_solver() -> Optional[TETracer]:
        print(as_info(
            f'Using master node communication backend `{solver.MasterBackendCLS.backend_name()}` with parameters:\n'+
            solver.MasterRPCParams.str_all()
        ))
        node_params = DistributedSolverNodeParams(
            CommunicationBackendCLS=solver.MasterBackendCLS,
            RPCParams_=solver.MasterRPCParams
        )
        return solve_te_and_check(
            problem, 
            solver.MasterCLS,
            solver.AlgorithmParams,
            node_params,
            solver.MLUCLS,
            solver.MLUParams
        )

    num_workers = len(solver.MasterRPCParams.Workers)
    # Number of workers that we want to spawn must match the number of workers
    # controlled by our controller node.
    assert len(solver.WorkerRPCParamList) == num_workers
    if is_local:
        print(as_warning(log_section_title("LOCAL EXPERIMENT")))
        print(as_info(
            f'A total of {num_workers} worker nodes will be spawned with addresses:\n'+
            PrettyAddressList(tuple([p.Peers[0] for p in solver.WorkerRPCParamList])).str_all()
        ))
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers, 
            mp_context=multiprocessing.get_context(method='spawn')
        ) as network_pool:
            # Worker nodes need no problem description and solver inputs, the
            # central controller will tell them all they need.
            for worker_rpc_params in solver.WorkerRPCParamList:
                network_pool.submit(
                    solver.WorkerCLS.spawn_and_run, 
                    DistributedSolverNodeParams(
                        CommunicationBackendCLS=solver.WorkerBackendCLS,
                        RPCParams_=worker_rpc_params
                    )
                )
            return _spawn_distributed_synchronous_solver()
    else:
        return _spawn_distributed_synchronous_solver()


__all__ = [
    'distributed_synchronous_admm_parser',
    'parse_distributed_synchronous_admm',
    'DistributedMLUSolverDescription',
    'spawn_distributed_synchronous_solver'
]