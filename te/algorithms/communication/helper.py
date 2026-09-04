import jsonargparse
import te.constants
from dataclasses import dataclass
from typing import Tuple, List, Optional
from te.algorithms.base import *
from te.algorithms.communication.base import *
from te.algorithms.formulations.helper import *
from te.algorithms.sub_algorithms.mlu_backends.aggregate import *
from .coordinator_backend import CoordinatorBackendBase
from .worker_backend import WorkerBackendBase
from .grpc import *


@dataclass
class OnlineTECommunicationBackendDescription:
    """Just a data structure for keeping relevant info about backends"""
    ControllerBackendCLS: type[CoordinatorBackendBase]
    ControllerBackendParams: RPCParams
    WorkerBackendCLS: type[WorkerBackendBase]
    WorkerBackendParams: List[RPCParams]


def add_communication_backend_params_parser(parser: jsonargparse.ArgumentParser):
    parser.add_argument('--comm_backend', choices=['grpc-asyn', 'mcast'], default='grpc-asyn')
    add_asyn_grpc_params(parser)
    # add_mcast_params(parser)


def parse_communication_backend_params(
    controller_addr: Tuple[str, int],
    worker_addr_list: List[Tuple[str, int]],
    args: jsonargparse.Namespace
) -> OnlineTECommunicationBackendDescription:
    # Get relevant backend
    if args.comm_backend == 'grpc-asyn':
        controller_params = parse_asyn_grpc_params(args, controller_addr, worker_addr_list)
        controller_cls = AsynchronousgRPCCoordinatorBackend
        worker_gen = generate_asyn_grpc_worker_params
        worker_cls = gRPCWorkerBackend
    # elif args.comm_backend == 'mcast':
    #     controller_params = parse_mcast_params(args)
    #     controller_cls = MulticastControllerBackend
    #     worker_gen = generate_mcast_worker_params
    else:
        raise ValueError(f'Unknown communication backend name: {args.comm_backend}')
    # Generate worker RPC parameters
    worker_params = worker_gen(controller_params)
    # Pack everything neatly ...
    return OnlineTECommunicationBackendDescription(
        ControllerBackendCLS=controller_cls,
        ControllerBackendParams=controller_params,
        WorkerBackendCLS=worker_cls,
        WorkerBackendParams=worker_params
    )


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


__all__ = [
    'OnlineTECommunicationBackendDescription',
    'add_communication_backend_params_parser',
    'parse_communication_backend_params',
    'single_controller_topology_address_parser',
    'parse_single_controller_topology_address_parser'
]