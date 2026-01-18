import jsonargparse
from typing import Tuple, List
from dataclasses import dataclass
from ..base import SynchADMMControllerBackendBase, SynchADMMWorkerBackendBase
from ...base import RPCParams
from .asynchronous_grpc_backend import *
from .udp_multicast_backend import *


@dataclass
class SynchADMMCommunicationBackendDescription:
    ControllerBackendCLS: type[SynchADMMControllerBackendBase]
    ControllerBackendParams: RPCParams
    WorkerBackendCLS: type[SynchADMMWorkerBackendBase]
    WorkerBackendParams: List[RPCParams]


def add_communication_backend_params_parser(parser: jsonargparse.ArgumentParser):
    parser.add_argument('--comm_backend', choices=['grpc-syn', 'grpc-asyn', 'mcast'], default='grpc-asyn')
    add_asyn_grpc_params(parser)
    add_mcast_params(parser)


def parse_communication_backend_params(
    controller_addr: Tuple[str, int],
    worker_addr_list: List[Tuple[str, int]],
    args: jsonargparse.Namespace
) -> SynchADMMCommunicationBackendDescription:
    if args.comm_backend == 'grpc-asyn':
        controller_params = parse_asyn_grpc_params(args)
        controller_cls = AsynchronousgRPCControllerBackend
        worker_gen = generate_asyn_grpc_worker_params
    elif args.comm_backend == 'mcast':
        controller_params = parse_mcast_params(args)
        controller_cls = MulticastControllerBackend
        worker_gen = generate_mcast_worker_params
    else:
        raise ValueError(f'Unknown communication backend name: {args.comm_backend}')
    
    controller_params.Peers = tuple([controller_addr])
    controller_params.Workers = tuple(worker_addr_list)
    worker_params, worker_cls = worker_gen(controller_params)

    return SynchADMMCommunicationBackendDescription(
        ControllerBackendCLS=controller_cls,
        ControllerBackendParams=controller_params,
        WorkerBackendCLS=worker_cls,
        WorkerBackendParams=worker_params
    )


__all__ = [
    'SynchADMMCommunicationBackendDescription',
    'add_communication_backend_params_parser', 
    'parse_communication_backend_params'
]