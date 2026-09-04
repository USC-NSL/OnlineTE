import jsonargparse
from typing import Tuple, List
from dataclasses import dataclass
from .asynchronous_grpc_backend import *
# from .udp_multicast_backend import *
from te.algorithms.communication import *
from te.algorithms.communication.grpc import *
from ..worker_backends.grpc_backend import SynchADMMgRPCWorkerBackend


def add_asyn_grpc_params(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(AsynchronousgRPCCoordinatorBackendParams, 'AsyngRPC', 
                               help='Asynchronous gRPC Communication Backend Parameters')

def parse_asyn_grpc_params(
    args: jsonargparse.Namespace,
    controller_addr: Tuple[str, int],
    worker_addr_list: List[Tuple[str, int]],
) -> AsynchronousgRPCCoordinatorBackendParams:
    args.AsyngRPC.Peers = tuple([controller_addr])
    args.AsyngRPC.Workers = tuple(worker_addr_list)
    return AsynchronousgRPCCoordinatorBackendParams.make_from_args(args.AsyngRPC)


def generate_asyn_grpc_worker_params(
    controller_params: AsynchronousgRPCCoordinatorBackendParams
) -> Tuple[List[gRPCWorkerBackendParams], type[SynchADMMgRPCWorkerBackend]]:
    return [gRPCWorkerBackendParams(
        PeerIndex=i, Peers=tuple([addr]),
        NumThreads=1
    ) for i, addr in enumerate(controller_params.Workers)], SynchADMMgRPCWorkerBackend


@dataclass
class SynchADMMCommunicationBackendDescription:
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
) -> SynchADMMCommunicationBackendDescription:
    if args.comm_backend == 'grpc-asyn':
        controller_params = parse_asyn_grpc_params(args, controller_addr, worker_addr_list)
        controller_cls = SynchADMMCoordinatorBackend
        worker_gen = generate_asyn_grpc_worker_params
    # elif args.comm_backend == 'mcast':
    #     controller_params = parse_mcast_params(args)
    #     controller_cls = MulticastControllerBackend
    #     worker_gen = generate_mcast_worker_params
    else:
        raise ValueError(f'Unknown communication backend name: {args.comm_backend}')
    
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