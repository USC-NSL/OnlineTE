import jsonargparse
from typing import List, Tuple
from .asynchronous_coordinator_backend import (
    AsynchronousgRPCCoordinatorBackendParams,
    AsynchronousgRPCCoordinatorBackend
)
from .worker_backend import gRPCWorkerBackendParams, gRPCWorkerBackend


def add_asyn_grpc_params(parser: jsonargparse.ArgumentParser):
    parser.add_class_arguments(
        AsynchronousgRPCCoordinatorBackendParams, 'AsyngRPC', 
        help='Asynchronous gRPC Communication Backend Parameters'
    )


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
) -> List[gRPCWorkerBackendParams]:
    return [gRPCWorkerBackendParams(
        PeerIndex=i, Peers=tuple([addr]),
        NumThreads=1
    ) for i, addr in enumerate(controller_params.Workers)]


__all__ = [
    'AsynchronousgRPCCoordinatorBackendParams',
    'AsynchronousgRPCCoordinatorBackend',
    'gRPCWorkerBackendParams', 'gRPCWorkerBackend',
    'add_asyn_grpc_params', 'parse_asyn_grpc_params',
    'generate_asyn_grpc_worker_params'
]