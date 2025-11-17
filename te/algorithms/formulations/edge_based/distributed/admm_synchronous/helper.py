import argparse
from typing import Optional, Tuple, List
from ..base import RPCParams
from .base import SynchADMMControllerBackendBase, SynchADMMWorkerBackendBase
from .controller_backends.udp_multicast_backend import MulticastControllerBackendParams, MulticastControllerBackend
from .controller_backends.synchronous_grpc_backend import SynchronousgRPCControllerBackendParams, SynchronousgRPCControllerBackend
from .controller_backends.asynchronous_grpc_backend import AsynchronousgRPCControllerBackendParams, AsynchronousgRPCControllerBackend
from .worker_backends.grpc_backend import gRPCWorkerBackendParams, gRPCWorkerBackend
from .worker_backends.udp_multicast_backend import MulticastWorkerBackendParams, MulticastWorkerBackend


def add_admm_synch_communication_backend_subparser(parser: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    backend_subparser = parser.add_subparsers(dest='communication_backend', help='Communication backend to use', required=True)

    MULTICAST_PARAMS = MulticastControllerBackendParams()
    multicast_params_parser = backend_subparser.add_parser('multicast', 
                                                           help='Options for the UDP Multicast communication backend')
    multicast_params_parser.add_argument('--group', default=MULTICAST_PARAMS.ScatterAddress, 
                                         help='Multicast group address for all worker nodes')
    multicast_params_parser.add_argument('--host', default=MULTICAST_PARAMS.HostName, 
                                         help='Multicast source hostname')
    multicast_params_parser.add_argument('--ttl', type=int, default=MULTICAST_PARAMS.TTL,
                                         help='Multicast packet TTL (should be at least 2)')
    multicast_params_parser.add_argument('--port', type=int, default=MULTICAST_PARAMS.ScatterPort, 
                                         help='UDP port to listen for responses')
    multicast_params_parser.add_argument('--timeout', type=float, default=MULTICAST_PARAMS.Timeout, 
                                         help='Timeout on `gather` operations')
    multicast_params_parser.add_argument('--copy', type=int, default=MULTICAST_PARAMS.UpdateCopyCount,
                                         help='Number of message copies sent during scattering for redundancy')

    GRPC_SYNCH_PARAMS = SynchronousgRPCControllerBackendParams()
    grpc_synch_params_parser = backend_subparser.add_parser('grpc-synch', 
                                                            help='Options for the synchronous gRPC communication backend')
    grpc_synch_params_parser.add_argument('--threads', default=GRPC_SYNCH_PARAMS.NumThreads, 
                                          help='Number of threads for broadcasting updates')

    GRPC_ASYNCH_PARAMS = AsynchronousgRPCControllerBackendParams()
    grpc_asynch_params_parser = backend_subparser.add_parser('grpc-asynch', 
                                                             help='Options for the asynchronous gRPC communication backend')
    grpc_asynch_params_parser.add_argument('--timeout', type=float, default=GRPC_ASYNCH_PARAMS.Timeout, 
                                           help='Timeout on future `get` operations')
    
    return [multicast_params_parser, grpc_synch_params_parser, grpc_asynch_params_parser]


def parse_add_admm_synch_communication_backend_params(
    num_workers: int, addr_list: Tuple[Tuple[str, int]],
    args: Optional[argparse.Namespace] = None, 
    parser: Optional[argparse.ArgumentParser] = None
) -> Tuple[
    RPCParams, 
    type[SynchADMMControllerBackendBase], 
    List[RPCParams], 
    type[SynchADMMWorkerBackendBase], 
    argparse.Namespace
]:
    assert len(addr_list) == num_workers
    
    if args is None:
        assert parser is not None
        args = parser.parse_args()
    
    if args.communication_backend == 'multicast':
        MULTICAST_CONTROLLER_PARAMS = MulticastControllerBackendParams()
        MULTICAST_CONTROLLER_PARAMS.Workers = addr_list
        MULTICAST_CONTROLLER_PARAMS.ScatterAddress = args.group
        MULTICAST_CONTROLLER_PARAMS.HostName = args.host
        MULTICAST_CONTROLLER_PARAMS.TTL = args.ttl
        MULTICAST_CONTROLLER_PARAMS.ScatterPort = args.port
        MULTICAST_CONTROLLER_PARAMS.Timeout = args.timeout
        MULTICAST_CONTROLLER_PARAMS.UpdateCopyCount = args.copy
        MULTICAST_WORKER_PARAMS = [
            MulticastWorkerBackendParams(
                IP=addr[0], port=addr[1], WorkerID=i, ScatterAddress=args.group, 
                ScatterPort=args.port, Timeout=args.timeout
            ) for i, addr in enumerate(addr_list)
        ]
        return MULTICAST_CONTROLLER_PARAMS, MulticastControllerBackend, MULTICAST_WORKER_PARAMS, MulticastWorkerBackend, args
    elif args.communication_backend == 'grpc-synch':
        GRPC_SYNCH_CONTROLLER_PARAMS = SynchronousgRPCControllerBackendParams()
        GRPC_SYNCH_CONTROLLER_PARAMS.AddressList = addr_list
        GRPC_SYNCH_CONTROLLER_PARAMS.NumWorkers = num_workers
        GRPC_SYNCH_CONTROLLER_PARAMS.NumThreads = args.threads
        GRPC_WORKER_PARAMS = [
            gRPCWorkerBackendParams(
                # TODO: Is there any good reason to go above one thread for workers?
                IP=addr[0], Port=addr[1], WorkerID=i, NumThreads=1
            ) for i, addr in enumerate(addr_list)
        ]
        return GRPC_SYNCH_CONTROLLER_PARAMS, SynchronousgRPCControllerBackend, GRPC_WORKER_PARAMS, gRPCWorkerBackend, args
    elif args.communication_backend == 'grpc-asynch':
        GRPC_ASYNCH_CONTROLLER_PARAMS = AsynchronousgRPCControllerBackendParams()
        GRPC_ASYNCH_CONTROLLER_PARAMS.Workers = addr_list
        GRPC_ASYNCH_CONTROLLER_PARAMS.Timeout = args.timeout
        GRPC_WORKER_PARAMS = [
            gRPCWorkerBackendParams(
                # TODO: Is there any good reason to go above one thread for workers?
                PeerIndex=i, Peers=tuple([addr]), NumThreads=1
                # IP=addr[0], Port=addr[1], WorkerID=i, NumThreads=1
            ) for i, addr in enumerate(addr_list)
        ]
        return GRPC_ASYNCH_CONTROLLER_PARAMS, AsynchronousgRPCControllerBackend, GRPC_WORKER_PARAMS, gRPCWorkerBackend, args
    else:
        raise ValueError(f"Unkown backend name {args.communication_backend}")

