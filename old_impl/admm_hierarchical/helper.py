import argparse
from typing import Optional, Tuple, List, Dict
from .. import P2PRPCParams, WorkerRPCParams
from .base import MasterCommunicationBackendBase, DomainControllerCommunicationBackendBase, DomainWorkerCommunicationBackendBase
from .domain_backends.asynchronous_grpc_backend import AsynchronousgRPCDomainControllerBackendParams, AsynchronousgRPCDomainControllerBackend
from .master_backends.asynchronous_grpc_backend import AsynchronousgRPCMasterBackendParams, AsynchronousgRPCMasterBackend
from ..admm_synchronous.worker_backends.grpc_backend import gRPCWorkerBackendParams, gRPCWorkerBackend


def add_admm_hierarchical_communication_backend_subparser(parser: argparse.ArgumentParser) -> List[argparse.ArgumentParser]:
    backend_subparser = parser.add_subparsers(dest='communication_backend', help='Communication backend to use', required=True)

    GRPC_ASYNCH_PARAMS = AsynchronousgRPCMasterBackendParams()
    grpc_asynch_params_parser = backend_subparser.add_parser('grpc-asynch', 
                                                             help='Options for the asynchronous gRPC communication backend')
    grpc_asynch_params_parser.add_argument('--timeout', type=float, default=GRPC_ASYNCH_PARAMS.Timeout, 
                                           help='Timeout on future `get` operations')
    grpc_asynch_params_parser.add_argument('--threads', type=int, default=GRPC_ASYNCH_PARAMS.Threads, 
                                           help='Number of threads for RPC servicers')
    
    return [grpc_asynch_params_parser]


def parse_add_admm_hierarchical_communication_backend_params(
    domain_partitions: List[int], 
    master_addr: Tuple[str, int],
    domain_addr_list: List[Tuple[str, int]],
    worker_addr_list_of_lists: List[List[Tuple[str, int]]],
    args: Optional[argparse.Namespace] = None, 
    parser: Optional[argparse.ArgumentParser] = None
) -> Tuple[
    P2PRPCParams, type[MasterCommunicationBackendBase],                   # Master backend param/cls
    List[P2PRPCParams], type[DomainControllerCommunicationBackendBase],   # Domain backends params/cls
    List[List[P2PRPCParams]], type[DomainWorkerCommunicationBackendBase], # Worker backends params/cls
    argparse.Namespace
]:
    NUM_DOMAINS = len(domain_partitions)
    assert len(domain_addr_list) == NUM_DOMAINS
    assert len(worker_addr_list_of_lists) == NUM_DOMAINS
    for partition_len, domain_worker_addr_list in zip(domain_partitions, worker_addr_list_of_lists):
        assert len(domain_worker_addr_list) == partition_len
    
    if args is None:
        assert parser is not None
        args = parser.parse_args()
    
    peer_network_addrs = tuple([master_addr] + domain_addr_list)
    
    if args.communication_backend == 'grpc-asynch':
        GRPC_ASYNCH_MASTER_PARAMS = AsynchronousgRPCMasterBackendParams(
            Index=0, Peers=peer_network_addrs, 
            Workers=tuple(), Timeout=args.timeout, 
            Threads=args.threads
        )
        GRPC_ASYNCH_DOMAIN_PARAMS_LIST = [
            AsynchronousgRPCDomainControllerBackendParams(
                Index=i+1, Peers=peer_network_addrs,
                Workers=worker_addr_list_of_lists[i],
                MasterPeerID=0, Threads=args.threads,
                Timeout=args.timeout
            ) for i in range(NUM_DOMAINS)
        ]
        GRPC_ASYNCH_WORKER_PARAMS_LIST_OF_LISTS = [
            [gRPCWorkerBackendParams(
                IP=worker_addr[0], Port=worker_addr[1],
                WorkerID=worker_id, NumThreads=args.threads
            ) for worker_id, worker_addr in enumerate(worker_addr_list_of_lists[domain_id])] 
            for domain_id in range(NUM_DOMAINS)
        ]
        return \
            GRPC_ASYNCH_MASTER_PARAMS, AsynchronousgRPCMasterBackend, \
            GRPC_ASYNCH_DOMAIN_PARAMS_LIST, AsynchronousgRPCDomainControllerBackend, \
            GRPC_ASYNCH_WORKER_PARAMS_LIST_OF_LISTS, gRPCWorkerBackend, \
            args
    else:
        raise ValueError(f"Unkown backend name {args.communication_backend}")

