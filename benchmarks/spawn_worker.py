import os
import sys
import argparse
import te.constants
import te.algorithms.formulations.edge_based.distributed as edge_based_solver
import te.algorithms.formulations.path_based.distributed as path_based_solver
from utils.logging import as_fail
from te.algorithms.communication import DistributedSolverNodeParams
from te.algorithms.communication.grpc import gRPCWorkerBackend, gRPCWorkerBackendParams


if __name__ == '__main__':
    """
    Utility for spawning worker nodes.
    Can read arguments from ENV if none are provided.
    """

    parser =argparse.ArgumentParser('Spawn A Worker Node')
    parser.add_argument('--worker-id', type=int, help='Worker ID')
    parser.add_argument('--solver-type', choices=['edge', 'path'], help='Solver type')
    parser.add_argument('--multicast', action='store_true', help='Use UDP Multicast backend')
    parser.add_argument('--hostname', help='Hostname to use')
    parser.add_argument('--port', type=int, help='Port number to bind to')
    parser.add_argument('--local', action='store_true', help='Assume everything is run locally')
    args = parser.parse_args()

    if args.worker_id is None:
        worker_id = int(os.getenv('WORKER_ID'))
    else:
        worker_id = args.worker_id

    if worker_id is None or worker_id < 0:
        print(as_fail('Worker ID was not properly initialized!'), file=sys.stderr)
        sys.exit(-1)

    if args.solver_type is None:
        solver_type = os.getenv('SOLVER_TYPE')
    else:
        solver_type = args.solver_type

    if solver_type is None:
        print(as_fail('Solver type not provided!'), file=sys.stderr)
        sys.exit(-1)
    elif (solver_type != 'edge' and solver_type != 'path'):
        print(as_fail(f'Cannot recognize solver type {solver_type}'), file=sys.stderr)
        sys.exit(-1)

    if args.local:
        hostname = "localhost"
    else:
        if args.hostname is None:
            hostname = os.getenv('HOSTNAME', f'n{worker_id}')
        else:
            hostname = args.hostname

    if args.local:
        if args.port is None:
            port = os.getenv('GRPC_PORT', te.constants.DEFAULT_RPC_PORT + worker_id + 1)
        else:
            port = args.port
    else:
        if args.port is None:
            port = os.getenv('GRPC_PORT', te.constants.DEFAULT_RPC_PORT)
        else:
            args.port

    multicast = True if args.multicast else int(os.getenv('TE_MULTICAST', 0)) > 0

    if not multicast:
        rpc_params = gRPCWorkerBackendParams(
            PeerIndex=worker_id, Peers=tuple([(hostname, port)])
        )
        rpc_cls = gRPCWorkerBackend
    else:
        raise NotImplementedError
        # rpc_params = MulticastWorkerBackendParams(
        #     PeerIndex=worker_id, Peers=tuple([(hostname, port)])
        # )
        # rpc_cls = MulticastWorkerBackend
    
    print(f'RPC Parameters:\n{rpc_params.str_all()}')
    worker_node_cls =\
        path_based_solver.OnlineTEWorkerNode\
        if solver_type == 'path'\
        else edge_based_solver.OnlineTEWorkerNode
    print(f'Node Class: {worker_node_cls}')
    worker_node_cls.spawn_and_run(DistributedSolverNodeParams(
        CommunicationBackendCLS=rpc_cls, RPCParams_=rpc_params
    ))
