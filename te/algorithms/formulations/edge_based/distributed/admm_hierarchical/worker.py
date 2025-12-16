import sys
import signal
import contextlib
from typing import Optional
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray, set_cpu_float_precision
from . import HierarchicalADMMSolverParams
from .base import DomainWorkerCommunicationBackendBase
from ..base import DistributedSolverNodeParams
from ..admm_synchronous.worker import SynchADMMWorkerNode


class DomainWorkerNode(SynchADMMWorkerNode):
    def __init__(self, params: DistributedSolverNodeParams):
        super().__init__(params)
        # self.worker_id = params.rpc_params.WorkerID
        # self._rpc_params = params.rpc_params
        # self._solver_params = solver_params
        # self._ready: bool = False
        
        # self._K: Optional[int] = None
        # self._T: Optional[int] = None
        # self._NUM_EDGES: Optional[int] = None
        # self._CHUNK_LEN: Optional[int] = None
        # self._NULL_M: Optional[CPUArray] = None
        # self._NNT_M: Optional[CPUArray] = None
        # self._MASK_M_chunk: Optional[BooleanCPUArray] = None
        # self._X_ek_start_chunk: Optional[CPUArray] = None
        # self._Y_bar_t_cached: Optional[CPUArray] = None
        # self._P_bar_t_cached: Optional[CPUArray] = None
        # self._u_t_cached: Optional[CPUArray] = None
        # self._Y_tk_chunk: Optional[CPUArray] = None
        # self._lambda_ek_chunk: Optional[CPUArray] = None

        # assert issubclass(params.communication_backend, DomainWorkerCommunicationBackendBase)
        # self._backend: DomainWorkerCommunicationBackendBase = params.communication_backend(params.rpc_params)
        # self._backend.start()
        
        # self._die_on_next_int = False
        # signal.signal(signal.SIGINT, self.stop)
        # signal.signal(signal.SIGTERM, self.die)
    
    def set_solver_parameters(self, new_params: HierarchicalADMMSolverParams):
        self._solver_params = new_params
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    @classmethod
    def spawn_and_run(cls, params: DistributedSolverNodeParams):
        with contextlib.closing(cls(params)) as worker:
            worker.initialize()
            worker.run()


if __name__ == '__main__':
    import socket
    import argparse
    import te.constants
    from utils.logging import as_fail
    from ..admm_synchronous.worker_backends.grpc_backend import gRPCWorkerBackend, gRPCWorkerBackendParams

    parser =argparse.ArgumentParser('Spawn A Domain Worker Node')
    parser.add_argument('--num-domains', type=int, help='Total number of domains', required=True)
    parser.add_argument('--worker-id', type=int, help='Globally unique node worker ID', required=True)
    parser.add_argument('--hostname', help='Hostname to use')
    parser.add_argument('--port', type=int, help='Port to bind to')
    parser.add_argument('--local', action='store_true', help='Assume everything is local')
    args = parser.parse_args()

    num_domains: int = args.num_domains
    worker_id: int = args.worker_id
    hostname: Optional[str] = args.hostname
    port: Optional[int] = args.port
    if num_domains <= 0:
        print(as_fail('Number of domains was not properly initialized'), file=sys.stderr)
        sys.exit(-1)
    if worker_id < 0:
        print(as_fail('Worker ID was not properly initialized'), file=sys.stderr)
        sys.exit(-1)
    
    if args.local:
        hostname = "localhost"
        if port is None:
            port = te.constants.DEFAULT_RPC_PORT + num_domains + 1 + worker_id
    else:
        if hostname is None:
            hostname = f'n{worker_id}'
        if port is None:
            port = te.constants.DEFAULT_RPC_PORT
    
    rpc_params = gRPCWorkerBackendParams(
        IP=socket.gethostbyname(hostname), Port=port,
        WorkerID=worker_id, NumThreads=1
    )
    rpc_cls = gRPCWorkerBackend
    
    print(f'RPC Parameters:\n{rpc_params.str_all()}')
    DomainWorkerNode.spawn_and_run(DistributedSolverNodeParams(
        CommunicationBackendCLS=rpc_cls, RPCParams_=rpc_params
    ))
