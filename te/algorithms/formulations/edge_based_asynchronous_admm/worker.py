import signal
import threading
import contextlib
from typing import List, Tuple, Optional
from utils.logging import as_warning
from utils.exceptions import Unreachable
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, set_cpu_float_precision
from ..edge_based_distributed_admm.worker import NetworkWorkerNode as SynchronousNetworkWorkerNode
from .worker_backends.base import WorkerNodeCommunicationBackendBase
from .worker_backends.udp_multicast_backend import MulticastBackend, MulticastWorkerBackendParams
from . import AsynchronousADMMSolverParams


class NetworkWorkerNode(SynchronousNetworkWorkerNode):
    def __init__(self, rpc_params, solver_params = None):
        self._is_active: bool = False
        self._quit_event: threading.Event = threading.Event()
        super().__init__(rpc_params, solver_params)

        for sig in ('INT', 'TERM'):
            signal.signal(getattr(signal, 'SIG'+sig), self.int_handler)
    
    def int_handler(self, _, __):
        self._is_active = False
        self._quit_event.set()
        self._backend.stop()
        print(as_warning('Interrupted. Will no longer serve update requests or solutions.'))
    
    def wait(self):
        raise Unreachable
    
    def initialize(self):
        self._backend: WorkerNodeCommunicationBackendBase = MulticastBackend(self._rpc_params)
        self._backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self._backend.set_null_space_basis = self.set_null_space_basis
        self._backend.set_solver_parameters = self.set_solver_parameters
        self._backend.report_chunk = self.report_chunk
        self._backend.set_active_commodity_count = self.set_active_commodity_count
        self._backend.is_initialized = self.is_initialized
        self._is_active = True
        self._quit_event.clear()

    def set_solver_parameters(self, new_params: AsynchronousADMMSolverParams):
        self._solver_params = new_params
        self._backend.WorkerBatchSize = new_params.WorkerBatchSize
        self._backend.Sigma = new_params.Sigma
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def consume_batch_update(self, batch: List[Tuple[CPUArray, CPUArray, CPUArray]]):
        """
        We need to think about this.
        Currently, the best thing that we might be able to do is to just pick the
        most recent update.
        """
        self._u_t_cached, self._P_bar_t_cached, self._Y_bar_t_cached = batch[-1]
    
    def is_initialized(self) -> bool:
        return all([
            self._NNT_M is not None,
            self._NULL_M is not None,
            self._X_ek_start_chunk is not None,
            self._solver_params is not None
        ])
    
    def solve(self):
        if not self._backend.wait_until_initialized():
            return
        assert self._solver_params.QPMethod == 'ADMM'
        number_of_consecutive_updates = 0
        while self._is_active:
            controller_update_batch = self._backend.gather_updates(number_of_consecutive_updates >= self._solver_params.Sigma)
            if controller_update_batch is not None:
                if len(controller_update_batch) > 0:
                    number_of_consecutive_updates = 0
                    self.consume_batch_update(controller_update_batch)
            else:
                break
            runtime, Y_bar = self.do_inner_loop_admm_update(0)
            self._backend.send_update_to_controller(runtime, Y_bar)
            number_of_consecutive_updates += 1

    @staticmethod
    def spawn_and_wait(rpc_params: MulticastWorkerBackendParams, 
                       solver_params: Optional[AsynchronousADMMSolverParams] = None):
        with contextlib.closing(NetworkWorkerNode(rpc_params, solver_params)) as worker:
            worker.initialize()
            worker.solve()


if __name__ == '__main__':
    import sys
    import socket
    import argparse
    from utils.logging import as_fail

    rpc_params = MulticastWorkerBackendParams()

    parser =argparse.ArgumentParser('Spawn A Worker Node')
    parser.add_argument('worker_id', type=int, 
                        help='Worker ID')
    parser.add_argument('--shost', default=rpc_params.IP, 
                        help='Switch hostname')
    parser.add_argument('--chost', default=rpc_params.ControllerHost, 
                        help='Controller hostname')
    parser.add_argument('--group', default=rpc_params.ScatterAddress, 
                        help='Multicast address to listen on for controller updates')
    parser.add_argument('--gport-base', type=int, default=rpc_params.Port, 
                        help='Base gRPC listening port')
    parser.add_argument('--gport', type=int, default=rpc_params.ScatterPort,
                        help='Multicast UDP port to listen on for controller updates')
    parser.add_argument('--cport', type=int, default=rpc_params.ControllerPort,
                        help='Destination UDP port to send updates to the controller')
    
    args = parser.parse_args()

    worker_id = args.worker_id
    if worker_id < 0:
        print(as_fail('Worker ID was not properly initialized!'), file=sys.stderr)
        sys.exit(-1)
    else:
        hostname = args.shost if args.shost is not None else f'n{worker_id}'
        rpc_params = MulticastWorkerBackendParams(
            IP=socket.gethostbyname(hostname), Port=args.gport_base + worker_id,
            WorkerID=worker_id, ScatterAddress=args.group, ScatterPort=args.gport,
            ControllerHost=args.chost, ControllerPort=args.cport
        )
        print(f'RPC Parameters:\n{rpc_params}')
        NetworkWorkerNode.spawn_and_wait(rpc_params)
