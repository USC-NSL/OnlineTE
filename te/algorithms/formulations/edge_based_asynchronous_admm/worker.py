import sys
import time
import signal
import contextlib
import numpy as np
from typing import List, Optional
from utils.logging import as_warning, as_info
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import CPUArray, cpu_zeros, cpu_array, set_cpu_float_precision
from .worker_backends.base import WorkerNodeCommunicationBackendBase
from .worker_backends.udp_multicast_backend import MulticastBackend, MulticastWorkerBackendParams
from . import AsynchronousADMMSolverParams, AsynchronousADMMWorkerRPCParams, ControllerUpdate, NetworkUpdate


class NetworkWorkerNode:
    def __init__(self, rpc_params: AsynchronousADMMWorkerRPCParams, 
                 solver_params: Optional[AsynchronousADMMSolverParams] = None):
        self.worker_id = rpc_params.WorkerID
        self._rpc_params = rpc_params
        self._solver_params = solver_params
        self._ready: bool = False

        self._K: Optional[int] = None
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._CHUNK_LEN: Optional[int] = None
        self._NULL_M: Optional[CPUArray] = None
        self._NNT_M: Optional[CPUArray] = None

        self._X_ek_start_w: Optional[CPUArray] = None
        # TODO: Get a lighter representation of this, it's too big!
        self._in_out_mask_ek_w: Optional[CPUArray] = None

        self._P_bar_t: Optional[CPUArray] = None
        self._P_bar_t_old: Optional[CPUArray] = None
        self._u_t_w: Optional[CPUArray] = None
        self._Y_tk_w: Optional[CPUArray] = None
        self._P_tk_w: Optional[CPUArray] = None
        self._S_ek_w: Optional[CPUArray] = None
        self._t_ek_w: Optional[CPUArray] = None

        self._backend: Optional[MulticastBackend] = None

        self._is_active: bool = False
        self._die_on_next_int = False
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.die)

    def start(self):
        self._backend.start()
        self._is_active = True

    def stop(self, _, __):
        if self._die_on_next_int:
            signal.raise_signal(signal.SIGTERM)
        else:
            print(as_warning('SIGINT: Stopping worker. Invoke again to kill the process.'))
            if self._backend:
                self._backend.stop()
            self._die_on_next_int = True
    
    def die(self, _, __):
        print(as_warning('SIGTERM: Killing the worker.'))
        if self._backend:
            self._backend.die()
    
    def initialize(self):
        self._backend: WorkerNodeCommunicationBackendBase = MulticastBackend(self._rpc_params)
        self._backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self._backend.set_mask = self.set_mask
        self._backend.set_null_space_basis = self.set_null_space_basis
        self._backend.set_solver_parameters = self.set_solver_parameters
        self._backend.report_chunk = self.report_chunk
        self._backend.set_active_commodity_count = self.set_active_commodity_count
        self._backend.is_initialized = self.is_initialized
        self.start()

    def set_solver_parameters(self, new_params: AsynchronousADMMSolverParams):
        self._solver_params = new_params
        self._backend.WorkerBatchSize = new_params.WorkerBatchSize
        self._backend.Sigma = new_params.Sigma
        set_global_precision(precision=new_params.Precision)
        set_cpu_float_precision()

    def consume_batch_update(self, batch: List[ControllerUpdate]):
        # TODO: Think about what to do when this assertion is not true ...
        assert len(batch) == 1
        update = batch[0]
        self._P_bar_t_old = np.copy(self._P_bar_t)
        self._P_bar_t = update.P_bar_t
        if update.Y_bar_sample is None:
            return
        shift = (self._K / update.sample_size) * (
            self._P_bar_t - self._P_bar_t_old
        ) - (update.Y_bar_sample - update.P_bar_sample + update.u_bar_sample)
        self._P_tk_w = self._Y_tk_w + np.expand_dims(self._u_t_w + shift, axis=1)
        self._u_t_w = -shift
    
    def is_initialized(self) -> bool:
        return all([
            self._NNT_M is not None,
            self._NULL_M is not None,
            self._X_ek_start_w is not None,
            self._solver_params is not None
        ])
    
    def set_initial_feasible_solution(self, X: CPUArray):
        self._X_ek_start_w = X
        self._NUM_EDGES, self._CHUNK_LEN = self._X_ek_start_w.shape
    
    def set_mask(self, mask: CPUArray):
        print(as_info('Running with an input-output mask.'))
        self._in_out_mask_ek_chunk = mask
    
    def set_null_space_basis(self, NULL_M: CPUArray):
        self._NULL_M = NULL_M
        assert self._X_ek_start_w is not None
        CHUNK_LEN = self._CHUNK_LEN
        self._NULL_M = NULL_M
        self._NNT_M = NULL_M @ NULL_M.T
        T = self._NULL_M.shape[1]
        self._T = T

        self._P_bar_t = cpu_zeros((T,))
        self._P_bar_t_old = cpu_zeros((T,))
        self._u_t_w = cpu_zeros((T,))
        self._Y_tk_w = cpu_zeros((T, CHUNK_LEN))
        self._P_tk_w = cpu_zeros((T, CHUNK_LEN))
        self._S_ek_w = np.copy(self._X_ek_start_w)
        self._t_ek_w = cpu_zeros(self._X_ek_start_w.shape)

    def _get_current_C(self) -> CPUArray:
        P_TK = self._P_tk_w
        U_T = self._u_t_w
        
        return P_TK - np.expand_dims(U_T, axis=1)

    def do_inner_loop_admm_update(self) -> NetworkUpdate:
        QP_STEP = self._solver_params.QPStep
        QP_ITERS = self._solver_params.QPIterations
        NULL_M = self._NULL_M
        Y_TK_W = self._Y_tk_w
        X_EK_START_W = self._X_ek_start_w
        C_TK_W = self._get_current_C()
        S_EK_W = self._S_ek_w
        T_EK_W = self._t_ek_w

        start = time.perf_counter_ns()
        for _ in range(QP_ITERS):
            Y_TK_W = (C_TK_W - QP_STEP * NULL_M.T @ (X_EK_START_W + T_EK_W - S_EK_W)) / (1 + QP_STEP)
            S_EK_W = np.clip(X_EK_START_W + NULL_M @ Y_TK_W + T_EK_W, a_min=0, a_max=None)
            if self._in_out_mask_ek_w is not None:
                S_EK_W *= self._in_out_mask_ek_w
            T_EK_W = T_EK_W + (X_EK_START_W + NULL_M @ Y_TK_W - S_EK_W)
        
        self._Y_tk_w = Y_TK_W
        self._S_ek_w = S_EK_W
        self._t_ek_w = T_EK_W
        runtime = time.perf_counter_ns() - start
        
        return NetworkUpdate(
            worker_id=self.worker_id, runtime=runtime,
            Y_bar_w=np.mean(self._Y_tk_w, axis=1),
            P_bar_w=np.mean(self._P_tk_w, axis=1),
            u_w=self._u_t_w,
            Xo_w=np.sum(self._S_ek_w, axis=1)
        )

    def set_active_commodity_count(self, K: int):
        self._K = K

    def update_cached_values(self, u_t: CPUArray, P_bar_t: CPUArray, Y_bar_t: CPUArray):
        self._u_t_cached = u_t
        self._P_bar_t_cached = P_bar_t
        self._Y_bar_t_cached = Y_bar_t

    def solve(self):
        if not self._backend.wait_until_initialized():
            return
        number_of_consecutive_updates = 0
        while self._is_active:
            controller_update_batch = self._backend.gather_updates(number_of_consecutive_updates >= self._solver_params.Sigma)
            if controller_update_batch is not None:
                if len(controller_update_batch) > 0:
                    number_of_consecutive_updates = 0
                    self.consume_batch_update(controller_update_batch)
            else:
                break
            update = self.do_inner_loop_admm_update()
            self._backend.send_update_to_controller(update)
            number_of_consecutive_updates += 1

    @staticmethod
    def spawn_and_wait(rpc_params: MulticastWorkerBackendParams, 
                       solver_params: Optional[AsynchronousADMMSolverParams] = None):
        with contextlib.closing(NetworkWorkerNode(rpc_params, solver_params)) as worker:
            worker.initialize()
            worker.solve()
            if worker._is_active:
                if worker._backend.killed:
                    print(as_warning('Aborting'))
                else:
                    print(as_info('Solution process ended. Will wait for the controller to close.'))
                    if worker._backend.wait_for_close():
                        print(as_warning('Will not wait for the controller anymore. Quitting ...'))

    def report_chunk(self) -> CPUArray:
        return cpu_array(self._S_ek_w)
    
    def close(self):
        self._backend.close()


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
