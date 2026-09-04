import time
import numpy as np
import asyncio.exceptions
from typing import Optional, Tuple
from ..base import DistributedSolverNodeBase
from utils.exceptions import SolutionInterrupted
from utils.logging import as_info, as_success
from te.algorithms.array_utils import set_global_precision
from te.algorithms.array_utils.cpu_utils import (CPUArray, BooleanCPUArray,
                                                 cpu_array, cpu_zeros, 
                                                 set_cpu_float_precision)
from te.algorithms.sub_algorithms.admm_consensus_test import norm_in_consensus
from . import HierarchicalADMMSolverParams
from .base import DomainControllerCommunicationBackendBase
from ..base import DistributedSolverNodeParams


class DomainControllerNode(DistributedSolverNodeBase):
    def __init__(self, params: DistributedSolverNodeParams) -> None:
        super().__init__(params)
        self._rpc_params = params.RPCParams_
        self._solver_params: Optional[HierarchicalADMMSolverParams] = None
        self._rng: Optional[np.random.Generator] = None

        self._NULL_M: CPUArray = None
        self._NNT_M: CPUArray = None
        
        self._T: Optional[int] = None
        self._NUM_EDGES: Optional[int] = None
        self._K: Optional[int] = None
        self._M_MASK: Optional[BooleanCPUArray] = None

        self._X_ek: Optional[CPUArray] = None
        self._X_ek_start: Optional[CPUArray] = None
        self._X_ek_sum_e: Optional[CPUArray] = None
        self._Z_e_start: Optional[CPUArray] = None
        self._Z_e: Optional[CPUArray] = None
        self._r_e: Optional[CPUArray] = None

        self._P_bar_t: Optional[CPUArray] = None
        self._Y_bar_t: Optional[CPUArray] = None
        self._u_t: Optional[CPUArray] = None

        self.backend: DomainControllerCommunicationBackendBase = params.CommunicationBackendCLS(params.RPCParams_)
        self.backend.set_commodity_in_out_mask = self.set_commodity_in_out_mask
        self.backend.set_initial_feasible_solution = self.set_initial_feasible_solution
        self.backend.set_null_space_basis = self.set_null_space_basis
        self.backend.get_admm_consensus_variables = self.get_admm_consensus_variables
        self.backend.record_master_update = self.record_master_update
        self.backend.start()
    
    def run(self):
        self.solve()
    
    def is_initialized_by_master(self) -> bool:
        return self._solver_params is not None and \
            self._X_ek_start is not None and \
            self._NULL_M is not None and \
            self._M_MASK is not None

    def initialize(self):
        print(as_info("Waiting for workers to become reachable"))
        while self.backend.is_alive and not self.backend.are_all_workers_ready():
            time.sleep(1)
        if not self.backend.is_alive:
            raise SolutionInterrupted
        print(as_success("All worker nodes are reachable"))
        print(as_info("Waiting for master to initialize domain node"))
        while self.backend.is_alive and not self.is_initialized_by_master():
            time.sleep(1)
        if not self.backend.is_alive:
            raise SolutionInterrupted
        print(as_success("Master initialized this domain node"))
        set_global_precision(self._solver_params.Precision)
        set_cpu_float_precision()
        self._set_initial_feasible_solution()
        self._set_NULL_M()
        self._initialize_variables_and_residuals()
        self.backend.initialize_worker_nodes(
            self._solver_params,
            self._NULL_M, 
            self._X_ek_start,
            self._M_MASK
        )

    def _set_initial_feasible_solution(self):
        self._X_ek_sum_e = np.sum(self._X_ek_start, axis=1)
        self._Z_e_start = cpu_array(self._X_ek_sum_e)
        self._K = self._X_ek_start.shape[-1]
    
    def _set_NULL_M(self):
        assert self._NULL_M is not None and self._M_MASK is not None
        N = self._NULL_M
        n, T = N.shape
        self._NULL_M = N
        self._NNT_M = N @ N.T
        self._T = T
        self._NUM_EDGES = n
    
    def _initialize_variables_and_residuals(self):
        T = self._T
        NUM_EDGES = self._NUM_EDGES
        self._Z_e = cpu_array(self._Z_e_start)
        self._r_e = cpu_zeros((NUM_EDGES,))
        self._u_t = cpu_zeros((T,))
        self._P_bar_t = cpu_zeros((T,))
        self._Y_bar_t = cpu_zeros((T,))
        self._X_ek = cpu_array(self._X_ek_start)

    def are_peer_network_nodes_ready(self):
        return self.backend.are_all_peers_reachable()

    def _get_F(self) -> np.ndarray:
        return self._Z_e - self._Z_e_start - self._r_e
    
    def _set_X_ek(self):
        self._X_ek = self.backend.get_X_ek(basis=self._NULL_M, initial_feasible_solution=self._X_ek_start)

    def _do_network_update(self, epoch: int):
        max_run, self._Y_bar_t = self.backend.do_network_update(epoch)
        return max_run
    
    def _update_P_bar(self):
        K = self._K
        ETA = self._solver_params.Eta
        RHO = self._solver_params.Rho
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        F_E = self._get_F()
        NULL_M = self._NULL_M
        P_BAR_T = (NULL_M.T @ F_E / K + (ETA/RHO) * (U_T + Y_BAR_T)) / (1 + (ETA/RHO))
        self._P_bar_t = P_BAR_T
    
    def _update_u_t(self):
        U_T = self._u_t
        Y_BAR_T = self._Y_bar_t
        P_BAR_T = self._P_bar_t

        self._u_t = U_T + (Y_BAR_T - P_BAR_T)
    
    def _reconvene_network_updates(self) -> bool:
        self._update_P_bar()
        self._update_u_t()
        self.backend.reconvene_network_updates(
            P_bar_t=self._P_bar_t,
            Y_bar_t=self._Y_bar_t,
            u_t=self._u_t
        )
        return norm_in_consensus(self._P_bar_t, self._Y_bar_t, 5e-4)
    
    def _update_X_ek_sum(self):
        self._X_ek_sum_e = self._Z_e_start + self._K * self._NULL_M @ self._Y_bar_t
    
    def _update_r_e(self):
        R_E = self._r_e
        Z_E = self._Z_e
        X_EK_SUM_E = self._X_ek_sum_e
        self._r_e = R_E + (X_EK_SUM_E - Z_E)

    def close(self):
        self.backend.close()
    
    def solve(self) -> int:
        PARAMS = self._solver_params
        try:
            while self.backend.is_alive:
                for _ in range(PARAMS.NumberOfDomainUpdates):
                    self._do_network_update(0)
                    self._reconvene_network_updates()
                self._update_X_ek_sum()
                self.backend.update_master(self._X_ek_sum_e, self._r_e)
                if not self.backend.wait_for_master_update():
                    break
                self._update_r_e()
            return 0
        except SolutionInterrupted:
            return 0
        except asyncio.exceptions.CancelledError:
            return -1
    
    def set_initial_feasible_solution(self, X: CPUArray):
        assert self._X_ek_start is None
        self._X_ek_start = X
    
    def set_null_space_basis(self, basis: CPUArray):
        assert self._NULL_M is None
        self._NULL_M = basis
    
    def set_commodity_in_out_mask(self, mask: BooleanCPUArray):
        assert self._M_MASK is None
        self._M_MASK = mask

    def record_master_update(self, new_Z: CPUArray):
        self._Z_e = new_Z

    def get_admm_consensus_variables(self) -> Tuple[CPUArray, CPUArray]:
        return self._Y_bar_t, self._P_bar_t


if __name__ == '__main__':
    import sys
    import argparse
    import te.constants
    from typing import List
    from utils.logging import as_fail
    from .domain_backends.asynchronous_grpc_backend import (AsynchronousgRPCDomainControllerBackend, 
                                                            AsynchronousgRPCDomainControllerBackendParams)
    from te.algorithms.sub_algorithms.mlu_backends.aggregate import add_mlu_backend_subparser, parse_mlu_backend_params

    parser = argparse.ArgumentParser('Spawn A Domain Controller')
    parser.add_argument('--partitions', nargs='+', type=int, help='Number of domain workers for each domain', required=True)
    parser.add_argument('--domain-id', type=int, help='Domain ID for this node', required=True)
    parser.add_argument('--peers', nargs='+', help='List of peer addresses (master and other domains)')
    parser.add_argument('--workers', nargs='+', help='List of worker addresses for this domain')
    parser.add_argument('--local', action='store_true', help='Assume everything is run locally')

    add_mlu_backend_subparser(parser)
    MLU_BACKEND_PARAMS, MLUCLS, args = parse_mlu_backend_params(parser=parser)

    def addr_str_to_tuple(addr_str: str) -> Tuple[str, int]:
        ls = addr_str.split(':')
        assert len(ls) == 2
        return ls[0], int(ls[1])

    num_domains = len(args.partitions)
    peers: Optional[List[str]] = args.peers
    if peers is not None:
        num_peers = len(peers)
        assert num_peers == num_domains+1
    else:
        num_peers = num_domains+1
    domain_id = args.domain_id
    worker_count = np.cumsum(args.partitions)
    worker_count_start = 0 if domain_id == 0 else worker_count[domain_id-1]
    worker_count_domain = worker_count[domain_id]
    DEFAULT_RPC_PORT = te.constants.DEFAULT_RPC_PORT

    if domain_id < 0 or domain_id >= num_peers-1:
        print(as_fail('Domain ID was not properly initialized!'), file=sys.stderr)
        sys.exit(-1)
    else:
        if args.local:
            PEER_ADDRS = tuple([('localhost', DEFAULT_RPC_PORT+i) for i in range(num_peers)])
            WORKER_ADDRS = tuple([('localhost', DEFAULT_RPC_PORT + num_peers + worker_count_start + i) for i in range(worker_count_domain)])
        else:
            if peers is None:
                PEER_ADDRS = tuple([('controller', DEFAULT_RPC_PORT)] + [(f'd{i}', DEFAULT_RPC_PORT) for i in range(num_domains)])
            else:
                PEER_ADDRS = tuple(map(addr_str_to_tuple, peers))
            if args.peers is None:
                WORKER_ADDRS = tuple([(f'n{worker_count_start+i}', DEFAULT_RPC_PORT) for i in range(worker_count_domain)])
            else:
                WORKER_ADDRS = tuple(map(addr_str_to_tuple, args.workers))
        rpc_params = AsynchronousgRPCDomainControllerBackendParams(
            Index=domain_id+1, Peers=PEER_ADDRS, Workers=WORKER_ADDRS
        )
        rpc_cls = AsynchronousgRPCDomainControllerBackend
        print(f'RPC Parameters:\n{rpc_params.str_all()}')
        DomainControllerNode.spawn_and_run(DistributedSolverNodeParams(
            mlu_backend=MLUCLS, mlu_params=MLU_BACKEND_PARAMS,
            communication_backend=rpc_cls, rpc_params=rpc_params
        ))
