from abc import abstractmethod
from typing import Tuple, List, Callable
from te.algorithms.formulations.edge_based.distributed.base import CommunicationBackendBase
from te.algorithms.array_utils.cpu_utils import CPUArray, IntegerCPUArray
from te.algorithms.base import SolverParams


class SynchADMMControllerBackendBase(CommunicationBackendBase):
    # For the synchronous case, there is only one controller!
    def are_all_peers_reachable(self):
        return True
    
    @abstractmethod
    def initialize_worker_nodes(self, solver_params: SolverParams,
                                alpha_rows: List[IntegerCPUArray],
                                alpha_cols: List[IntegerCPUArray],
                                alpha_shape: Tuple[int, int, int],
                                beta_k: IntegerCPUArray, demands_k: CPUArray):
        """Initialize worker nodes with solver parameters and the path mask matrix"""

    @abstractmethod
    def update_demands(self, demands_k: CPUArray):
        """Update the current demand values (d_k)"""
    
    @abstractmethod
    def get_X_ek(self) -> CPUArray:
        """Get the final (edge-based) solution array (X_ek)"""
    
    @abstractmethod
    def do_network_update(self, epoch: int) -> Tuple[int, CPUArray]:
        """Do network update for a given epoch and return the aggregate"""
    
    @abstractmethod
    def reconvene_network_updates(self, sharing_mean_1: CPUArray, sharing_mean_2: CPUArray, sharing_dual: CPUArray):
        """Finalize network updates for a single inner ADMM iteration"""
    
    @abstractmethod
    def set_active_commodity_count(self, K: int):
        """Set total number of active commodities in the network (needed for local updates)"""


class SynchADMMWorkerBackendBase(CommunicationBackendBase):
    # These workers never directly reach-out to any node, they are passive
    def are_all_peers_reachable(self):
        return True
    def are_all_workers_reachable(self):
        return True

    @property
    def set_path_mask_shape(self) -> Callable[[Tuple[int, int, int]], None]:
        return self._set_path_mask_shape
    @set_path_mask_shape.setter
    def set_path_mask_shape(self, f: Callable[[Tuple[int, int, int]], None]):
        self._set_path_mask_shape = f
    @property
    def set_path_mask_rows(self) -> Callable[[List[IntegerCPUArray]], None]:
        return self._set_path_mask_rows
    @set_path_mask_rows.setter
    def set_path_mask_rows(self, f: Callable[[List[IntegerCPUArray]], None]):
        self._set_path_mask_rows = f
    @property
    def set_path_mask_cols(self) -> Callable[[List[IntegerCPUArray]], None]:
        return self._set_path_mask_cols
    @set_path_mask_cols.setter
    def set_path_mask_cols(self, f: Callable[[List[IntegerCPUArray]], None]):
        self._set_path_mask_cols = f

    @property
    def set_path_count(self) -> Callable[[IntegerCPUArray], None]:
        return self._set_path_count
    @set_path_count.setter
    def set_path_count(self, f: Callable[[IntegerCPUArray], None]):
        self._set_path_count = f

    @property
    def set_demands(self) -> Callable[[CPUArray], None]:
        return self._set_demands
    @set_demands.setter
    def set_demands(self, f: Callable[[CPUArray], None]):
        self._set_demands = f

    @property
    def set_solver_parameters(self) -> Callable[[SolverParams], None]:
        return self._set_solver_parameters
    @set_solver_parameters.setter
    def set_solver_parameters(self, f: Callable[[SolverParams], None]):
        self._set_solver_parameters = f
    
    @property
    def do_inner_loop_update(self) -> Callable[[int], Tuple[int, CPUArray]]:
        return self._do_inner_loop_update
    @do_inner_loop_update.setter
    def do_inner_loop_update(self, f: Callable[[int], Tuple[int, CPUArray]]):
        self._do_inner_loop_update = f

    @property
    def set_active_commodity_count(self) -> Callable[[int], None]:
        return self._set_active_commodity_count
    @set_active_commodity_count.setter
    def set_active_commodity_count(self, f: Callable[[int], None]):
        self._set_active_commodity_count = f
    
    @property
    def update_cached_values(self) -> Callable[[CPUArray, CPUArray, CPUArray], None]:
        return self._update_cached_values
    @update_cached_values.setter
    def update_cached_values(self, f: Callable[[CPUArray, CPUArray, CPUArray], None]):
        self._update_cached_values = f
    
    @property
    def report_chunk(self) -> Callable[[None], CPUArray]:
        return self._report_chunk
    @report_chunk.setter
    def report_chunk(self, f: Callable[[None], CPUArray]):
        self._report_chunk = f
    
    @property
    def report_aggregate(self) -> Callable[[None], CPUArray]:
        return self._report_aggregate
    @report_aggregate.setter
    def report_aggregate(self, f: Callable[[None], CPUArray]):
        self._report_aggregate = f
    
    @property
    def jit_warmstart(self) -> Callable[[None], None]:
        return self._jit_warmstart
    @jit_warmstart.setter
    def jit_warmstart(self, f: Callable[[None], None]):
        self._jit_warmstart = f
