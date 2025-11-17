from abc import abstractmethod
from typing import Tuple, Optional, Callable
from ..base import CommunicationBackendBase
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray
from te.algorithms.base import SolverParams


class SynchADMMControllerBackendBase(CommunicationBackendBase):
    # For the synchronous case, there is only one controller!
    def are_all_peers_reachable(self):
        return True
    
    @abstractmethod
    def initialize_worker_nodes(self, solver_params: SolverParams, basis: CPUArray, initial_feasible_solution: CPUArray,
                                in_out_mask: Optional[BooleanCPUArray] = None):
        """Initialize worker nodes with solver parameters and initial feasible solution (X_ek_0)"""

    @abstractmethod
    def update_demands(self, updated_feasible_solution: CPUArray):
        """Update the initial feasible solution (X_ek_0)"""
    
    @abstractmethod
    def get_X_ek(self, basis: CPUArray, initial_feasible_solution: CPUArray) -> CPUArray:
        """Get the final solution array (X_ek)"""
    
    @abstractmethod
    def get_X_ek_sum(self) -> CPUArray:
        """Get the total flow over each edge"""
    
    @abstractmethod
    def do_network_update(self, epoch: int, F_e: Optional[CPUArray] = None) -> Tuple[int, CPUArray]:
        """Do network update for a given epoch and return the aggregate"""
    
    @abstractmethod
    def reconvene_network_updates(self, P_bar_t: CPUArray, Y_bar_t: CPUArray, u_t: CPUArray):
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
    def set_initial_feasible_solution(self) -> Callable[[CPUArray], None]:
        return self._set_initial_feasible_solution
    @set_initial_feasible_solution.setter
    def set_initial_feasible_solution(self, f: Callable[[CPUArray], None]):
        self._set_initial_feasible_solution = f

    @property
    def set_null_space_basis(self) -> Callable[[CPUArray], None]:
        return self._set_null_space_basis
    @set_null_space_basis.setter
    def set_null_space_basis(self, f: Callable[[CPUArray], None]):
        self._set_null_space_basis = f

    @property
    def set_commodity_in_out_mask(self) -> Callable[[CPUArray], None]:
        return self._set_commodity_in_out_mask
    @set_commodity_in_out_mask.setter
    def set_commodity_in_out_mask(self, f: Callable[[CPUArray], None]):
        self._set_commodity_in_out_mask = f
    
    @property
    def do_inner_loop_update(self) -> Callable[[int, Optional[CPUArray]], Tuple[int, CPUArray]]:
        return self._do_inner_loop_update
    @do_inner_loop_update.setter
    def do_inner_loop_update(self, f: Callable[[int, Optional[CPUArray]], Tuple[int, CPUArray]]):
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
    def set_solver_parameters(self) -> Callable[[SolverParams], None]:
        return self._set_solver_parameters
    @set_solver_parameters.setter
    def set_solver_parameters(self, f: Callable[[SolverParams], None]):
        self._set_solver_parameters = f
