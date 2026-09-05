from abc import abstractmethod
from typing import Tuple, Optional, Callable, List
from ..base import CommunicationBackendBase
from te.algorithms.array_utils.cpu_utils import CPUArray, BooleanCPUArray
from te.algorithms.base import SolverParams
from ..admm_synchronous.base import SynchADMMWorkerBackendBase as DomainWorkerCommunicationBackendBase


class MasterCommunicationBackendBase(CommunicationBackendBase):
    @abstractmethod
    def initialize_domain_peers(self, solver_params: SolverParams, basis: CPUArray, 
                                initial_feasible_solution: List[CPUArray],
                                in_out_mask: List[BooleanCPUArray]):
        """Initialize domain controllers with solver parameters and initial feasible solution (X_ek_0)"""

    # @abstractmethod
    # def update_demands(self, updated_feasible_solution: CPUArray):
    #     """Update the initial feasible solution (X_ek_0)"""
    
    @abstractmethod
    def collect_X_ek(self) -> CPUArray:
        """Get the final solution array (X_ek) from the domains"""
    
    @abstractmethod
    def get_admm_consensus_variables(self) -> Tuple[CPUArray, CPUArray]:
        """Get the inner ADMM loop consensus variables for convergence checks"""
    
    @abstractmethod
    def notify_arrived_peers(self, arrival_list: List[Tuple[int, CPUArray]], z_de: CPUArray):
        """Broadcast the updated `Z` values to the arrived domains"""
    
    @abstractmethod
    def close_domains(self):
        """Instruct all domain controllers to stop and close their solvers"""

    @property
    def enqueue_domain_update(self) -> Callable[[CPUArray, CPUArray], None]:
        return self._enqueue_domain_update
    @enqueue_domain_update.setter
    def enqueue_domain_update(self, f: Callable[[CPUArray, CPUArray], None]):
        self._enqueue_domain_update = f


class DomainControllerCommunicationBackendBase(CommunicationBackendBase):
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
    def update_master(self, X_ek_sum: CPUArray, r_e: CPUArray):
        """Enqueue an asynchronous update for the master node"""
    
    @abstractmethod
    def wait_for_master_update(self) -> bool:
        """
        Block until an update from the master node arrives.
        Return `True` when there is an update and `False` if solution has been 
        interrupted.
        """

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
    def set_solver_parameters(self) -> Callable[[SolverParams], None]:
        return self._set_solver_parameters
    @set_solver_parameters.setter
    def set_solver_parameters(self, f: Callable[[SolverParams], None]):
        self._set_solver_parameters = f

    @property
    def collect_X_ek(self) -> Callable[[CPUArray], None]:
        return self._collect_X_ek
    @collect_X_ek.setter
    def collect_X_ek(self, f: Callable[[CPUArray], None]):
        self._collect_X_ek = f

    @property
    def record_master_update(self) -> Callable[[CPUArray], None]:
        return self._record_master_update
    @record_master_update.setter
    def record_master_update(self, f: Callable[[CPUArray], None]):
        self._record_master_update = f
    
    @property
    def get_admm_consensus_variables(self) -> Callable[[None], Tuple[CPUArray, CPUArray]]:
        return self._get_admm_consensus_variables
    @get_admm_consensus_variables.setter
    def get_admm_consensus_variables(self, f: Callable[[None], Tuple[CPUArray, CPUArray]]):
        self._get_admm_consensus_variables = f
    