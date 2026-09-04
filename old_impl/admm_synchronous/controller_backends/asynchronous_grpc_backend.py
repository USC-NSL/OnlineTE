
from .. import SynchADMMSolverParams
from te.algorithms.communication.grpc.asynchronous_coordinator_backend import *

from protos.edge_based.edge_based_pb2 import EdgeBasedSolverParameters


class SynchADMMCoordinatorBackend(AsynchronousgRPCCoordinatorBackend[SynchADMMSolverParams]):
    def are_all_peers_reachable(self):
        # This is the only peer!
        return True

    def serialize_solver_params(self, solver_params: SynchADMMSolverParams):
        return EdgeBasedSolverParameters(**solver_params.child_fields)


__all__ = ['SynchADMMCoordinatorBackend', 'AsynchronousgRPCCoordinatorBackendParams']