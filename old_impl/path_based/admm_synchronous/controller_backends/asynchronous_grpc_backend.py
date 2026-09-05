
from .. import SynchADMMSolverParams
from te.algorithms.communication.grpc.asynchronous_coordinator_backend import *

from protos.path_based.path_based_pb2 import PathBasedSolverParameters


class SynchADMMCoordinatorBackend(AsynchronousgRPCCoordinatorBackend[SynchADMMSolverParams]):
    def are_all_peers_reachable(self):
        # This is the only peer!
        return True

    def serialize_solver_params(self, solver_params: SynchADMMSolverParams):
        return PathBasedSolverParameters(**solver_params.child_fields)


__all__ = ['SynchADMMCoordinatorBackend', 'AsynchronousgRPCCoordinatorBackendParams']