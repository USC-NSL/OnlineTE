from typing import Any, Optional
from .. import SynchADMMSolverParams
from te.algorithms.communication.grpc.worker_backend import *
from utils.logging import as_fail

from protos.path_based.path_based_pb2 import PathBasedSolverParameters
from google.protobuf.json_format import MessageToDict


class SynchADMMgRPCWorkerBackend(gRPCWorkerBackend[SynchADMMSolverParams]):
    def deserialize_solver_params(self, buf: Any) -> Optional[SynchADMMSolverParams]:
        if buf.Is(PathBasedSolverParameters.DESCRIPTOR):
            new_params = PathBasedSolverParameters()
            buf.Unpack(new_params)
            return SynchADMMSolverParams(**MessageToDict(new_params))
        print(as_fail(f"Failed to parse solver parameters from coordinator!"))