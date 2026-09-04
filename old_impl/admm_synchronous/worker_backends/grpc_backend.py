from typing import Any, Optional
from .. import SynchADMMSolverParams
from te.algorithms.communication.grpc.worker_backend import *
from utils.logging import as_fail

from protos.edge_based.edge_based_pb2 import EdgeBasedSolverParameters
from google.protobuf.json_format import MessageToDict


class SynchADMMgRPCWorkerBackend(gRPCWorkerBackend[SynchADMMSolverParams]):
    def deserialize_solver_params(self, buf: Any) -> Optional[SynchADMMSolverParams]:
        if buf.Is(EdgeBasedSolverParameters.DESCRIPTOR):
            new_params = EdgeBasedSolverParameters()
            buf.Unpack(new_params)
            return SynchADMMSolverParams(**MessageToDict(new_params))
        print(as_fail(f"Failed to parse solver parameters from coordinator!"))
