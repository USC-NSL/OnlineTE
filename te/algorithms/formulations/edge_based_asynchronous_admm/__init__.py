from dataclasses import dataclass
from typing import Tuple, ClassVar
from ..edge_based_distributed_admm import DistributedADMMSolverParams
from te.algorithms.base import SolverParams


@dataclass
class AsynchronousADMMSolverParams(DistributedADMMSolverParams):
    Upsilon: int = 1
    WorkerBatchSize: int = 1
    Sigma: int = 1
    ADMMConvTol: float = 1e-6


@dataclass
class AsynchronousADMMControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    NumWorkers: int = 2
    Backend: ClassVar[str] = ""
    QueueTimeout: float = 5.0

    def __post_init__(self):
        self.left_column_share = 0.2


@dataclass
class AsynchronousADMMWorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = 13000
    NumThreads: int = 1
    WorkerID: int = 0
    Backend: ClassVar[str] = ""
    QueueTimeout: float = 5.0
    
    def __post_init__(self):
        self.left_column_share = 0.5
