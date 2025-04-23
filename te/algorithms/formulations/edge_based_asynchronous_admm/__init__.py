from ..edge_based_distributed_admm import *


@dataclass
class AsynchronousADMMSolverParams(DistributedADMMSolverParams):
    Upsilon: int = 1
    WorkerBatchSize: int = 1
    Sigma: int = 1
    ADMMConvTol: float = 1e-6


@dataclass
class AsynchronousADMMControllerRPCParams(DistributedADMMControllerRPCParams):
    QueueTimeout: float = 5.0


@dataclass
class AsynchronousADMMWorkerRPCParams(DistributedADMMWorkerRPCParams):
    QueueTimeout: float = 5.0
