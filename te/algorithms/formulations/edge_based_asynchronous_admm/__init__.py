import te.constants
from dataclasses import dataclass
from typing import Tuple, ClassVar, Literal, Optional
from te.algorithms.base import SolverParams, GurobiSolverParams
from te.algorithms.array_utils import SINGLE_PRECISION


@dataclass
class AsynchronousADMMSolverParams(GurobiSolverParams):
    NumberOfEpochs: int = 100
    Rho: float = 1.0
    Eta: float = 8.0
    Gamma: float = 1.0
    QPIterations: int = 2
    BigGamma: float = 1e-7
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    Seed: int = te.constants.DEFAULT_SEED
    Upsilon: int = 1
    WorkerBatchSize: int = 1
    Sigma: int = 1
    ADMMConvTol: float = 1e-6


@dataclass
class AsynchronousADMMControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    NumWorkers: int = 2
    Backend: ClassVar[str] = ""
    QueueTimeout: float = 1.0

    def __post_init__(self):
        self.left_column_share = 0.2


@dataclass
class AsynchronousADMMWorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = 13000
    NumThreads: int = 1
    WorkerID: int = 0
    Backend: ClassVar[str] = ""
    QueueTimeout: float = 1.0
    QuitTimeout: Optional[float] = 30.0
    
    def __post_init__(self):
        self.left_column_share = 0.5
