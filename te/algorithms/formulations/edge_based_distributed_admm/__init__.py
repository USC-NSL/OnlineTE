import te.constants
from typing import Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass
from te.algorithms.base import GurobiSolverParams, SolverParams
from te.algorithms.array_utils import SINGLE_PRECISION

import warnings
warnings.filterwarnings("error")
"""This is mostly to catch overflow, they can be devistating!"""


@dataclass
class DistributedADMMSolverParams(GurobiSolverParams):
    NumberOfEpochs: Optional[int] = 100
    NumberOfNetworkUpdates: int = 4
    Rho: float = 1.0
    Eta: float = 8.0
    Gamma: float = 1.0
    Kappa: float = 0.2
    PGDIterations: int = 2
    BigGamma: float = 1e-7
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    Seed: int = te.constants.DEFAULT_SEED


@dataclass
class DistributedADMMWorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = 13000
    NumThreads: int = 1
    WorkerID: int = 0

    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class DistributedADMMControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    NumWorkers: int = 2
    Backend: ClassVar[str] = ""
    
    def __post_init__(self):
        self.left_column_share = 0.2
