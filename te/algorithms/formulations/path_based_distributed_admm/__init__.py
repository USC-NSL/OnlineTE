import te.constants
from typing import Optional, Tuple, Literal, ClassVar
from dataclasses import dataclass
from te.algorithms.base import GurobiSolverParams, SolverParams
from te.algorithms.array_utils import SINGLE_PRECISION

import warnings
warnings.filterwarnings("error")
"""This is mostly to catch overflow, they can be devistating!"""


@dataclass
class PathBasedDistributedADMMSolverParams(GurobiSolverParams):
    NumberOfPathsPerCommodity: int = 16
    TopologyName: Optional[str] = None
    NumberOfEpochs: Optional[int] = 50
    NumberOfNetworkUpdates: int = 2
    Rho: float = 1.0
    Eta: float = 8.0
    Gamma: float = 1.0
    Kappa: float = 0.2
    QPIterations: int = 4
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    Seed: int = te.constants.DEFAULT_SEED


@dataclass
class PathBasedDistributedADMMWorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = 13000
    NumThreads: int = 1
    WorkerID: int = 0
    Multicast: bool = False
    Timeout: float = 5

    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class PathBasedDistributedADMMControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    NumWorkers: int = 2
    Backend: ClassVar[str] = ""
    
    def __post_init__(self):
        self.left_column_share = 0.2
