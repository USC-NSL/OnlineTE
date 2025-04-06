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
    NumberOfEpochs: Optional[int] = None
    NumberOfNetworkUpdates: int = te.constants.DEFAULT_NUMBER_OF_NETWORK_UPDATES
    Rho: float = te.constants.DEFAULT_RHO
    Eta: float = te.constants.DEFAULT_ETA
    Gamma: float = 1
    Kappa: float = 0
    PGDIterations: int = 5
    BigGamma: float = te.constants.DEFAULT_BIG_GAMMA
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    Seed: int = te.constants.DEFAULT_SEED


@dataclass
class DistributedADMMWorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = 13000
    NumThreads: int = 1


@dataclass
class DistributedADMMControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    NumWorkers: int = 1
    NumThreads: int = 1
    Backends: str = "gRPC-asynchronous"
    
    def __post_init__(self):
        self.left_column_share = 0.2
