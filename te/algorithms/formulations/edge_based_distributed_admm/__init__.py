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
    NumberOfNetworkUpdates: int = 3
    NumberOfLocalUpdates: int = 0
    Rho: float = 8.0
    Eta: float = 0.5
    Gamma: float = 1.0
    Kappa: float = 0.01
    QPIterations: int = 2
    QPMethod: Literal['PGD', 'ADMM'] = 'PGD'
    BigGamma: float = 1e-7
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    Seed: int = te.constants.DEFAULT_SEED

    def __post_init__(self):
        # We override `BarConvTol` to be the same value as `BigGamma`
        self.ConvTol = self.BigGamma
        self._left_column_share = 0.5


@dataclass
class DistributedADMMWorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = 13000
    NumThreads: int = 1
    WorkerID: int = 0
    Multicast: bool = False
    Timeout: float = 5

    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class DistributedADMMControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    NumWorkers: int = 2
    Backend: ClassVar[str] = ""
    
    def __post_init__(self):
        self.left_column_share = 0.2
