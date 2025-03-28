import te.constants
from typing import Optional, Tuple, Literal
from dataclasses import dataclass
from te.algorithms.base import GurobiSolverParams
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
    NumWorkers: int = 1
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    Seed: int = te.constants.DEFAULT_SEED


@dataclass
class DistributedADMMWorkerRPCParams:
    ip: str = "localhost"
    port: int = 13000
    num_threads: int = 1


@dataclass
class DistributedADMMControllerRPCParams:
    addr_list: Tuple[Tuple[str, int]] = (("localhost", 13000),)
    num_threads: int = 1
