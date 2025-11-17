import te.constants
from typing import Optional, Literal
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from te.algorithms.array_utils import SINGLE_PRECISION

import warnings
warnings.filterwarnings("error")
"""This is mostly to catch overflow, they can be devistating!"""


@dataclass
class HierarchicalADMMSolverParams(SolverParams):
    # NumberOfConsensusUpdates: Optional[int] = 10
    # NumberOfDomainUpdates: int = 10
    # NumberOfNetworkUpdates: int = 3
    # Beta: float = 1.0
    NumberOfDomains: int = 1
    NumberOfMasterUpdates: Optional[int] = 30
    NumberOfDomainUpdates: int = 3
    MasterUpdateBarrierSize: int = 1
    MasterUpdateMaxLag: int = 1
    Rho: float = 1.0
    Eta: float = 0.5
    Gamma: float = 1.0
    PGDIterations: int = 2
    ConvTol: float = 1e-3
    Precision: Literal['double', 'single', 'half'] = SINGLE_PRECISION
    TMSeed: int = te.constants.DEFAULT_SEED

    def __post_init__(self):
        self._left_column_share = 0.5
