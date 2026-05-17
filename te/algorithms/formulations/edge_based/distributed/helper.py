from typing import Tuple
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from .base import DistributedSolverNodeParams, RPCParams, DistributedSolverNodeBase, CommunicationBackendBase


@dataclass
class PrettyAddressList(SolverParams):
    Addresses: Tuple[Tuple[str, int]]
    
    def __post_init__(self):
        self._left_column_share = 0.2


__all__ = [
    'PrettyAddressList',
    'DistributedSolverNodeParams', 'RPCParams', 'DistributedSolverNodeBase', 
    'CommunicationBackendBase'
]