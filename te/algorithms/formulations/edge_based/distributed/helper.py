from typing import Tuple
from dataclasses import dataclass
from te.algorithms.base import SolverParams
from .base import DistributedSolverNodeParams, RPCParams, DistributedSolverNodeBase, CommunicationBackendBase


@dataclass(frozen=True)
class PrettyAddressList(SolverParams):
    Addresses: Tuple[Tuple[str, int]]
    _left_column_share = 0.2


__all__ = [
    'PrettyAddressList',
    'DistributedSolverNodeParams', 'RPCParams', 'DistributedSolverNodeBase', 
    'CommunicationBackendBase'
]