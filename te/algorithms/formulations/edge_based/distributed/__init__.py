from typing import Tuple
from dataclasses import dataclass
from te.algorithms.base import SolverParams


DEFAULT_RPC_PORT = 13000


@dataclass
class WorkerRPCParams(SolverParams):
    IP: str = "localhost"
    Port: int = DEFAULT_RPC_PORT
    WorkerID: int = 0

    def __post_init__(self):
        self.left_column_share = 0.5


@dataclass
class ControllerRPCParams(SolverParams):
    AddressList: Tuple[Tuple[str, int]] = (("localhost", DEFAULT_RPC_PORT),)
    NumWorkers: int = 1
    
    def __post_init__(self):
        self.left_column_share = 0.2
