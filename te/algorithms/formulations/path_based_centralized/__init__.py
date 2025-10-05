from typing import Optional
from dataclasses import dataclass
from te.algorithms.base import GurobiSolverParams


@dataclass
class CentralizedPathBasedSolverParams(GurobiSolverParams):
    NumberOfPathsPerCommodity: int = 16
    TopologyName: Optional[str] = None
