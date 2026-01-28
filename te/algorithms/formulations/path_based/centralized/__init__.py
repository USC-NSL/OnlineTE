from dataclasses import dataclass
from ...edge_based.centralized import GurobiSolverParams


@dataclass
class GurobiPathBasedSolverParams(GurobiSolverParams):
    NumberOfPathsPerCommodity: int = 8
