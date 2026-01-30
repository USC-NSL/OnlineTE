from dataclasses import dataclass
from ...edge_based.centralized import GurobiSolverParams, PDLPParams


@dataclass
class GurobiPathBasedSolverParams(GurobiSolverParams):
    NumberOfPathsPerCommodity: int = 8


@dataclass
class PDLPPathBasedSolverParams(PDLPParams):
    NumberOfPathsPerCommodity: int = 8
