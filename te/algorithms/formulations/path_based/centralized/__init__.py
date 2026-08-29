from typing import Optional
from dataclasses import dataclass
from ...edge_based.centralized import GurobiSolverParams, PDLPParams


@dataclass(frozen=True)
class GurobiPathBasedSolverParams(GurobiSolverParams):
    max_num_paths_per_commodity: int = 8
    path_file: Optional[str] = None


@dataclass(frozen=True)
class PDLPPathBasedSolverParams(PDLPParams):
    max_num_paths_per_commodity: int = 8
    path_file: Optional[str] = None
