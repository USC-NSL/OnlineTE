from typing import Optional
from dataclasses import dataclass
from utils.gurobi_utils import GurobiSolverParams
from utils.pdlp_utils import PDLPSolverParams


@dataclass(frozen=True)
class GurobiPathBasedSolverParams(GurobiSolverParams):
    max_num_paths_per_commodity: int = 8
    path_file: Optional[str] = None


@dataclass(frozen=True)
class PDLPPathBasedSolverParams(PDLPSolverParams):
    max_num_paths_per_commodity: int = 8
    path_file: Optional[str] = None
