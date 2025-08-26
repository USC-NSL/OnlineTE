import math
import numpy as np
import te.constants
from typing import Optional
from utils.logging import as_warning
from te.algorithms.base import TrafficEngineeringLPEvaluationParams
from te.algorithms.utils import str_round


NEGLIGIBLE_FLOW_ABS_TOL = 1e-3
NEGLIGIBLE_NULL_SPACE_ELEMENT = 5e-3
SEVERE_CONSENSUS_VIOLATION_REL_TOL = 5e-2


def in_consensus(optim, actual, feasibility_tol: Optional[float], feasibility_ratio: Optional[float]):
    """
    Check if `actual` is close to `optim` assignment.
    The test can either use absolute or relative tolerance (if both are present, only
    absolute tolerance is considered).
    """
    if feasibility_tol is not None:
        return math.isclose(optim, actual, abs_tol=feasibility_tol)
    if abs(optim) < te.constants.FLOAT_RES:
        return math.isclose(actual, 0, abs_tol=te.constants.FLOAT_RES)
    return math.isclose(optim, actual, rel_tol=feasibility_ratio)


def outer_admm_consensus_test(primal: np.ndarray, pair: np.ndarray, eval_params: TrafficEngineeringLPEvaluationParams):
    n = np.shape(primal)[0]
    assert primal.shape == pair.shape
    
    if eval_params.FeasibilityRatio is not None:
        feasibility_ratio = min(SEVERE_CONSENSUS_VIOLATION_REL_TOL, eval_params.FeasibilityRatio)
    else:
        feasibility_ratio = SEVERE_CONSENSUS_VIOLATION_REL_TOL
    
    violations = 0
    for e in range(n):
        violated = False
        primal_e = primal[e]
        pair_e = pair[e]
        primal_str = str_round(primal_e, 4)
        pair_str = str_round(pair_e, 4)
        if (abs(primal_e) < NEGLIGIBLE_FLOW_ABS_TOL) and (abs(pair_e) < NEGLIGIBLE_FLOW_ABS_TOL):
            violations += 1
            violated = True
        elif not in_consensus(primal_e, pair_e, eval_params.FeasibilityTolerance, feasibility_ratio):
            violations += 1
            violated = True
        if eval_params.PrintReports and violated:
            print(as_warning(f"Edge {e} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
    
    if not eval_params.PrintReports and violations > 0:
        print(as_warning(f'Outer ADMM Consensus Violations: {violations} ({str(round(violations/n*100, 1))}% of pairs)'))


def inner_admm_consensus_test(primal: np.ndarray, pair: np.ndarray, eval_params: TrafficEngineeringLPEvaluationParams):
    T = np.shape(primal)[0]
    assert primal.shape == pair.shape
    
    if eval_params.FeasibilityRatio is not None:
        feasibility_ratio = min(SEVERE_CONSENSUS_VIOLATION_REL_TOL, eval_params.FeasibilityRatio)
    else:
        feasibility_ratio = SEVERE_CONSENSUS_VIOLATION_REL_TOL
    
    violations = 0
    for t in range(T):
        violated = False
        primal_t = primal[t]
        pair_t = pair[t]
        primal_str = str_round(primal_t, 4)
        pair_str = str_round(pair_t, 4)
        if (abs(primal_t) < NEGLIGIBLE_NULL_SPACE_ELEMENT) and (abs(pair_t) < NEGLIGIBLE_NULL_SPACE_ELEMENT):
            violations += 1
            violated = True
        elif not in_consensus(primal_t, pair_t, eval_params.FeasibilityTolerance, feasibility_ratio):
            violations += 1
            violated = True
        if eval_params.PrintReports and violated:
            print(as_warning(f"Axis {t} --> Inner ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
    
    if not eval_params.PrintReports and violations > 0:
        print(as_warning(f'Inner ADMM Consensus Violations: {violations} ({str(round(violations/T*100, 1))})% of pairs'))


def norm_in_consensus(primal: np.ndarray, pair: np.ndarray, ratio: Optional[float]) -> bool:
    primal_norm = np.linalg.norm(primal)
    pair_norm = np.linalg.norm(pair)
    div = min(pair_norm, primal_norm)
    if div < 1e-8:
        return False
    return (abs(primal_norm - pair_norm) / div) < ratio
