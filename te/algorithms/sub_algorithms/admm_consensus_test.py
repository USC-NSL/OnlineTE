import numpy as np
from typing import Optional
from utils.logging import as_warning
from te.algorithms.utils import is_satisfied


NEGLIGIBLE_FLOW_ABS_TOL = 1e-3
NEGLIGIBLE_NULL_SPACE_ELEMENT = 5e-3
SEVERE_CONSENSUS_VIOLATION_REL_TOL = 5e-2


def outer_admm_consensus_test(primal: np.ndarray, pair: np.ndarray, feasibility_tol: Optional[float] = None, 
                              feasibility_ratio: Optional[float] = None, report: bool = False):
    n = np.shape(primal)[0]
    assert primal.shape == pair.shape
    
    if feasibility_ratio is not None:
        feasibility_ratio = min(SEVERE_CONSENSUS_VIOLATION_REL_TOL, feasibility_ratio)
    else:
        feasibility_ratio = SEVERE_CONSENSUS_VIOLATION_REL_TOL
    
    violations = 0
    for e in range(n):
        violated = False
        primal_e = primal[e]
        pair_e = pair[e]
        primal_str = str(np.round(primal_e, 4))
        pair_str = str(np.round(pair_e, 4))
        if (abs(primal_e) < NEGLIGIBLE_FLOW_ABS_TOL) and (abs(pair_e) < NEGLIGIBLE_FLOW_ABS_TOL):
            violations += 1
            violated = True
        elif not is_satisfied(primal_e, pair_e, feasibility_tol, feasibility_ratio):
            violations += 1
            violated = True
        if report and violated:
            print(as_warning(f"Edge {e} --> Outer ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
    
    if not report and violations > 0:
        print(as_warning(f'Outer ADMM Consensus Violations: {violations} ({str(round(violations/n*100, 1))}% of pairs)'))


def inner_admm_consensus_test(primal: np.ndarray, pair: np.ndarray, feasibility_tol: Optional[float] = None, 
                              feasibility_ratio: Optional[float] = None, report: bool = False):
    T = np.shape(primal)[0]
    assert primal.shape == pair.shape
    
    if feasibility_ratio is not None:
        feasibility_ratio = min(SEVERE_CONSENSUS_VIOLATION_REL_TOL, feasibility_ratio)
    else:
        feasibility_ratio = SEVERE_CONSENSUS_VIOLATION_REL_TOL
    
    violations = 0
    for t in range(T):
        violated = False
        primal_t = primal[t]
        pair_t = pair[t]
        primal_str = str(np.round(primal_t, 4))
        pair_str = str(np.round(pair_t, 4))
        if (abs(primal_t) < NEGLIGIBLE_NULL_SPACE_ELEMENT) and (abs(pair_t) < NEGLIGIBLE_NULL_SPACE_ELEMENT):
            violations += 1
            violated = True
        elif not is_satisfied(primal_t, pair_t, feasibility_tol, feasibility_ratio):
            violations += 1
            violated = True
        if report and violated:
            print(as_warning(f"Axis {t} --> Inner ADMM pairing is not in consensus with primal variable: {primal_str} vs {pair_str}"))
    
    if not report and violations > 0:
        print(as_warning(f'Inner ADMM Consensus Violations: {violations} ({str(round(violations/T*100, 1))})% of pairs'))

