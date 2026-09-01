import numpy as np
from typing import Optional, Union, Tuple
from array_utils.cpu.types import *
from te.traffic_models.base import commodity_source_destination_lists


def get_feasible_flow_assignment(
    adjacency_matrix: CPUArray,
    commodity_id_start: Optional[int] = 0,
    commodity_id_end: Optional[int] = None,
    demands: Optional[CPUArray] = None,
) -> Union[CPUArray, Tuple[CPUArray, CPUArray]]:
    """
    Create an assignment that satisfies all demands by taking the least
    squares solution to the demand constraints.

    When `demands' is 'None', this returns pseudo-inverse
    basis for calculating the result quickly in case demands change.
    While this eats double the memory, it is worthwhile since finding
    the pseudo-inverse hides a SVD operation which is quite slow and we
    will be calling this function on the switches.

    When the basis is known, a new feasible solution can be calculated
    by just multiplying the demand vector with the basis.
    """
    M, _ = adjacency_matrix.shape
    pinv = np.linalg.pinv(adjacency_matrix)
    sources, destinations = commodity_source_destination_lists(
        num_nodes=M,
        inclusive_start=commodity_id_start,
        exclusive_end=commodity_id_end
    )
    basis = cpu_array(pinv[:, sources] - pinv[:, destinations])
    if demands is not None:
        return demands * basis
    return basis
