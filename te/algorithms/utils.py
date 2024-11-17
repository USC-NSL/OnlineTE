import numpy as np
from typing import List
from te.traffic_models.base import Commodity


def report_commodity_assignments(expected: List[Commodity], actual: List[Commodity], unsatisfied: float, verbose: bool = True):
    assert len(expected) == len(actual)

    if verbose:
        print(" "*12 + "{:^10}    {:^10}".format("DESIRED", "ALLOCATED"))
        print("-"*36)
        for inp, out in zip(expected, actual):
            assert inp.source == out.source and inp.destination == out.destination
            print(
                "{:<4} -> {:<4}".format(inp.source, inp.destination) +
                "{:^10}    {:^10}".format(str(np.round(inp.demand, 2)), str(np.round(out.demand, 2)))
            )
    
    if unsatisfied == 0.0:
        print("ALL DEMANDS WERE SATISFIED")
    else:
        print("{:.1f}% OF DEMANDS WERE NOT SATISFIED".format(unsatisfied*100))
