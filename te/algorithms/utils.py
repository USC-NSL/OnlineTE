import gurobipy
import numpy as np
import networkx as nx
import te.constants
from typing import List, Tuple, Dict, Union
from collections import defaultdict
from te.traffic_models.base import Commodity


def optimize_or_scream(model: gurobipy.Model):
    model.optimize()
    if model.Status != gurobipy.GRB.OPTIMAL:
        raise RuntimeError(f"Optimizing model {model.ModelName} returned non-optimal status: {model.Status}")


def report_commodity_assignments(expected: List[Commodity], actual: List[Tuple[Commodity, Commodity]], unsatisfied: float, verbose: bool = True):
    assert len(expected) == len(actual)

    if verbose:
        print(" "*12 + "{:^10}    {:^20}".format("DESIRED", "ALLOCATED"))
        print("-"*46)
        for inp, out in zip(expected, actual):
            assert inp.source == out[0].source and inp.destination == out[0].destination
            assert inp.source == out[1].source and inp.destination == out[1].destination
            print(
                "{:<4} -> {:<4}".format(inp.source, inp.destination) +
                "{:^10}    {:^7} <--> {:^7}".format(
                    str(np.round(inp.demand, 2)),
                    str(np.round(out[0].demand, 2)),
                    str(np.round(out[1].demand, 2))
                )
            )
    
    if unsatisfied == 0.0:
        print("ALL DEMANDS WERE SATISFIED")
    else:
        print("{:.1f}% OF DEMANDS WERE NOT SATISFIED".format(unsatisfied*100))


def check_centralized_flow_conservation(
        flows: Union[gurobipy.MVar, np.ndarray], graph: nx.DiGraph, commodities: List[Commodity], 
        feasibility_tol: float = te.constants.DEFAULT_FEASIBILITY_TOLERANCE
    ):

    IS_GUROBI_VAR = isinstance(flows, gurobipy.MVar)

    for k, commodity in enumerate(commodities):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand

        flow_out = defaultdict(list)
        flow_in = defaultdict(list)
        for e, edge in enumerate(graph.edges()):
            flow_out[edge[0]].append(flows[e, k].X if IS_GUROBI_VAR else flows[e, k])
            flow_in[edge[1]].append(flows[e, k].X if IS_GUROBI_VAR else flows[e, k])

        for v in graph.nodes():
            fout = sum(flow_out[v])
            fin  = sum(flow_in[v])

            demand_str = str(np.round(DEMAND, 4))
            fout_str = str(np.round(fout, 4))
            fin_str = str(np.round(fin, 4))

            if v == SOURCE:
                assert abs(fout - DEMAND) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand outflow does not hold at source: {fout_str} vs {demand_str}"
                assert abs(fin) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Source receives its own demand! {fin_str}"
            elif v == DESTINATION:
                assert abs(fout) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Destination is leaking demand! {fout_str}"
                assert abs(fin - DEMAND) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand inflow does not hold at destination: {fin_str} vs {demand_str}"
            else:
                assert abs(fout - fin) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Transit demand conservation does not hold: {fin_str} --> {fout_str}"


def check_distributed_flow_conservation(
        flows: List[gurobipy.MVar], graph: nx.DiGraph, out_index_mapping: Dict[Tuple[int, int], int], 
        commodities: List[Commodity], feasibility_tol: float = te.constants.DEFAULT_FEASIBILITY_TOLERANCE
    ):

    for k, commodity in enumerate(commodities):
        SOURCE = commodity.source
        DESTINATION = commodity.destination
        DEMAND = commodity.demand

        flow_out = defaultdict(list)
        flow_in = defaultdict(list)
        for edge in graph.edges():
            v = edge[0]
            i = out_index_mapping[edge]
            flow_out[edge[0]].append(flows[v][i, k].X)
            flow_in[edge[1]].append(flows[v][i, k].X)
        
        for v in graph.nodes():
            fout = sum(flow_out[v])
            fin  = sum(flow_in[v])

            demand_str = str(np.round(DEMAND, 4))
            fout_str = str(np.round(fout, 4))
            fin_str = str(np.round(fin, 4))
        
            if v == SOURCE:
                assert abs(fout - DEMAND) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand outflow does not hold at source: {fout_str} vs {demand_str}"
                assert abs(fin) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Source receives its own demand! {fin_str}"
            elif v == DESTINATION:
                assert abs(fout) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Destination is leaking demand! {fout_str}"
                assert abs(fin - DEMAND) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Demand inflow does not hold at destination: {fin_str} vs {demand_str}"
            else:
                assert abs(fout - fin) < 2*feasibility_tol , \
                    f"Commodity {k}: Node {v} --> Transit demand conservation does not hold: {fin_str} --> {fout_str}"

