import numpy as np
import networkx as nx
from typing import List, Dict
from te.traffic_models.base import Commodity
# Assume 'A' is the n x K assignment matrix
# 'edges' is a list of tuples where edges[e] = (u, v)
# 'demands' is a list of tuples where demands[k] = (source, destination)

def count_all_demand_paths(graph: nx.DiGraph, assignment: np.ndarray, demands: List[Commodity]):
    n, K = assignment.shape
    m = graph.number_of_nodes()
    edges = list(graph.edges)
    assert len(demands) == K
    path_counts = [0] * K
    
    for k, commodity in enumerate(demands):
        s =  commodity.source
        d = commodity.destination
        
        # 1. Build the active adjacency list for demand k
        adj = {i: [] for i in range(m)}
        for e in range(n):
            if assignment[e][k] > 0:       # Edge e is used by demand k
                u, v = edges[e]
                adj[u].append(v)
                
        # 2. DFS with Memoization
        memo = {}
        
        def dfs(u, ls):
            if u in ls:
                return 0
            # Base case: reached the destination
            if u == d:
                return 1
            # Return cached result if already computed
            if u in memo:
                return memo[u]
                
            total_paths = 0
            for v in adj[u]:
                total_paths += dfs(v, ls + [u])
            
            memo[u] = total_paths
            return total_paths
            
        # 3. Store the total paths starting from the source
        path_counts[k] = dfs(s, [])
        
    return path_counts
