import tqdm
import numpy as np
import networkx as nx


def remove_all_cycles(graph: nx.DiGraph, assignments: np.ndarray):
    """
    Iteratively finds and removes all cycles from the assignment matrix A.
    Uses an explicit stack to prevent RecursionError on deep networks.
    """
    m = graph.number_of_nodes()
    n, K = assignments.shape
    edges = list(graph.edges)
    # Use a small epsilon to prevent infinite loops from floating-point inaccuracies
    EPS = 1e-9 
    
    for k in tqdm.tqdm(range(K)):
        # 1. Build an active adjacency list for the current demand k
        # Stored as adj[u] = {v: edge_index} for fast lookups and edge mapping
        adj = {i: {} for i in range(m)}
        for e in range(n):
            if assignments[e][k] > EPS:
                u, v = edges[e]
                adj[u][v] = e
                
        while True:
            cycle = None
            visited_global = set()
            
            # 2. Iterative DFS to find a cycle
            for start_node in range(m):
                if start_node in visited_global or not adj[start_node]:
                    continue
                    
                # Stack stores tuples of: (current_node, list_of_active_neighbors)
                stack = [(start_node, list(adj[start_node].items()))]
                
                # State trackers for the current DFS path
                in_path = {start_node: 0}  # Maps node to its index in the stack
                edge_path = []             # Tracks the edge indices used in the current path
                
                while stack:
                    u, neighbors = stack[-1]
                    
                    if not neighbors:
                        # Backtrack: Node is fully explored
                        stack.pop()
                        del in_path[u]
                        visited_global.add(u)
                        if edge_path:
                            edge_path.pop()
                        continue
                        
                    v, e_idx = neighbors.pop()
                    
                    if v in in_path:
                        # CYCLE DETECTED!
                        # Extract the loop from the first time we saw 'v' to the current edge
                        start_idx = in_path[v]
                        cycle = edge_path[start_idx:] + [e_idx]
                        break
                    elif v not in visited_global:
                        # Continue DFS forward
                        in_path[v] = len(stack)
                        edge_path.append(e_idx)
                        stack.append((v, list(adj[v].items())))
                        
                if cycle:
                    break  # Break out of the start_node loop to process the cycle
                    
            # 3. If no cycle was found in the entire graph, we are done with demand k
            if not cycle:
                break
                
            # 4. Cycle Elimination
            # Find the smallest flow on the cycle
            min_flow = min(assignments[e][k] for e in cycle)
            
            # Subtract this flow from all edges in the cycle
            for e in cycle:
                assignments[e][k] -= min_flow
                
                # If an edge's flow drops to zero, prune it from the active subgraph
                if assignments[e][k] <= EPS:
                    assignments[e][k] = 0.0
                    u, v = edges[e]
                    if v in adj[u]:
                        del adj[u][v]
                        
    return assignments
