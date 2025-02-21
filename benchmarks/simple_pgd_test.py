import time
import numpy as np
import gurobipy as gp
from typing import Tuple, List
from gurobipy import quicksum
from scipy.linalg import null_space
from te.algorithms.utils import dykstra_proj


def gurobi_test_primal(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, env: gp.Env) -> Tuple[np.ndarray, np.ndarray]:
    """
    This solves the QP:
        minimize 0.5 * || X - c ||^2
            s.t. 0 <= x_0 + n @ X
    With Gurobi.
    We'll use Barrier, with no presolve.
    It returns both `X` and the dual variabels associated with
    the constraints.
    We treat this as our baseline.
    """
    
    num_edges, null_dim = np.shape(n)

    model = gp.Model('ExactQP', env=env)
    model.Params.Method = gp.GRB.METHOD_BARRIER
    model.Params.BarConvTol = 1e-6
    model.Params.FeasibilityTol = 1e-6
    model.Params.Presolve = 0

    Y_k = model.addVars(null_dim, lb=-float('inf'), vtype=gp.GRB.CONTINUOUS)

    obj: gp.QuadExpr = gp.QuadExpr()
    for t in range(null_dim):
        y = Y_k[t]
        _c = c[t]
        obj.addTerms(0.5, y, y)
        obj.addTerms(-_c, y)
        obj.addConstant(0.5 * _c ** 2)
    
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    constraints: List[gp.Constr] = [
        model.addConstr(
            0 <= (x_0[e] + quicksum([
                n[e, t] * Y_k[t] for t in range(null_dim)
            ]))
        ) for e in range(num_edges)
    ]
    
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL

    out = np.zeros((null_dim,))
    for t in range(null_dim):
        out[t] = Y_k[t].X
    return out, np.array([c.Pi for c in constraints])


def gurobi_test_dual(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, env: gp.Env) -> np.ndarray:
    """
    This solves the QP:
        maximize -0.5 || n.T lambda_k ||^2 - lambda_k.^T (x_0 + n @ c)
            s.t. 0 <= lambda_k
    Using Gurobi (again with barrier).
    The model for this can take a very long time to make, as the objective has to be
    expanded and added term by term.
    This is the dual of the previous problem, however, we cannot crossover to the
    primal as easily here and so we just output the dual optimal solution.
    """
    
    num_edges, null_dim = np.shape(n)

    model = gp.Model('DualQP', env=env)
    model.Params.Method = gp.GRB.METHOD_BARRIER
    model.Params.BarConvTol = 1e-6
    model.Params.FeasibilityTol = 1e-6
    model.Params.Presolve = 0

    lambda_k = model.addVars(number_of_edges, lb=0, vtype=gp.GRB.CONTINUOUS)

    big_c = x_0 + n @ c

    obj: gp.QuadExpr = gp.QuadExpr()
    for e in range(num_edges):
        for t in range(null_dim):
            obj.addTerms(-0.5 * n[e, t]**2, lambda_k[e], lambda_k[e])
            for e_prime in range(e+1, num_edges):
                obj.addTerms(-n[e, t] * n[e_prime, t], lambda_k[e], lambda_k[e_prime])
        obj.addTerms(-big_c[e], lambda_k[e])
    
    model.setObjective(obj, sense=gp.GRB.MAXIMIZE)
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL

    out = np.zeros((num_edges,))
    for e in range(num_edges):
        out[e] = lambda_k[e].X
    return out


def pgd_test_primal(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, num_iters: int, gamma: float) -> np.ndarray:
    """
    Simple Projected Gradient Descent (PGD) on the primal problem. Uses Dykstra's projection
    algorithm for the projection step.
    Works quite bad ...
    """
    
    start = time.time()
    _, null_dim = np.shape(n)

    Y_k = np.zeros((null_dim,))
    scale_factor = np.sqrt(null_dim)
    i = 0
    while i < num_iters:
        Y_k_old = Y_k
        grad = Y_k_old - c
        Y_k, _ = dykstra_proj(x_0, n, Y_k_old - gamma * grad, 1e-6)
        if np.linalg.norm(Y_k - Y_k_old) / scale_factor < 1e-8:
            break
        i += 1
    print(f"Primal PGD took {i} iterations ({time.time() - start} seconds)")
    return Y_k


def pgd_test_dual(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, num_iters: int, gamma: float) -> np.ndarray:
    """
    Simple PGD on the dual problem. Projections are very cheap.
    Works OK, but can take a long time.
    """
    
    start = time.time()
    num_edges, _ = np.shape(n)

    lambda_k = np.zeros((num_edges,))
    nnt = n @ n.T
    big_c = x_0 + n @ c
    i = 0
    scale_factor = np.sqrt(num_edges)
    while i < num_iters:
        lambda_k_old = lambda_k
        grad = nnt @ lambda_k_old + big_c
        lambda_k = np.clip(lambda_k_old - gamma * grad, a_min=0, a_max=None)
        if np.linalg.norm(lambda_k - lambda_k_old) / scale_factor < 1e-8:
            break
        i += 1
    print(f"Dual PGD took {i} iterations ({time.time() - start} seconds)")
    return lambda_k


def exact_line_search_pgd_test_dual(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, num_iters: int) -> np.ndarray:
    """
    PGD + Exact line search on the dual problem.
    No longer needs to be told what the step size is.
    Works quite well, does not need too many iterations.
            
            (This is our method of choice)
    """
    
    start = time.time()
    num_edges, _ = np.shape(n)

    lambda_k = np.zeros((num_edges,))
    nnt = n @ n.T
    big_c = x_0 + n @ c
    big_lambda = nnt @ big_c
    norm_1 = 0.5 * np.linalg.norm(big_c) ** 2
    norm_2 = np.linalg.norm(n.T @ big_c) ** 2

    def get_alpha(current_lambda):
        norm = np.linalg.norm(n.T @ current_lambda) ** 2
        dot = np.dot(current_lambda, big_lambda)
        return (norm + 1.5 * dot + norm_1) / (norm + norm_2 + 2 * dot)

    i = 0
    scale_factor = np.sqrt(num_edges)
    while i < num_iters:
        lambda_k_old = lambda_k
        grad = nnt @ lambda_k + big_c
        alpha = get_alpha(lambda_k_old)
        lambda_k = np.clip(lambda_k_old - alpha * grad, a_min=0, a_max=None)
        if np.linalg.norm(lambda_k - lambda_k_old) / scale_factor < 1e-8:
            break
        i += 1
    print(f"Exact Line Search PGD took {i} iterations ({time.time() - start} seconds)")
    return lambda_k


if __name__ == '__main__':
    env = gp.Env()
    env.start()
    
    number_of_edges = 1000
    number_of_nodes = 300
    rng = np.random.default_rng(45678)
    m = rng.random((number_of_nodes, number_of_edges))
    n: np.ndarray = null_space(m)
    _, null_dim = np.shape(n)
    c = rng.random((null_dim,))
    x_0 = rng.random((number_of_edges,))

    # These will be our baselines
    primal, dual = gurobi_test_primal(n, c, x_0, env)
    
    # Get dual solution by solving the dual problem exactly and crossover to primal (both feasible and infeasible)
    kkt_dual = gurobi_test_dual(n, c, x_0, env)
    crossover_kkt_primal = c + n.T @ kkt_dual
    projected_crossover_kkt_primal, _ = dykstra_proj(x_0, n, crossover_kkt_primal, feasibility_tol=1e-6)
    
    # Use simple PGD on primal (there is no way to get the dual as-is)
    pgd_primal = pgd_test_primal(n, c, x_0, 1000, 0.1)
    
    # Use simple PGD on dual and crossover to primal (both feasible and infeasible)
    pgd_dual = pgd_test_dual(n, c, x_0, 1000, 0.1)
    crossover_pgd_primal = c + n.T @ pgd_dual
    projected_crossover_pgd_primal, _ = dykstra_proj(x_0, n, crossover_pgd_primal, feasibility_tol=1e-6)
    
    # Use PGD with exact line search on dual and crossover to primal (both feasible and infeasible)
    exact_search_pgd_dual = exact_line_search_pgd_test_dual(n, c, x_0, 1000)
    crossover_exact_search_pgd_primal = c + n.T @ exact_search_pgd_dual
    projected_crossover_exact_search_pgd_primal, _ = dykstra_proj(x_0, n, crossover_exact_search_pgd_primal, feasibility_tol=1e-6)

    print("="*10 + " PRIMAL SOLUTION GAPS " + "="*10)
    print(f"Gurobi/Crossover-KKT solution gap: {np.linalg.norm(crossover_kkt_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Projected-Crossover-KKT solution gap: {np.linalg.norm(projected_crossover_kkt_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/PGD solution gap: {np.linalg.norm(pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Crossover-PGD solution gap: {np.linalg.norm(crossover_pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Projected-Crossover-PGD solution gap: {np.linalg.norm(projected_crossover_pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Crossover-Exact-PGD solution gap: {np.linalg.norm(crossover_exact_search_pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Projected-Crossover-Exact-PGD solution gap: {np.linalg.norm(projected_crossover_exact_search_pgd_primal - primal) / np.sqrt(null_dim)}")

    print("="*10 + " DUAL SOLUTION GAPS " + "="*10)
    print(f"Gurobi/KKT-Dual solution gap: {np.linalg.norm(kkt_dual - dual) / np.sqrt(number_of_edges)}")
    print(f"Gurobi/PGD solution gap: {np.linalg.norm(pgd_dual - dual) / np.sqrt(number_of_edges)}")
    print(f"Gurobi/Exact-PGD solution gap: {np.linalg.norm(exact_search_pgd_dual - dual) / np.sqrt(number_of_edges)}")
    
    print("="*10 + " PRIMAL INFEASIBILITIES " + "="*10)

    print(f"KKT crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ crossover_kkt_primal, a_min=None, a_max=0.0))}")
    print(f"Projected KKT crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ projected_crossover_kkt_primal, a_min=None, a_max=0.0))}")
    print(f"PGD infeasibility: {np.linalg.norm(np.clip(x_0 + n @ pgd_primal, a_min=None, a_max=0.0))}")
    print(f"KKT-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ crossover_pgd_primal, a_min=None, a_max=0.0))}")
    print(f"Projected-KKT-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ projected_crossover_pgd_primal, a_min=None, a_max=0.0))}")
    print(f"Exact-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ crossover_exact_search_pgd_primal, a_min=None, a_max=0.0))}")
    print(f"Projected-Exact-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ projected_crossover_exact_search_pgd_primal, a_min=None, a_max=0.0))}")

    env.close()
