import time
import numpy as np
import gurobipy as gp
import scipy.optimize as scipopt
from typing import Tuple, List
from gurobipy import quicksum
from scipy.linalg import null_space
from te.algorithms.sub_algorithms.dykstra import dykstra_proj


def gurobi_test_primal(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, 
                       env: gp.Env) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
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
    return out, np.array([c.Pi for c in constraints]), model.ObjVal, model.ObjBound, model.Runtime


def gurobi_test_dual(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, 
                     env: gp.Env) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    This solves the QP:
        maximize -0.5 || n.T lambda_k ||^2 - lambda_k^T (x_0 + n @ c)
            s.t. 0 <= lambda_k
    Using Gurobi (again with barrier).
    The model for this can take a very long time to make, as the objective has to be
    expanded and added term by term.
    This is the dual of the previous problem.
    """
    
    num_edges, null_dim = np.shape(n)

    model = gp.Model('DualQP', env=env)
    model.Params.Method = gp.GRB.METHOD_BARRIER
    model.Params.BarConvTol = 1e-6
    model.Params.FeasibilityTol = 1e-6
    model.Params.Presolve = 0

    lambda_k = model.addVars(num_edges, lb=-float('inf'), vtype=gp.GRB.CONTINUOUS)

    big_c = x_0 + n @ c

    obj: gp.QuadExpr = gp.QuadExpr()
    for e in range(num_edges):
        for t in range(null_dim):
            obj.addTerms(-0.5 * n[e, t]**2, lambda_k[e], lambda_k[e])
            for e_prime in range(e+1, num_edges):
                obj.addTerms(-n[e, t] * n[e_prime, t], lambda_k[e], lambda_k[e_prime])
        obj.addTerms(-big_c[e], lambda_k[e])
    
    model.setObjective(obj, sense=gp.GRB.MAXIMIZE)

    constraints: List[gp.Constr] = [
        model.addConstr(
            0 <= lambda_k[e]
        ) for e in range(num_edges)
    ]

    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL

    out = np.zeros((num_edges,))
    for e in range(num_edges):
        out[e] = lambda_k[e].X
    return out, np.array([c.Pi for c in constraints]), model.ObjBound, model.ObjVal, model.Runtime


def gurobi_test_nnls(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, 
                     env: gp.Env) -> Tuple[np.ndarray, float, float, float]:
    """
    This solves the QP:
        maximize - 0.5 * || n.T lambda_k - b ||^2 + 0.5 * || b ||^2
            s.t. 0 <= lambda_k
    Where:
            b := -n.T (n @ n.T)^-1 (x_0 + n @ c)
    This problem is equivalent to the previous dual problem (and so the
    same considerations that we had above, applies to this one too).
    """
    
    num_edges, null_dim = np.shape(n)

    model = gp.Model('NNLS', env=env)
    model.Params.Method = gp.GRB.METHOD_BARRIER
    model.Params.BarConvTol = 1e-6
    model.Params.FeasibilityTol = 1e-6
    model.Params.Presolve = 0

    lambda_k = model.addVars(num_edges, lb=0, vtype=gp.GRB.CONTINUOUS)

    b = -n.T @ np.linalg.inv(n @ n.T) @ (x_0 + n @ c)
    print(f'Det = {np.linalg.det(n @ n.T)}')

    obj: gp.QuadExpr = gp.QuadExpr()
    for t in range(null_dim):
        for e in range(num_edges):
            obj.addTerms(-0.5 * n[e, t]**2, lambda_k[e], lambda_k[e])
            for e_prime in range(e+1, num_edges):
                obj.addTerms(-n[e, t] * n[e_prime, t], lambda_k[e], lambda_k[e_prime])
            obj.addTerms(b[t] * n[e, t], lambda_k[e])
    obj.addConstant(0.5 * np.linalg.norm(b)**2)
    
    model.setObjective(obj, sense=gp.GRB.MAXIMIZE)
    model.optimize()
    assert model.Status == gp.GRB.OPTIMAL

    out = np.zeros((num_edges,))
    for e in range(num_edges):
        out[e] = lambda_k[e].X
    return out, model.ObjBound, model.ObjVal, model.Runtime


def get_primal_objective_from_feasible_solution(c: np.ndarray, y_k: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(y_k - c) ** 2

def get_dual_objective_from_feasible_solution(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, lambda_k: np.ndarray) -> float:
    big_c = x_0 + n @ c
    return -0.5 * np.linalg.norm(n.T @ lambda_k) ** 2 - np.dot(lambda_k, big_c)

def get_objective_gap(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, y_k: np.ndarray, lambda_k: np.ndarray) -> float:
    return get_dual_objective_from_feasible_solution(n, c, x_0, lambda_k) - get_primal_objective_from_feasible_solution(c, y_k)


def pgd_test_primal(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, num_iters: int, gamma: float) -> Tuple[np.ndarray, float]:
    """
    Simple Projected Gradient Descent (PGD) on the primal problem. Uses Dykstra's projection
    algorithm for the projection step.
    Works quite bad ...
    """
    
    _, null_dim = np.shape(n)

    Y_k = np.zeros((null_dim,))
    scale_factor = np.sqrt(null_dim)
    i = 0
    start = time.time()
    while i < num_iters:
        Y_k_old = Y_k
        grad = Y_k_old - c
        Y_k, _ = dykstra_proj(x_0, n, Y_k_old - gamma * grad, 1e-6)
        if np.linalg.norm(Y_k - Y_k_old) / scale_factor < 1e-8:
            break
        i += 1
    # print(f"Primal PGD took {i} iterations ({time.time() - start} seconds)")
    return Y_k, time.time() - start


def pgd_test_dual(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, num_iters: int, 
                  gamma: float) -> Tuple[np.ndarray, float]:
    """
    Simple PGD on the dual problem. Projections are very cheap.
    Works OK, but can take a long time.
    """
    
    num_edges, _ = np.shape(n)

    lambda_k = np.zeros((num_edges,))
    nnt = n @ n.T
    big_c = x_0 + n @ c
    i = 0
    scale_factor = np.sqrt(num_edges)
    start = time.time()
    while i < num_iters:
        lambda_k_old = lambda_k
        grad = nnt @ lambda_k_old + big_c
        lambda_k = np.clip(lambda_k_old - gamma * grad, a_min=0, a_max=None)
        if np.linalg.norm(lambda_k - lambda_k_old) / scale_factor < 1e-8:
            break
        i += 1
    # print(f"Dual PGD took {i} iterations ({time.time() - start} seconds)")
    return lambda_k, time.time() - start


def exact_line_search_pgd_test_dual(n: np.ndarray, c: np.ndarray, x_0: np.ndarray, 
                                    num_iters: int) -> Tuple[np.ndarray, float]:
    """
    PGD + Exact line search on the dual problem.
    No longer needs to be told what the step size is.
    Works quite well, does not need too many iterations.
            
            (This is our method of choice)
    """
    
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
    start = time.time()
    while i < num_iters:
        lambda_k_old = lambda_k
        grad = nnt @ lambda_k + big_c
        alpha = get_alpha(lambda_k_old)
        lambda_k = np.clip(lambda_k_old - alpha * grad, a_min=0, a_max=None)
        if np.linalg.norm(lambda_k - lambda_k_old) / scale_factor < 1e-8:
            break
        i += 1
    # print(f"Exact Line Search PGD took {i} iterations ({time.time() - start} seconds)")
    return lambda_k, time.time() - start


def active_set_dual_test(n: np.ndarray, c: np.ndarray, x_0: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Active set method over a non-negative least squares problem.
    The implementation just calls `scipy.optimize.nnls`.
    """

    nnt = n @ n.T
    B = n.T @ np.linalg.inv(nnt)
    b = -B @ (x_0 + n @ c)

    start = time.time()
    lambda_k, _ = scipopt.nnls(A=n.T, b=b, maxiter=10000)
    return lambda_k, time.time() - start


if __name__ == '__main__':
    env = gp.Env()
    env.start()
    
    number_of_edges = 100
    number_of_nodes = 50
    seed = 12345
    rng = np.random.default_rng(seed)
    m = rng.random((number_of_nodes, number_of_edges))
    n: np.ndarray = null_space(m)
    assert n.shape[-1] == number_of_edges - number_of_nodes
    assert np.linalg.det(n @ n.T) > 0
    _, null_dim = np.shape(n)
    c = rng.random((null_dim,))
    x_0 = rng.random((number_of_edges,))

    # These will be our Gurobi baselines (one solves the primal, the other the dual)
    primal, dual, optimal_primal_obj, optimal_dual_obj, primal_runtime = gurobi_test_primal(n, c, x_0, env)
    kkt_dual, kkt_primal, kkt_optimal_primal_obj, kkt_optimal_dual_obj, kkt_dual_runtime = gurobi_test_dual(n, c, x_0, env)

    # Get dual solution by solving the dual problem as an instance of NNLS
    nnls_dual, nnls_optimal_primal_obj, nnls_optimal_dual_obj, nnls_dual_runtime = gurobi_test_nnls(n, c, x_0, env)
    crossover_nnls_primal = c + n.T @ nnls_dual
    projected_crossover_nnls_primal, _ = dykstra_proj(x_0, n, crossover_nnls_primal, feasibility_tol=1e-6)
    
    # Use simple PGD on primal (there is no way to get the dual as-is)
    pgd_primal, pgd_primal_runtime = pgd_test_primal(n, c, x_0, 1000, 0.1)
    
    # Use simple PGD on dual and crossover to primal (both feasible and infeasible)
    pgd_dual, pgd_dual_runtime = pgd_test_dual(n, c, x_0, 1000, 0.1)
    crossover_pgd_primal = c + n.T @ pgd_dual
    projected_crossover_pgd_primal, _ = dykstra_proj(x_0, n, crossover_pgd_primal, feasibility_tol=1e-6)
    
    # Use PGD with exact line search on dual and crossover to primal (both feasible and infeasible)
    exact_search_pgd_dual, exact_search_pgd_dual_runtime = exact_line_search_pgd_test_dual(n, c, x_0, 1000)
    crossover_exact_search_pgd_primal = c + n.T @ exact_search_pgd_dual
    projected_crossover_exact_search_pgd_primal, _ = dykstra_proj(x_0, n, crossover_exact_search_pgd_primal, feasibility_tol=1e-6)

    # Finally, the active set method over the dual
    active_set_dual, active_set_dual_runtime = active_set_dual_test(n, c, x_0)
    crossover_active_set_dual = c + n.T @ active_set_dual
    projected_crossover_active_set_dual, _ = dykstra_proj(x_0, n, crossover_active_set_dual, feasibility_tol=1e-6)

    print("="*10 + " RUNTIMES " + "="*10)
    print(f"Gurobi primal: {str(round(primal_runtime, 3))}")
    print(f"Gurobi dual: {str(round(kkt_dual_runtime, 3))}")
    print(f"Gurobi NNLS: {str(round(nnls_dual_runtime, 3))}")
    print(f"PGD: {str(round(pgd_primal_runtime, 3))}")
    print(f"KKT-PGD: {str(round(exact_search_pgd_dual_runtime, 3))}")
    print(f"Dual Acitve-Set: {str(round(active_set_dual_runtime, 3))}")

    print("="*10 + " DUALITY GAPS " + "="*10)
    print(f"Gurobi primal gap: {optimal_primal_obj - optimal_dual_obj}")
    print(f"Gurobi dual gap: {kkt_optimal_primal_obj - kkt_optimal_dual_obj}")
    print(f"Gurobi NNLS gap: {nnls_optimal_primal_obj - nnls_optimal_dual_obj}")
    print(f"Primal PGD gap: < DON'T KNOW! >")
    print(f"Dual PGD gap: {get_objective_gap(n, c, x_0, crossover_pgd_primal, pgd_dual)}")
    print(f"Exact line search PGD gap: {get_objective_gap(n, c, x_0, crossover_exact_search_pgd_primal, exact_search_pgd_dual)}")
    print(f"Dual Active-Set gap: {get_objective_gap(n, c, x_0, crossover_active_set_dual, active_set_dual)}")

    print("="*10 + " PRIMAL OBJECTIVE GAPS " + "="*10)
    print(f"Gurobi/KKT gap: {np.abs(kkt_optimal_primal_obj - optimal_primal_obj)}")
    print(f"Gurobi/NNLS gap: {np.abs(nnls_optimal_primal_obj - optimal_primal_obj)}")
    print(f"Gurobi/Primal-PGD gap: {np.abs(get_primal_objective_from_feasible_solution(c, pgd_primal) - optimal_primal_obj)}")
    print(f"Gurobi/Dual-PGD gap: {np.abs(get_primal_objective_from_feasible_solution(c, crossover_pgd_primal) - optimal_primal_obj)}")
    print(f"Gurobi/Exact-PGD gap: {np.abs(get_primal_objective_from_feasible_solution(c, crossover_exact_search_pgd_primal) - optimal_primal_obj)}")
    print(f"Gurobi/Dual Active-Set gap: {np.abs(get_primal_objective_from_feasible_solution(c, crossover_active_set_dual) - optimal_primal_obj)}")

    print("="*10 + " PRIMAL SOLUTION GAPS " + "="*10)
    # print(f"Gurobi/Crossover-KKT solution gap: {np.linalg.norm(crossover_kkt_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Crossover-NNLS solution gap: {np.linalg.norm(crossover_nnls_primal - primal) / np.sqrt(null_dim)}")
    # print(f"Gurobi/Projected-Crossover-KKT solution gap: {np.linalg.norm(projected_crossover_kkt_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/PGD solution gap: {np.linalg.norm(pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Crossover-PGD solution gap: {np.linalg.norm(crossover_pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Projected-Crossover-PGD solution gap: {np.linalg.norm(projected_crossover_pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Crossover-Exact-PGD solution gap: {np.linalg.norm(crossover_exact_search_pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Projected-Crossover-Exact-PGD solution gap: {np.linalg.norm(projected_crossover_exact_search_pgd_primal - primal) / np.sqrt(null_dim)}")
    print(f"Gurobi/Projected-Crossover-Active-Set solution gap: {np.linalg.norm(projected_crossover_active_set_dual - primal) / np.sqrt(null_dim)}")

    print("="*10 + " DUAL SOLUTION GAPS " + "="*10)
    print(f"Gurobi/KKT-Dual solution gap: {np.linalg.norm(kkt_dual - dual) / np.sqrt(number_of_edges)}")
    print(f"Gurobi/NNLS-Dual solution gap: {np.linalg.norm(nnls_dual - dual) / np.sqrt(number_of_edges)}")
    print(f"Gurobi/PGD solution gap: {np.linalg.norm(pgd_dual - dual) / np.sqrt(number_of_edges)}")
    print(f"Gurobi/Exact-PGD solution gap: {np.linalg.norm(exact_search_pgd_dual - dual) / np.sqrt(number_of_edges)}")
    print(f"Gurobi/Active-Set solution gap: {np.linalg.norm(active_set_dual - dual) / np.sqrt(number_of_edges)}")
    
    print("="*10 + " PRIMAL INFEASIBILITIES " + "="*10)
    # print(f"KKT crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ crossover_kkt_primal, a_min=None, a_max=0.0))}")
    print(f"NNLS crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ crossover_nnls_primal, a_min=None, a_max=0.0))}")
    # print(f"Projected KKT crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ projected_crossover_kkt_primal, a_min=None, a_max=0.0))}")
    print(f"PGD infeasibility: {np.linalg.norm(np.clip(x_0 + n @ pgd_primal, a_min=None, a_max=0.0))}")
    print(f"KKT-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ crossover_pgd_primal, a_min=None, a_max=0.0))}")
    print(f"Projected-KKT-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ projected_crossover_pgd_primal, a_min=None, a_max=0.0))}")
    print(f"Exact-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ crossover_exact_search_pgd_primal, a_min=None, a_max=0.0))}")
    print(f"Projected-Exact-PGD crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ projected_crossover_exact_search_pgd_primal, a_min=None, a_max=0.0))}")
    print(f"Active-Set crossover infeasibility: {np.linalg.norm(np.clip(x_0 + n @ projected_crossover_active_set_dual, a_min=None, a_max=0.0))}")

    env.close()
