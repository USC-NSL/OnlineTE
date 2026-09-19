---
name: te-solvers
description: Writing new TE solvers on top of the nested ADMM structure
---

# OnlineTE Solvers

## When To Use?
When implementing or modifying a TE solver, follow this workflow.

## Workflow
In general, there should be at least one _coordinator_ node and one or more _worker_ nodes (either they live locally on the same host or are seperated).
- The coordinator must almost always inherit from `TELP` in `te/algorithms/base.py` and implement it.
- The worker node must almost always inherit from `DistributedSolverNodeBase` in `te/algorithms/communication/base.py`.
- The solver paramters must be explicitly declared in a dedicated `solver_params.py` file and inherit from `SolverParams` in `te/algorithms/base.py`.

**Note:** Purely centralized solvers (like Gurobi) need only implement the "coordinator" and inherit from `TELP`. In general, any entity that feed in demand matrices and evaluates a TE solution must inherit from `TELP` and implement it.