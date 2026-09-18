---
name: evaluating
description: Running evaluations or comparing against other baselines
---

# When To Use?
Whenever OnlineTE needs to be compared against something else (be that Gurobi or another TE solver), invoke this skill.

# Workflow
- Define precisely the metric to evaluate (it is not just the objective, i.e. MLU or Max-Flow value), but may also encompass solve time and sensitivty to traffic churn.
- _NEVER_ evaluate on a single matrix alone. Evaluate on a chain of matrices and warm-start to the extent possible between iterations (how friendly a matrix is to warm-starting is not judged by the solver usually)
- We usually use a _regret_ metric, defined as accumulated sum of objective gap against the optimal value over time (usually evaluated in 1-second steps).
- WAN topologies must have significant disparity in link capacities. Never evaluate on topologies with uniform links unless directed by the user. If needed, use available script tools to generate random matrix and topologies.
