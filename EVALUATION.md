# Generating Paths
For our path-based solvers, paths must be pre-computed for the solvers to use. Under `te/path_providers`, we have many utilities for generating paths.

For AE, we use `Cogentco`, a medium sized topology, for testing, as large topologies will take very long to fully test on just a single machine.

We default to 16 shortest paths in all settings for simplicity. To generate the paths, do:
```bash
python -m te.path_providers Cogentco 16 SHORTEST_PATH
```
You can also do this for any other topology in the Zoo.
This generates a `path_Cogentco.pkl` file which can be passed to solvers.

## The Baseline
> This needs Gurobi

Generate a baseline to check against:
```bash
python -m benchmarks.path_based_centralized --config config/gurobi/path.yaml
```

This will take a bit, but will eventually generate a trace file, `Cogentco_trace_MLU.txt`, which lists the optimal objective for each matrix. This is what we will use to validate against (e.g. to compute regret).

## Running `OnlineTE`
Simply:
```bash
# LOCAL RUN
python -m benchmarks.path_based_distributed --config config/onlinete/path_local.yaml
```
Or:
```bash
# SPHERE RUN
python -m benchmarks.path_based_distributed --config config/onlinete/path_sphere.yaml
```
You must see something like the following in the beginning:
```sh
======================= SOLVING WITH: Path Based OnlineTE ========================
==================================== TM 1/20 =====================================
Total demand: 196254.3
Cold Start.:  132 iters [02:12,  1.00s/it] , Cont. Util.=23.0867, Net. Util.=23.5772, Obj. Gap=0.2019, Outer Step.=1.00
```
Here:
- `Cont. Util.` is the controller side utilization of the network and `Net. Util.` is the _actual_ network side utilization of the current solution. The two may not be in consensus, but as the algorithm progresses, the two must eventually agree and reach the optimal MLU.
- `Gap` is a measure of objective gap that we use for our stopping criterion. When the gap is small enough (currently, around 1 percent), `OnlineTE` stops.

> **The first matrix will take a long time to solve**.
> This is expected (this is the _Cold Start_ of `OnlineTE`), where we start from a blank slate and must build the correct state among the switches (in particular, the ADMM dual variables and switch-side dual variables).
>
> You may notice that even though the utilization is very close to the optimal value, `OnlineTE` refuses to stop. This is not unique to `OnlineTE` and Gurobi does the same, as the optimality gap isn't small enough yet which means that the dual variables are not reliable. `OnlineTE` requires these variables to be accurate so that it can warm-start correctly on next iterations.

Eventually, Cold Sart will finish and we can proceed to our next matrices. For these, _universal_ observation must be that in the majority of the cases, `OnlineTE` will only iterate a few times before declaring optimality (although, in a few cases, it will still be delayed, but convergence for the next iteration is still unaffected. We discuss these cases in the paper).

After each iteration from, `OnlineTE` prints the gap with its reference solution for checking. It should print something like:
```
Objective gap to reference: 0.01%
```
This gap must be quite small (less that 1 percent).

As an example, the first few lines should be something like the following:
```
Solved in 40.112 seconds. Objective Value: 23.3052
Objective gap to reference: 0.02%
==================================== TM 2/20 =====================================
Total demand: 197490.1
  6%|█▉                                  | 11/200 [00:02<00:39,  4.74it/s, Cont. Util.
Crossed the convergance bound. Breaking early ...
Solved in 2.493 seconds. Objective Value: 23.4823
Objective gap to reference: 0.05%
==================================== TM 3/20 =====================================
Total demand: 199024.2
 12%|████▌                               | 25/200 [00:05<00:36,  4.83it/s, Cont. Util.
Crossed the convergance bound. Breaking early ...
Solved in 5.334 seconds. Objective Value: 23.6567
Objective gap to reference: 0.04%
```
Pay attention to the solve time and the objective gap to reference.

## Edge-based Evaluation

`OnlineTE` also provides an edge-based solver (to our knowledge, the first demonstration of a feasible edge-based solver).

> **Note**: If testing on SPHERE, be sure to select the edge-based solver by setting an environment variable: `export SOLVER_TYPE=edge`.

For this test, we use `Interoute`, a smaller topology, as generating the baseline with Gurobi for the edge-based setting will take very long (a few hours).

To generate the baseline:
```bash
python -m benchmarks.edge_based_centralized --config config/gurobi/edge.yaml
```

This alone takes some time (warm-starting in this setting is very hard, even with Dual Simplex). To evaluate `OnlineTE` in this setting, simply do:
```bash
# LOCAL RUN
python -m benchmarks.edge_based_distributed --config config/onlinete/edge_sphere.yaml
# SPHERE RUN
python -m benchmarks.edge_based_distributed --config config/onlinete/edge_sphere.yaml
```