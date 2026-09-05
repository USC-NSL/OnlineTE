# Some FAQs

## What is the structure of this repo?
From a high level:
- `topologies` contains utilities for loading topologies and setting link capacities
- `protos` is the protobuf description of our gRPC backend, which we use to communicate between coordinator(s) and switches
- `utils` are mostly logging utilities
- `config` are CMD configs for quickly testing `OnlineTE`
- `benchmarks` are helper scripts for quickly calling different solvers (not just `OnlineTE`)
- `array_utils` contains wrappers for different matrix operations (including on a GPU, but that is outside of the scope of this AE)
- `ansible` contains all that is needed to work on the `SPHERE` testbed

The folder `te` contains the main implementation.
- `path_providers` defines utilities to quickly handle paths and operations with paths.
    - Most importantly, it defines sparse operations on a path mask matrix under `path_providers/sparse_ops.py`. These are crucial to our path-based solvers.
- `traffic_models` defines utilities for loading and handling different traffic matrices, as well as generating different kinds of matrices (the implementations for uniform, bimodal and gravity matrices used in the paper are there)
- `algorithms` defines the actual `OnlineTE` implementation and the exact baselines

## What is a `TMGenerator`?
We do not evaluate on a single traffic matrix. The `TMGenerator` is a base class the defines entire series of traffic matrices (in our case, these are synthetic, but we have a `FileBacked` matrix that can pull from matrices in a series of given paths).

## How is `te/algorithms` structured?
- The `te/algorithms/base.py` is a core part of `OnlineTE`. It describes an ABC for a traffic engineering solver with many basic methods that log and handle callbacks for different things. It also describes the main loop of the solver over multiple traffic matrices (as opposed to just one matrix being solved).
- `communications` describes the communication backend for _all_ of our algorithms. A nice thing about the nested ADMM structure (as we describe in the paper) is that it is mostly agnostic to what happens in the switches, hence the structure of messages remains _exactly_ the same across different objectives or settings.
- `sub_algorithms` are small algorithms for checking solutions or doing projections. Most important algorithms here are in `pgd.py`. These are routines executed by the switches to solve their small switch-side problem.
- `formulations` contains the main solver definition, divorced from how messages are communicated.

## How is `OnlineTE` itself strcutured?

There is always a worker node, `worker.py` and a coordinator, `coordinator.py` and a standalone file for solver paramters.

## Where is the sparse solver for the edge-based setting?

In two places:
- The main implementation is in `te/algorithms/sub_algorithms/lasso.py` which handles the L1 regularizer using shrinkage
- The edge-based worker under `te/algorithms/formulations/edge_based/distributed/worker.py` also defines a `SparseSolver` class that is the main wrapper around the LASSO solver above.

## Where is the Partial Barrier that handles asynchrony for ADMM which you used for the hierarchical solver?

It is in `te/algorithms/communication/grpc/partial_barrier.py`. It can be activated by setting its parameters in the config file. For example:
```yaml
online_te:
    AsyngRPC:
        min_arrival: 10
        max_lag: 2
```
Creates a barrier that needs at least 10 arrivals and tolerates at most 2 iterations of delay before it can unblock.

As for hierarchy itself, we bring up the coordinators manually and impose delays using the deployed `SPHERE` model.

## Where is your main ADMM implementation?

It is in `te/algorithms/sub_algorithms/admm.py`. We use this for the outer ADMM loop, as the inner ADMM loop is distributed over the switches and cannot be written under a single object (the `SharingWrapper` class under that implementation we will depricate soon and replace with something else).

## Does `OnlineTE` call another solver?

On the coordinator(s), `OnlineTE` solves a very lightly constrained QP that scales linearly with the number of edges in the network (regardles of the number of demands). This QP must be solved to a high degree of optimality (if not, the solution oscillates near convergence which is terrible for warm-starting).
For this problem (and _ONLY_ this problem), `OnlineTE` calls an external solver. Gurobi is in our experience too heavy for this and becomes slow, as model updates become a bottleneck.

We have found that Google's PDLP is much better in this regard, and that is the solver that `OnlineTE` employs in the coordinator(s).

You can find the implementations for these "mini-solvers" under `te/algorithms/sub_algorithms/mlu_backends`.

## Are you solving max concurrent flow?

No, you will not find it here (and not in the paper), there are pointers to it in the paper and this repository, but we have left it for future work. Our path-based solver supports Max-Flow and MLU which can be selected with the config file. The same applies to max-min fairness.

## Why is your first path-based iteration sooooo slow?

It is indeed extremely slow, but you should only see that once. The overhead is:
- Worker nodes actually compute and cache paths locally for the first iteration. This is because sending the paths over the wire from the coordinator is too slow for startup. The coordinator also does not even need to know the paths, as such, for the very first time where a worker node comes up, it needs to recalcualte paths as it has nothing on hand.

- The sparse path operations under `te/path_providers/sparse_ops.py` use `Numba` which is based on JIT compilation. The first iteration takes very long by design, but we do cache the generated bytecode for future use, so it should not happen again.

## Why is your path-based solver _slower_ than your edge-based solver??

This is due to a trick that we use to remove flow-conservation constraints in the edge-based solver, which drastically cuts down on the number of constraints in that problem (even less than the path-based case!), as such, besides the large number of variables, edge-based problem has little difficuly and paralelizes beautifully.

The same cannot be said for the path-based solver as its sparse path mask matrices prevent us from using the same trick (see paper appendix for more details).

## Why is your coordinator sending demands to switches? Aren't switches supposed to sample demands?

We are not passing live traffic through the "switches", thus we have to reveal the traffic matrix to the switches externally. The coordinator already has access to the switches via its gRPC backend, so we decided to keep the matrix on the coordinator and emit it to switches between solves.

You can verify for yourself that besides sending the matrix, the coordinator never does anything else with it.
