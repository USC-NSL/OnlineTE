> **Note To The SIGCOMM AE Committee:** The branch `sigcomm-ae` has been prepared for you. This main branch will undergo changes in the coming weeks (including things that are not directly related to the paper).

# Distributed TE Problems

A (hoperfully generic, in time) framrwork for implementation of Traffic Engineering (TE) problems (mostly MLU) over a cluster of workers. The framework handles both the problem formulation, and its solver procedure, alongside code to check each solution.

At times when we see no need to implement a solver for a specific problem, we let Gurobi handle it.

```
DistributedTE
  |
  +---- ansible             Ansible scripts for use with the SPHERE testbed
  |
  +---- benchmarks          Benchmarks for different formulations over arbirary networks
  |
  +---- notebooks           Some interactive notebooks (mostly for playing around with things)
  |
  +---- protos              ProtoBuff definitions
  |
  +---- results             (In time!) Important results and solution outputs
  |
  +---- te                  All TE problem formulations and abstract definitions
  |
  +---- topologies          Utilities for working with topologies
  |
  +---- utils               Misc. utilities (logging, making things pretty, etc.)
```
