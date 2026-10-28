# Solver Algorithms

Each _algorithm_ defines both a solver and a specific formulation that it works well with. The bulk of our code describes the solver process.
```
algorithms
  |
  +---- communication       The communication backends for workign with multiple worker nodes
  |
  +---- formulations        Different TE problem formulations and their solvers
  |
  +---- sub_algorithms      These are utility algorithms that we use
  |
  +---- base.py             Contains ABC definitions for our TE problems, solver parameters, and
  |                         many more. Almost everything inherits its structure from these.
  |
  +---- utils.py            Algorithm utility definitions and helper functions to quickly define
                            a problem and send it to the solvers
```