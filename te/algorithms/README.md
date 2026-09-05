# TE Problem Algorithms

Each _algorithm_ defines both a solver and a specific formulation that it works well with. The bulk of our code describes the solver process.
```
algorithms
  |
  +---- communication             Defines the communication backends for distributing message between
  |                               nodes (basically completely on top of gRPC)
  |
  +---- formulations              Different TE problem formulations
  |
  +---- statistics                Utilities for gathering runtime statistics (i.e. how long each part of
  |                               the solver process took, how much memory it used, etc.)
  |
  +---- sub_algorithms            These are utility algorithms that we use (many of which are used to
  |                               check the output of our algorithms to see if they make sense)
  |
  +---- base.py                   Contains ABC definitions for our TE problems, solver parameters, and
  |                               many more. Almost everything inherits its structure from these.
  |
  +---- objective_evalutators.py  Defines simple functions for evaluating objective from solutions
  |                             
  |
  +---- utils.py                  Algorithm utility definitions and helper functions to quickly define
                                  a problem and send it to the solvers.
```