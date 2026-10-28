# TE Problem Definitions and Implementation

The main three components of the TE problem are:
- The problem formulation and objective
- The inputs (topology and traffic matrix)
- The solver algorithm

Here, we define these elements (the topolgy definitions are in `topologies`).
```
 te
  |
  +---- algorithms          Contains different TE formulations and solver algorithms
  |
  +---- path_providers      Utilities for working and generating network paths
  |
  +---- traffic_models      Different Traffic Matrix (TM) generators and definitions
  |
  +---- constants.py        Contains the definitions of global constants that we'll
                            be using throughout all formulations
```