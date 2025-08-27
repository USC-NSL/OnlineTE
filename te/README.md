# TE Problem Definitions

The main two components of the TE problem are:
- The algorithm that we run
- The topology and traffic matrix that it receives as input

Here, we define these elements (the topolgy definitions are in `topologies`).
```
 te
  |
  +---- algorithms          Contains different TE formulations and solver algorithms
  |
  +---- solutions           Contains any solution (i.e. flow assignment matrix) that
  |                         we want to save and keep (note that this can get big)
  |
  +---- traffic_models      Different Traffic Matrix (TM) definitions
  |
  +---- constants.py        Contains the definitions of global constants that we'll
                            be using throughout all formulations.
```