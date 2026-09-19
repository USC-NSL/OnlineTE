# OnlineTE

This project implements a high performance, distributed solver for Traffic Engineering (TE) problems (currently targeted at Wide Area Neworks (WANs)).
The goal of distribution here is not to just solve the problem quickly, but to also make sure that upon convergence, all switches know the final answer immediately and can program locally.

# Software Stack

- Most logic is expressed in Python. This includes input/output scripting, problem creation, logging, etc.
- For optimizations (linear or quadratic), we can backend to Gurobi or OR-tools (in particular, PDLP).
    - Custom solvers may be implemented, but generally discouraged unless necessary.
    - The only exception here are for worker nodes (which are just abstractions for network switches). These devices are resource constrained (as a rule of thumb, take 4 cores at 2.0 GHz and 16 GB of RAM at most). In such cases, we have no choice but to implement efficient solvers directly.
- Low level optimizations are permited in one of two formats:
    - Optimized JIT Python via Numba
    - Custom C++ code using `nanobind`
    The latter is prefered. JIT overhead is quite annoying.
- For GPUs, `cupy` is currently the only available choice. GPU implementation is assumed to be unnecessary unless explicitly mentioned by the user.
- Communication is handled primarily using asynchronous gRPC directly via its Python API.

# Toolchain Commands and Formats

- Unless specified otherwise by the user, Python must be used through a virtual environment on the root of the project named `.onlinete-venv`. If needed, create this ENV using `python3 -m venv .onlinete-venv` from the root of the repo.
- It is usually expected to invoke code as modules (e.g. `python -m benchmarks.edge_based_centralized --help`).

# Boundaries

## NEVER
- Commit secrets or huge result files
- On TE coordinator nodes, full knowledge of the demand matrix on the switch side _MUST NEVER BE REQUIRED_, unless for sanity checks at the end of the solution. The coordinator is allowed to generate and send matrices to the workers, but not actually calculate anything from them during the solve procedure.

## ASK
- For any algorithmic modifications or step size heuristics
- If any test or sanity checks fails 3 times in a row for a single user input while evaluating, stop and notify the user with full description of the failure mode
- Ask anytime a test or sanity check needs to be relaxed or skipped (e.g. if the overhead is too much)
- When determining new dependencies are needed
- If a necessary change is required to the nested ADMM structure

## ALWAYS
- Provide a one line summary of the in-context plan before any implementation effort.
- Try to minimize repeated code. If aggregation is helpful, stop and ask the user for approval before refactoring.

# Skills

All skills exist under `.skills` on the root of the repository.
- `grpc-protos` for writing/modifying gRPC protocol buffers
- `comm-backends` for writing/modifying communication backends
- `te-solvers` for writing/modifying TE solvers
- `evaluating` for writing/modifying TE evaluations
