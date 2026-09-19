---
name: grpc-protos
description: Writing/modifying gRPC protocol buffers
---

# When To Use?
When implementing or modifying gRPC protocol buffers for our communication backend.

# Workflow
- gRPC protocol buffers live under `protos`. A custom script for protobuff compile exists under `protos/__init__.py`. Add the associated proto and its dependencies and then issue `python protos/__init__.py` to compile.
- Modifications to `core.proto` (the core nested ADMM message schema) should be raised to the user before proceeding.
- Different solvers have different solver paramters. Add them to `solver_params.proto`, no need to be fancy here.

