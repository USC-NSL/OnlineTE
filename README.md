> **Note To The SIGCOMM AE Committee:** The branch `sigcomm-ae` has been prepared for you. This main branch will undergo changes in the coming weeks (including things that are not directly related to the paper).

# OnlineTE

OnlineTE is a framework for solving Traffic Engineering problems (essentially, classes of Linear Programs) at very large scale and in a distributed manner.

The problems are split and solved concurrently over multiple nodes (which are supposed to be individual network switches), with OnlineTE handling the communication and algorithmic work transparently.

# Environment Requirements

OnlineTE requires a C++17 compiler with OpenMP support. Most scripts (in particular those under the `ansible` directory) expect Python use through a virtual environment named `.onlinete-venv`.

After installing `requirements.txt` in `.onlinete-venv`, build the native kernels from the repository root:
```sh
python -m pip install -e .
```

Docker and SPHERE/Ansible provisioning perform this build automatically.

# Structure

```
DistributedTE
  |
  +---- ansible             Ansible scripts for use with the SPHERE testbed
  |
  +---- benchmarks          Benchmarks for different formulations over arbirary networks
  |
  +---- native              C++ kernels for path-based operations
  |
  +---- protos              ProtoBuff definitions
  |
  +---- te                  All TE problem formulations and algorithm definitions
  |
  +---- tests               Tests
  |
  +---- topologies          Utilities for working with topologies
  |
  +---- utils               Misc. utilities (logging, making things pretty, etc.)
```
