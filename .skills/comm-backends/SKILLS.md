---
name: comm-backends
description: Writing/modifying communication backends
---

# When To Use?
The term "communication backend" is used for any substrate that facilitates the exchange of messages between nodes (usually coordinator and workers) so that the solver object is relatively unbothered by such things.
Invoke this skill for changing or implementing any code that handles this.

# Workflow
- All backend implementation must inherit from `CommunicationBackendBase` in `te/algorithms/communication/base.py` either directly or indirectly, and implement it.
- Specialized parents for some nodes are allowed. For example, coordinators usually inherit from `CoordinatorBackendBase` in `te/algorithms/communication/coordinator_backend.py` instead of `CommunicationBackendBase` directly.
- Remember that broadcasts directly between worker nodes are usually faster and easier compared to in-cast towards the coordinator(s) and back. Such operations are allowed (so switches are allowed to "bounce" from the coordinator(s), but there is a cost to that in practice, thus such operations benefit from optimization)