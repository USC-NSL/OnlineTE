from .base import ControllerCommunicationBackendBase, _BACKENDS
from .. import DistributedADMMControllerRPCParams

from . import synchronous_grpc_backend
from . import asynchronous_grpc_backend


def get_backend(params: DistributedADMMControllerRPCParams) -> ControllerCommunicationBackendBase:
    global _BACKENDS
    name = params.Backends
    assert name in _BACKENDS, f'No communication backend `{name}` has been registered'

    return _BACKENDS[name](params)


def list_backends():
    return list(_BACKENDS.keys())
