from typing import Type
from te.traffic_models.base import TrafficMatrixBase, _MODELS


def get_traffic_model(name: str) -> Type[TrafficMatrixBase]:
    global _MODELS

    assert name in _MODELS, f'No traffic model `{name}` has been registered'

    return _MODELS[name]


def list_traffic_models():
    return list(_MODELS.keys())
