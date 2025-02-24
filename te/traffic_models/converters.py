import numpy as np
from dataclasses import dataclass
from te.traffic_models.base import (
    TrafficMatrixConverterBase, TrafficMatrixConverterParamsBase, TrafficMatrixBase, 
    traffic_matrix_converter, traffic_matrix_converter_param)
from te.traffic_models.models import CustomTrafficMatrix


@traffic_matrix_converter_param('Uniform')
@dataclass
class UniformTrafficMatrixParams(TrafficMatrixConverterParamsBase):
    delta_max: float
    delta_min: float

    def __post_init__(self):
        assert self.delta_max > self.delta_min


@traffic_matrix_converter('Uniform')
class UniformConverter(TrafficMatrixConverterBase):
    """Shift demands by a random value chosen between `delta_max` and `delta_min`"""
    def __init__(self, seed: int, params: UniformTrafficMatrixParams):
        super().__init__(seed)
        self._params = params
        self._rng = np.random.default_rng(seed=seed)
    
    def convert(self, tm: TrafficMatrixBase) -> CustomTrafficMatrix:
        arr = tm.tm
        sample = self._rng.random(size=arr.shape) * (self._params.delta_max - self._params.delta_min) + self._params.delta_min
        return CustomTrafficMatrix(tm=np.clip(arr + sample, a_min=0, a_max=1))
