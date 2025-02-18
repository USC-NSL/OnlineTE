import numpy as np
from te.traffic_models.base import TrafficMatrixConverterBase, TrafficMatrixBase
from te.traffic_models.models import CustomTrafficMatrix


class UniformConverter(TrafficMatrixConverterBase):
    """Shift demands by a random value chosen between `delta_max` and `delta_min`"""
    def __init__(self, seed, delta_max: float, delta_min: float):
        super().__init__(seed)
        assert delta_max > delta_min
        self._delta_max = delta_max
        self._delta_min = delta_min
        self._rng = np.random.default_rng(seed=seed)
    
    def convert(self, tm: TrafficMatrixBase) -> CustomTrafficMatrix:
        arr = tm.tm
        sample = self._rng.random(size=arr.shape) * (self._delta_max - self._delta_min) + self._delta_min
        return CustomTrafficMatrix(tm=np.clip(arr + sample, a_min=0, a_max=1))
