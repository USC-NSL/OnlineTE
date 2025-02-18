import numpy as np
import networkx as nx
from dataclasses import dataclass
from te.traffic_models.base import TrafficMatrixBase, traffic_matrix


"""
Certain distributions depend very much on the graph that we use
for the traffic matrix itself.
For these distributions, we need to include the graph object (i.e. a
`nx.DiGraph` object) with the parameters.
"""


@dataclass
class UniformTrafficMatrixParams:
    n: int
    min: float
    max: float


@dataclass
class ExponentialTrafficMatrixParams:
    graph: nx.DiGraph
    beta: float
    gamma: float


class CustomTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: np.ndarray = None, seed: int = None, params=None):
        self.tm = tm
    
    def _make_tm(self):
        pass

    @property
    def name(self) -> str:
        return "Custom"
    
    @property
    def type(self) -> str:
        return "Custom"


@traffic_matrix
class UniformTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: np.ndarray = None, seed: int=None, params: UniformTrafficMatrixParams=None):
        super().__init__(tm, seed, params)

    @classmethod
    def type(cls) -> str:
        return 'Uniform'
    
    @property
    def name(self) -> str:
        PARAMS: UniformTrafficMatrixParams = self.params

        return f'Uniform_{PARAMS.n}_{PARAMS.min}_{PARAMS.max}'
    
    def _make_tm(self):
        PARAMS: UniformTrafficMatrixParams = self.params

        tm = self._rng.random(
            size=(PARAMS.n, PARAMS.n),
            dtype=np.float32
        ) * (PARAMS.max - PARAMS.min) + PARAMS.min
        np.fill_diagonal(tm, 0.0)

        self.tm = tm
    

@traffic_matrix
class ExponentialTrafficMatrix(TrafficMatrixBase):
    def __init__(self, tm: np.ndarray = None, seed: int = None, params: ExponentialTrafficMatrixParams=None):
        super().__init__(tm, seed, params)
    
    @classmethod
    def type(cls) -> str:
        return 'Exponential'
    
    @property
    def name(self) -> str:
        PARAMS: ExponentialTrafficMatrixParams = self.params
        GRAPH: nx.DiGraph = PARAMS.graph

        return f'Exponential_G({len(GRAPH.nodes)},{len(GRAPH.edges)})_{PARAMS.beta}_{PARAMS.gamma}'
    
    def _make_tm(self):
        PARAMS: ExponentialTrafficMatrixParams = self.params
        GRAPH: nx.DiGraph = PARAMS.graph
        NUMBER_OF_NODES = len(GRAPH.nodes)

        distances = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES), dtype=np.int32)
        for src, dist_dict in nx.shortest_path_length(GRAPH):
            for target, dist in dist_dict.items():
                distances[src, target] = dist

        tm = np.array([
            [self._rng.exponential(PARAMS.beta * (PARAMS.gamma**dist)) for dist in row] \
                for row in distances
            ], dtype=np.float32)
        np.fill_diagonal(self.tm, 0.0)

        self.tm = tm
