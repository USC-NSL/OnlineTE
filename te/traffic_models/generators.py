import pickle
import argparse
import numpy as np
import jsonargparse
import networkx as nx
from io import BufferedReader
from dataclasses import dataclass, field
from typing import Tuple, Optional, Iterator, Callable, List, ClassVar
from .base import *

#### UNIFORM ####

@dataclass(frozen=True)
class UniformTMGeneratorParams(TMGeneratorParams):
    """
    Parameters defining a random, uniform traffic matrix.

    Attributes
    ----------
    min: float
        Minimum admissible demand value
    max: float
        Maximum admissible demand value
    n: int
        Number of nodes
    """
    min: float = 0.0
    max: float = 1.0
    n: int = field(default=0, metadata={'help': argparse.SUPPRESS})
    _type: ClassVar[str] = 'uniform'

    def __post_init__(self):
        assert self.min >= 0 and self.max >= 0 and self.max >= self.min
        assert self.n > 0


class UniformTMGenerator(TMGenerator):
    _TYPE = UniformTMGeneratorParams._type

    def __init__(self, params: UniformTMGeneratorParams):
        super().__init__(params)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: UniformTMGeneratorParams = self.params
        RNG = self.get_rng()
        for _ in range(PARAMS.count):
            tm = RNG.random(
                size=(PARAMS.n, PARAMS.n),
                dtype=PARAMS.dtype
            ) * (PARAMS.max - PARAMS.min) + PARAMS.min

            np.fill_diagonal(tm, 0.0)
            np.multiply(tm, PARAMS.scale_factor, out=tm)
            yield tm

#### EXPONENTIAL ####

@dataclass(frozen=True)
class ExponentialTMGeneratorParams(TMGeneratorParams):
    """
    Parameters defining an exponential traffic matrix.
    The matrix is defined by a mean `beta` and a decay of `gamma`.
    Demand between two nodes a distance of `d` hops apart is
    sampled from `Exp(beta * (gamma)^d)`.

    Attributes
    ----------
    beta: float
        Distribution mean; can be arbitrary positive value.
    gamma: float
        Distribution decay; must be in `(0, 1]`
    graph: nx.DiGraph
        Topology graph object. We need distances between
        nodes, so that is why we need it.
    """
    beta: float = 0.5
    gamma: float = 0.1
    graph: nx.DiGraph = field(default=nx.DiGraph(), metadata={'help': argparse.SUPPRESS})
    _type: ClassVar[str] = 'exponential'

    def __post_init__(self):
        assert self.beta > 0 and self.gamma > 0 and self.gamma <= 1
        assert self.graph.number_of_nodes() > 0


class ExponentialTMGenerator(TMGenerator):
    _TYPE = ExponentialTMGeneratorParams._type

    def __init__(self, params: ExponentialTMGeneratorParams):
        super().__init__(params)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: ExponentialTMGeneratorParams = self.params
        GRAPH: nx.DiGraph = PARAMS.graph
        NUMBER_OF_NODES = len(GRAPH.nodes)
        RNG = self.get_rng()
        for _ in range(self.params.count):
            distances = np.zeros((NUMBER_OF_NODES, NUMBER_OF_NODES), dtype=np.int32)
            for src, dist_dict in nx.shortest_path_length(GRAPH):
                for target, dist in dist_dict.items():
                    distances[src, target] = dist

            tm = np.array([
                [RNG.exponential(PARAMS.beta * (PARAMS.gamma**dist)) for dist in row] \
                    for row in distances
                ], dtype=PARAMS.dtype)

            np.fill_diagonal(tm, 0.0)
            np.multiply(tm, PARAMS.scale_factor, out=tm)
            yield tm

#### BIMODAL ####

@dataclass(frozen=True)
class BimodalTMGeneratorParams(TMGeneratorParams):
    """
    Parameters defining an bimodal traffic matrix.
    A given fraction of the entries are sampled from one uniform distribution,
    and the rest from another one.

    Attributes
    ----------
    n: int
        Number of nodes for our matrix.
    range1: Tuple[float, float]
        First distribution min and max values.
    range2: Tuple[float, float]
        Second distribution min and max values.
    fraction: float
        What fraction of entries belong to the first distribution.
        Must be in `(0, 1)`.
    """
    range1: Tuple[float, float] = (0, 0.2)
    range2: Tuple[float, float] = (0.95, 1.0)
    fraction: float = 0.9
    n: int = field(default=0, metadata={'help': argparse.SUPPRESS})
    _type: ClassVar[str] = 'bimodal'

    def __post_init__(self):
        assert self.range1[0] <= self.range1[1]
        assert self.range2[0] <= self.range2[1]
        assert self.fraction > 0 and self.fraction < 1


class BimodalTMGenerator(TMGenerator):
    _TYPE = BimodalTMGeneratorParams._type

    def __init__(self, params: BimodalTMGeneratorParams):
        super().__init__(params)

    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: BimodalTMGeneratorParams = self.params
        num_nodes = PARAMS.n
        RNG = self.get_rng()
        for _ in range(self.params.count):
            tm = np.zeros((num_nodes, num_nodes), dtype=PARAMS.dtype)
            inds = RNG.choice(
                a=[False, True],
                size=(num_nodes, num_nodes),
                p=[PARAMS.fraction, 1 - PARAMS.fraction]
            )
            tm[inds] = np.random.uniform(PARAMS.range1[0], PARAMS.range1[1], np.sum(inds))
            tm[~inds] = np.random.uniform(PARAMS.range2[0], PARAMS.range2[1], np.sum(~inds))

            np.fill_diagonal(tm, 0.0)
            np.multiply(tm, PARAMS.scale_factor, out=tm)
            yield tm

#### FILE-BACKED ####

@dataclass(frozen=True)
class FilebackedTMGeneratorParams(TMGeneratorParams):
    """
    Parameters defining a traffic matrix backed by a file.

    Attributes
    ----------
    paths: List[str]
        A list of paths to traffic matrices.
    scale: Optional[float]
        If present, the traffic matrix will be divided by this value
        element-wise. If `None`, the matrix is normalized between
        0 and 1 by dividing by the maximum value.
        Defaults to `1.0`.
    loader: Optional[Callable[[BufferedReader], np.ndarray]] = `pickle.load`
        A callable that takes a file object and returns a Numpy array.
        Defaults to `pickle.load`.
    """
    paths: List[str] = field(default_factory=list)
    scale: Optional[float] = 1.0
    loader: Callable[[BufferedReader], np.ndarray] = field(default_factory=lambda: pickle.load, metadata={'help': argparse.SUPPRESS})
    _type: ClassVar[str] = 'file-backed'

    def __post_init__(self):
        assert len(self.paths) > 0


class FilebackedTMGenerator(TMGenerator):
    _TYPE = FilebackedTMGeneratorParams._type

    def __init__(self, params: FilebackedTMGeneratorParams):
        super().__init__(params)

    def __iter__(self) -> Iterator[str]:
        PARAMS: FilebackedTMGeneratorParams = self.params

        counter = 0
        for path in PARAMS.paths:
            if counter >= PARAMS.count:
                break
            counter += 1
            with open(path, "rb") as tm_file:
                tm = PARAMS.loader(tm_file)
                assert isinstance(tm, np.ndarray), 'TM object is not a Numpy array'
                np.fill_diagonal(tm, 0)
                if PARAMS.scale is not None:
                    tm = tm / PARAMS.scale
                else:
                    tm = tm / tm.max()

                np.fill_diagonal(tm, 0.0)
                np.multiply(tm, PARAMS.scale_factor, out=tm)
                yield tm

#### UNIFORM-DRIFT ####

@dataclass(frozen=True)
class UniformDriftTMGeneratorParams(TMGeneratorParams):
    delta_max: float = 1.0
    delta_min: float = 0.0
    initial_tm: np.ndarray = field(default_factory=lambda: np.empty((1,)), metadata={'help': argparse.SUPPRESS})
    _type: ClassVar[str] = 'uniform-drift'

    def __post_init__(self):
        assert self.delta_max > self.delta_min
        assert self.dtype == self.initial_tm.dtype
        shape = self.initial_tm.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        assert np.allclose(np.diag(self.initial_tm), 0)


class UniformDriftTMGenerator(TMGenerator):
    """Shift demands by a random value chosen between `delta_max` and `delta_min`"""
    _TYPE = UniformTMGeneratorParams._type
    def __init__(self, params: UniformDriftTMGeneratorParams):
        super().__init__(params)
    
    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: UniformDriftTMGeneratorParams = self.params
        RNG = self.get_rng()
        arr = PARAMS.initial_tm
        for _ in range(PARAMS.count):
            drift = RNG.random(size=arr.shape, dtype=PARAMS.dtype) * (PARAMS.delta_max - PARAMS.delta_min) + \
                PARAMS.delta_min
            tm = np.clip(arr + drift, a_min=0, a_max=None)

            np.fill_diagonal(tm, 0.0)
            np.multiply(tm, PARAMS.scale_factor, out=tm)
            yield tm

#### SAMPLED ####

@dataclass(frozen=True)
class SampledTMGeneratorParams(TMGeneratorParams):
    delta_max: float = 1.0
    delta_min: float = 0.0
    sample_rate: float = 0.05
    initial_tm: np.ndarray = field(default_factory=lambda: np.empty((1,)), metadata={'help': argparse.SUPPRESS})
    _type: ClassVar[str] = 'sampled'

    def __post_init__(self):
        assert self.delta_max > self.delta_min
        assert self.dtype == self.initial_tm.dtype
        shape = self.initial_tm.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        assert np.allclose(np.diag(self.initial_tm), 0)
        assert 0 <= self.sample_rate and self.sample_rate <= 1


class SampledTMGenerator(TMGenerator):
    """
    Shift demands by a random value chosen between `delta_max` and `delta_min`
    for only at most a few randomly chosen demands.
    """
    _TYPE = SampledTMGeneratorParams._type
    def __init__(self, params: SampledTMGeneratorParams):
        super().__init__(params)

    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: SampledTMGeneratorParams = self.params
        number_of_samples = int(PARAMS.initial_tm.size * PARAMS.sample_rate)
        RNG = self.get_rng()
        arr = PARAMS.initial_tm
        m, _ = arr.shape
        for _ in range(PARAMS.count):
            sample = np.zeros((m, m))
            indices = [
                (RNG.integers(0, m), RNG.integers(0, m))
                    for _ in range(number_of_samples)
            ]
            shifts = [RNG.random() * (PARAMS.delta_max - PARAMS.delta_min) + PARAMS.delta_min 
                    for _ in range(number_of_samples)]
            for index, shift in zip(indices, shifts):
                sample[index] = shift
            drift = RNG.random(size=arr.shape, dtype=PARAMS.dtype) * (PARAMS.delta_max - PARAMS.delta_min) + \
                PARAMS.delta_min
            yield np.clip(arr + drift, a_min=0, a_max=1)


@dataclass(frozen=True)
class NCFlowTrafficMatrixGeneratorParams(TMGeneratorParams):
    rel_mean: float = 0.0
    rel_stddev: float = 1.0
    initial_tm: np.ndarray = field(default_factory=lambda: np.empty((1,)), metadata={'help': argparse.SUPPRESS})
    _type: ClassVar[str] = 'ncflow'

    def __post_init__(self):
        assert self.delta_max > self.delta_min
        assert self.dtype == self.initial_tm.dtype
        shape = self.initial_tm.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        assert np.allclose(np.diag(self.initial_tm), 0)
        self.original_mean = np.mean(self.initial_tm)


class NCFlowTrafficMatrixGenerator(TMGenerator):
    """
    Converter based on what was used for `NCFlow`.
    Pick an up/down direction, perturb by sampling
    a normal distribution, then clip it.
    """
    _TYPE = NCFlowTrafficMatrixGeneratorParams._type
    def __init__(self,  params: NCFlowTrafficMatrixGeneratorParams):
        super().__init__(params)

    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: NCFlowTrafficMatrixGeneratorParams = self.params 
        RNG = self.get_rng()
        arr = PARAMS.initial_tm
        for _ in range(PARAMS.count):
            new_mean = PARAMS.original_mean * PARAMS.rel_mean * RNG.choice([-1, 1])
            new_stddev = PARAMS.original_mean * PARAMS.rel_stddev
            sample = RNG.normal(new_mean, new_stddev, arr.shape)
            yield np.clip(arr + sample, a_min=0, a_max=1)


def get_param_class_from_name(name: str) -> Tuple[type[TMGeneratorParams], type[TMGenerator]]:
    params = None
    generator = None
    for param in TMGeneratorParams.__subclasses__():
        if param._type == name:
            params = param
            break
    for tm_gen in TMGenerator.__subclasses__():
        if tm_gen.type() == name:
            generator = tm_gen
            break
    assert params is not None and generator is not None
    return params, generator

def attach_TM_class_parser(parser: jsonargparse.ArgumentParser):
    parser.add_argument('--tm-class', choices=[tp.type() for tp in TMGenerator.__subclasses__()], required=True)
    for param in TMGeneratorParams.__subclasses__():
        parser.add_class_arguments(param, nested_key=param._type)

def parse_and_get_TM(
    tm_seed: Optional[int], tm_count: int, scale_factor: float,
    graph: nx.DiGraph, args: jsonargparse.Namespace
) -> TMGenerator:
    cls_name = args.tm_class
    tm_class_args = getattr(args, cls_name)
    tm_class_args.seed = tm_seed
    tm_class_args.count = tm_count
    tm_class_args.scale_factor = scale_factor
    if cls_name == UniformTMGenerator.type():
        tm_class_args.n = graph.number_of_nodes()
    elif cls_name == ExponentialTMGenerator.type():
        tm_class_args.graph = graph
    else:
        raise ValueError
    param_cls, tm_cls = get_param_class_from_name(cls_name)
    param = param_cls.make_from_args(tm_class_args)
    return tm_cls(param)
