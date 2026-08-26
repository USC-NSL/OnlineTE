import pickle
import argparse
import numpy as np
import jsonargparse
import networkx as nx
from io import BufferedReader
from dataclasses import dataclass, field
from typing import Tuple, Optional, Iterator, Callable, Dict
from .base import *

#### UNIFORM ####

@dataclass(frozen=True)
class UniformTMGeneratorParams(TMGeneratorParams):
    """
    Parameters defining a random, uniform traffic matrix.

    Attributes
    ----------
    n: int
        Number of nodes
    min: float
        Minimum admissible demand value
    max: float
        Maximum admissible demand value
    """
    n: int = field(default=0, metadata={'help': argparse.SUPPRESS})
    min: float
    max: float

    def __post_init__(self):
        assert self.min >= 0 and self.max >= 0 and self.max >= self.min
        assert self.n > 0


class UniformTMGenerator(TMGenerator):
    _TYPE = 'Uniform'

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
    graph: nx.DiGraph
        Topology graph object. We need distances between
        nodes, so that is why we need it.
    beta: float
        Distribution mean; can be arbitrary positive value.
    gamma: float
        Distribution decay; must be in `(0, 1]`
    """
    graph: nx.DiGraph = field(default=nx.DiGraph(), metadata={'help': argparse.SUPPRESS})
    beta: float
    gamma: float

    def __post_init__(self):
        assert self.beta > 0 and self.gamma > 0 and self.gamma <= 1
        assert self.graph.number_of_nodes() > 0


class ExponentialTMGenerator(TMGenerator):
    _TYPE = 'Exponential'

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
    n: int
    range1: Tuple[float, float]
    range2: Tuple[float, float]
    fraction: float

    def __post_init__(self):
        assert self.range1[0] <= self.range1[1]
        assert self.range2[0] <= self.range2[1]
        assert self.fraction > 0 and self.fraction < 1


class BimodalTMGenerator(TMGenerator):
    _TYPE = 'Bimodal'

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
    paths: Iterator[str]
        An iterator that returns paths to traffic matrices.
    scale: Optional[float]
        If present, the traffic matrix will be divided by this value
        element-wise. If `None`, the matrix is normalized between
        0 and 1 by dividing by the maximum value.
        Defaults to `1.0`.
    loader: Optional[Callable[[BufferedReader], np.ndarray]] = `pickle.load`
        A callable that takes a file object and returns a Numpy array.
        Defaults to `pickle.load`.
    """
    paths: Iterator[str]
    scale: Optional[float] = 1.0
    loader: Callable[[BufferedReader], np.ndarray] = pickle.load


class FilebackedTMGenerator(TMGeneratorParams):
    _TYPE = 'File-backed'

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
    initial_tm: np.ndarray
    delta_max: float
    delta_min: float

    def __post_init__(self):
        assert self.delta_max > self.delta_min
        assert self.dtype == self.initial_tm.dtype
        shape = self.initial_tm.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        assert np.allclose(np.diag(self.initial_tm), 0)


class UniformDriftTMGenerator(TMGenerator):
    """Shift demands by a random value chosen between `delta_max` and `delta_min`"""
    _TYPE = 'Uniform Drift'
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
    initial_tm: np.ndarray
    delta_max: float
    delta_min: float
    number_of_samples: int

    def __post_init__(self):
        assert self.delta_max > self.delta_min
        assert self.dtype == self.initial_tm.dtype
        shape = self.initial_tm.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        assert np.allclose(np.diag(self.initial_tm), 0)


class SampledTMGenerator(TMGenerator):
    """
    Shift demands by a random value chosen between `delta_max` and `delta_min`
    for only at most a few randomly chosen demands.
    """
    _TYPE = 'Sampled'
    def __init__(self, params: SampledTMGeneratorParams):
        super().__init__(params)

    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: SampledTMGeneratorParams = self.params
        RNG = self.get_rng()
        arr = PARAMS.initial_tm
        m, _ = arr.shape
        for _ in range(PARAMS.count):
            sample = np.zeros((m, m))
            indices = [
                (RNG.integers(0, m), RNG.integers(0, m))
                    for _ in range(PARAMS.number_of_samples)
            ]
            shifts = [RNG.random() * (PARAMS.delta_max - PARAMS.delta_min) + PARAMS.delta_min 
                    for _ in range(PARAMS.number_of_samples)]
            for index, shift in zip(indices, shifts):
                sample[index] = shift
            drift = RNG.random(size=arr.shape, dtype=PARAMS.dtype) * (PARAMS.delta_max - PARAMS.delta_min) + \
                PARAMS.delta_min
            yield np.clip(arr + drift, a_min=0, a_max=1)


@dataclass(frozen=True)
class NCFlowTrafficMatrixConverterParams(TMGeneratorParams):
    rel_mean: float
    rel_stddev: float
    initial_tm: np.ndarray

    def __post_init__(self):
        assert self.delta_max > self.delta_min
        assert self.dtype == self.initial_tm.dtype
        shape = self.initial_tm.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        assert np.allclose(np.diag(self.initial_tm), 0)
        self.original_mean = np.mean(self.initial_tm)


class NCFlowTrafficMatrixConverter(TMGenerator):
    """
    Converter based on what was used for `NCFlow`.
    Pick an up/down direction, perturb by sampling
    a normal distribution, then clip it.
    """
    _TYPE = 'NCFlow'
    def __init__(self,  params: NCFlowTrafficMatrixConverterParams):
        super().__init__(params)

    def __iter__(self) -> Iterator[np.ndarray]:
        PARAMS: NCFlowTrafficMatrixConverterParams = self.params 
        RNG = self.get_rng()
        arr = PARAMS.initial_tm
        for _ in range(PARAMS.count):
            new_mean = PARAMS.original_mean * PARAMS.rel_mean * RNG.choice([-1, 1])
            new_stddev = PARAMS.original_mean * PARAMS.rel_stddev
            sample = RNG.normal(new_mean, new_stddev, arr.shape)
            yield np.clip(arr + sample, a_min=0, a_max=1)


def attach_TM_class_parser(parser: jsonargparse.ArgumentParser):
    parser.add_argument(
        '--tm-class', required=True, type=TMGeneratorParams,
        help='Traffic Matrix class'
    )

def parse_and_get_TM(
    tm_seed: Optional[int], tm_count: int, scale_factor: float,
    graph: nx.DiGraph, args: jsonargparse.Namespace
) -> TMGenerator:
    args.tm_class.init_args.seed = tm_seed
    args.tm_class.init_args.count = tm_count
    args.tm_class.init_args.scale_factor = scale_factor
    if args.tm_class == UniformTMGeneratorParams.__name__:
        args.tm_class.init_args.n = graph.number_of_nodes()
    elif args.tm_class == ExponentialTMGeneratorParams.__name__:
        args.tm_class.init_args.graph = graph
    else:
        raise ValueError
    return jsonargparse._util.import_object(args.tm_class.class_path)(**vars(args.tm_class.init_args))
