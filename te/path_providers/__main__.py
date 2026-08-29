import jsonargparse
from topologies.utils import load_zoo_topology, set_random_capacities, get_edge_indexing
from .base import *
from .schemes import *


if __name__ == '__main__':
    parser = jsonargparse.ArgumentParser('Compute paths for topologies')
    parser.add_argument('topo_name', type=str, help='Topology name (without the postfix of .json, .gml, etc.)')
    parser.add_argument('max_path', type=int, help='Maximum number of paths per commodity')
    parser.add_argument('scheme', type=PathSchemes, help='Which scheme to use for path generation')
    parser.add_argument('--topo-seed', type=int, help='Seed for capacity generation')
    parser.add_argument('--output-path', type=str, help='Output path for the path provider')
    args = parser.parse_args()

    graph = load_zoo_topology(args.topo_name)
    indexing = get_edge_indexing(graph)
    set_random_capacities(graph, args.topo_seed)
    scheme = get_scheme(args.scheme)
    output_path = args.output_path
    if output_path is None:
        output_path = f'paths_{args.topo_name}.pkl'

    provider = build_provider(
        T = args.max_path,
        graph = graph,
        per_commodity_provider=scheme,
        edge_indexing=indexing
    )
    provider.save(output_path)
