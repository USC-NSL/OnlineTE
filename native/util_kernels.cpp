#include "path_kernels.hpp"

template <typename Scalar>
void initialize_splits(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> capacities,
    Array1<Scalar> output,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *edge_offsets = static_cast<const int64_t *>(path_edge_offsets.data());
    const auto *edges = static_cast<const int32_t *>(path_edges.data());
    const auto *caps = static_cast<const Scalar *>(capacities.data());
    auto *out = static_cast<Scalar *>(output.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        Scalar total = Scalar(0);
        for (int64_t path = commodity_offsets[k]; path < commodity_offsets[k + 1]; ++path) {
            Scalar bottleneck = std::numeric_limits<Scalar>::max();
            for (int64_t i = edge_offsets[path]; i < edge_offsets[path + 1]; ++i) {
                bottleneck = std::min(bottleneck, caps[edges[i]]);
            }
            out[path] = bottleneck;
            total += bottleneck;
        }
        for (int64_t path = commodity_offsets[k]; path < commodity_offsets[k + 1]; ++path) {
            out[path] /= total;
        }
    }
}

template <typename Scalar>
void paths_to_edge(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    Array2<Scalar> output,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *edge_offsets = static_cast<const int64_t *>(path_edge_offsets.data());
    const auto *edges = static_cast<const int32_t *>(path_edges.data());
    const auto *scales = static_cast<const Scalar *>(edge_scale.data());
    const auto *y = static_cast<const Scalar *>(values.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    auto *out = static_cast<Scalar *>(output.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        for (int64_t path = commodity_offsets[k]; path < commodity_offsets[k + 1]; ++path) {
            const Scalar flow = d[k] * y[path];
            for (int64_t i = edge_offsets[path]; i < edge_offsets[path + 1]; ++i) {
                const int32_t edge = edges[i];
                out[static_cast<int64_t>(edge) * commodities + k] += flow * scales[edge];
            }
        }
    }
}

template <typename Scalar>
void paths_to_edge_mean(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    Array1<Scalar> output,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *edge_offsets = static_cast<const int64_t *>(path_edge_offsets.data());
    const auto *edges = static_cast<const int32_t *>(path_edges.data());
    const auto *scales = static_cast<const Scalar *>(edge_scale.data());
    const auto *y = static_cast<const Scalar *>(values.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    auto *out = static_cast<Scalar *>(output.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int64_t num_edges = static_cast<int64_t>(output.shape(0));
    const int threads = effective_threads(num_threads);

#pragma omp parallel num_threads(threads) if(commodities > 1)
    {
        std::vector<Scalar> local(static_cast<size_t>(num_edges), Scalar(0));
#pragma omp for schedule(static)
        for (int64_t k = 0; k < commodities; ++k) {
            for (int64_t path = commodity_offsets[k]; path < commodity_offsets[k + 1]; ++path) {
                const Scalar flow = d[k] * y[path] / Scalar(commodities);
                for (int64_t i = edge_offsets[path]; i < edge_offsets[path + 1]; ++i) {
                    const int32_t edge = edges[i];
                    local[edge] += flow * scales[edge];
                }
            }
        }
#pragma omp critical
        {
            for (int64_t edge = 0; edge < num_edges; ++edge) {
                out[edge] += local[edge];
            }
        }
    }
}

template <typename Scalar>
Scalar total_flow(
    I64Array commodity_path_offsets,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *y = static_cast<const Scalar *>(values.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);
    Scalar total = Scalar(0);

#pragma omp parallel for reduction(+ : total) schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        Scalar split_sum = Scalar(0);
        for (int64_t path = commodity_offsets[k]; path < commodity_offsets[k + 1]; ++path) {
            split_sum += y[path];
        }
        total += d[k] * split_sum;
    }
    return total;
}

template void initialize_splits(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> capacities,
    Array1<float> output,
    int num_threads
);
template void initialize_splits(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> capacities,
    Array1<double> output,
    int num_threads
);

template void paths_to_edge(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> edge_scale,
    ConstArray1<float> values,
    ConstArray1<float> demands,
    Array2<float> output,
    int num_threads
);
template void paths_to_edge(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> edge_scale,
    ConstArray1<double> values,
    ConstArray1<double> demands,
    Array2<double> output,
    int num_threads
);

template void paths_to_edge_mean(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> edge_scale,
    ConstArray1<float> values,
    ConstArray1<float> demands,
    Array1<float> output,
    int num_threads
);
template void paths_to_edge_mean(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> edge_scale,
    ConstArray1<double> values,
    ConstArray1<double> demands,
    Array1<double> output,
    int num_threads
);

template float total_flow(
    I64Array commodity_path_offsets,
    ConstArray1<float> values,
    ConstArray1<float> demands,
    int num_threads
);
template double total_flow(
    I64Array commodity_path_offsets,
    ConstArray1<double> values,
    ConstArray1<double> demands,
    int num_threads
);