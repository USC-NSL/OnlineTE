#include "path_kernels.hpp"

template <typename Scalar>
void capacity_feasible_splits(
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> edge_factors,
    Array1<Scalar> output,
    int num_threads
) {
    const size_t paths = values.shape(0);
    if (path_edge_offsets.shape(0) != paths + 1 || output.shape(0) != paths) {
        throw std::invalid_argument("path offsets and output must match the path count");
    }
    // All arrays are contiguous. Compare byte ranges, including partial views
    // and integer topology buffers, before validating or writing any values.
    const auto output_start = reinterpret_cast<uintptr_t>(output.data());
    const size_t output_bytes = output.nbytes();
    const auto reject_overlap = [&](const auto &input) {
        const auto input_start = reinterpret_cast<uintptr_t>(input.data());
        const size_t input_bytes = input.nbytes();
        if (output_bytes && input_bytes &&
            (output_start <= input_start ? input_start - output_start < output_bytes
                                         : output_start - input_start < input_bytes)) {
            throw std::invalid_argument("output must not overlap any input");
        }
    };
    reject_overlap(path_edge_offsets);
    reject_overlap(path_edges);
    reject_overlap(values);
    reject_overlap(edge_factors);

    const auto *offsets = path_edge_offsets.data();
    const auto *edges = path_edges.data();
    const auto *y = values.data();
    const auto *factors = edge_factors.data();
    auto *out = output.data();
    if (offsets[0] != 0 || offsets[paths] < 0 ||
        static_cast<size_t>(offsets[paths]) != path_edges.shape(0)) {
        throw std::invalid_argument("invalid path edge offsets");
    }
    for (size_t p = 0; p < paths; ++p) {
        if (offsets[p] < 0 || offsets[p + 1] <= offsets[p] ||
            offsets[p + 1] > offsets[paths]) {
            throw std::invalid_argument("every path must contain valid edge offsets");
        }
        if (!std::isfinite(y[p]) || y[p] < Scalar(0)) {
            throw std::invalid_argument("values must be finite and nonnegative");
        }
    }
    for (size_t e = 0; e < edge_factors.shape(0); ++e) {
        if (!std::isfinite(factors[e]) || factors[e] < Scalar(0) || factors[e] > Scalar(1)) {
            throw std::invalid_argument("edge_factors must be finite and in [0, 1]");
        }
    }
    for (size_t i = 0; i < path_edges.shape(0); ++i) {
        if (edges[i] < 0 || static_cast<size_t>(edges[i]) >= edge_factors.shape(0)) {
            throw std::invalid_argument("path edge index is outside edge_factors");
        }
    }
    const int threads = effective_threads(num_threads);
#pragma omp parallel for schedule(static) num_threads(threads) if(paths > 1)
    for (int64_t p = 0; p < static_cast<int64_t>(paths); ++p) {
        Scalar factor = Scalar(1);
        for (int64_t i = offsets[p]; i < offsets[p + 1]; ++i) {
            factor = std::min(factor, factors[edges[i]]);
        }
        out[p] = y[p] * factor;
    }
}

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

template void capacity_feasible_splits<float>(
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> values,
    ConstArray1<float> edge_factors,
    Array1<float> output,
    int num_threads
);
template void capacity_feasible_splits<double>(
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> values,
    ConstArray1<double> edge_factors,
    Array1<double> output,
    int num_threads
);
