#include "path_kernels.hpp"

template <typename Scalar>
void build_gram(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    I64Array gram_offsets,
    Array1<Scalar> output,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *edge_offsets = static_cast<const int64_t *>(path_edge_offsets.data());
    const auto *edges = static_cast<const int32_t *>(path_edges.data());
    const auto *scales = static_cast<const Scalar *>(edge_scale.data());
    const auto *q_offsets = static_cast<const int64_t *>(gram_offsets.data());
    auto *q = static_cast<Scalar *>(output.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        const int64_t p_start = commodity_offsets[k];
        const int64_t beta = commodity_offsets[k + 1] - p_start;
        Scalar *block = q + q_offsets[k];
        std::fill(block, block + beta * beta, Scalar(0));

        for (int64_t i = 0; i < beta; ++i) {
            const int64_t path_i = p_start + i;
            for (int64_t j = i; j < beta; ++j) {
                const int64_t path_j = p_start + j;
                Scalar shared_weight = Scalar(0);
                for (int64_t a = edge_offsets[path_i]; a < edge_offsets[path_i + 1]; ++a) {
                    const int32_t edge = edges[a];
                    for (int64_t b = edge_offsets[path_j]; b < edge_offsets[path_j + 1]; ++b) {
                        if (edge == edges[b]) {
                            const Scalar scale = scales[edge];
                            shared_weight += scale * scale;
                        }
                    }
                }
                block[i * beta + j] = shared_weight;
                block[j * beta + i] = shared_weight;
            }
        }
    }
}

template <typename Scalar>
void block_gram_matvec(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<Scalar> gram_values,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    Array1<Scalar> output,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *q_offsets = static_cast<const int64_t *>(gram_offsets.data());
    const auto *q = static_cast<const Scalar *>(gram_values.data());
    const auto *x = static_cast<const Scalar *>(values.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    auto *out = static_cast<Scalar *>(output.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        const int64_t start = commodity_offsets[k];
        const int64_t beta = commodity_offsets[k + 1] - start;
        const Scalar *block = q + q_offsets[k];
        const Scalar demand_sq = d[k] * d[k];
        for (int64_t i = 0; i < beta; ++i) {
            Scalar acc = Scalar(0);
            for (int64_t j = 0; j < beta; ++j) {
                acc += block[i * beta + j] * x[start + j];
            }
            out[start + i] = demand_sq * acc;
        }
    }
}

template void build_gram(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> edge_scale,
    I64Array gram_offsets,
    Array1<float> output,
    int num_threads
);
template void build_gram(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> edge_scale,
    I64Array gram_offsets,
    Array1<double> output,
    int num_threads
);

template void block_gram_matvec(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<float> gram_values,
    ConstArray1<float> values,
    ConstArray1<float> demands,
    Array1<float> output,
    int num_threads
);
template void block_gram_matvec(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<double> gram_values,
    ConstArray1<double> values,
    ConstArray1<double> demands,
    Array1<double> output,
    int num_threads
);