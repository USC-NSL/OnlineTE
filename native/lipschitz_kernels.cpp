#include "path_kernels.hpp"

template <typename Scalar>
void estimate_lipschitz(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<Scalar> gram_values,
    ConstArray1<Scalar> demands,
    int iterations,
    Array1<Scalar> output,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *q_offsets = static_cast<const int64_t *>(gram_offsets.data());
    const auto *q = static_cast<const Scalar *>(gram_values.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    auto *out = static_cast<Scalar *>(output.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        const int64_t beta = commodity_offsets[k + 1] - commodity_offsets[k];
        const Scalar *block = q + q_offsets[k];
        const Scalar demand_sq = d[k] * d[k];
        std::vector<Scalar> v(static_cast<size_t>(beta), Scalar(1));
        std::vector<Scalar> result(static_cast<size_t>(beta), Scalar(0));

        for (int iteration = 0; iteration < iterations; ++iteration) {
            Scalar norm_sq = Scalar(0);
            for (int64_t i = 0; i < beta; ++i) {
                norm_sq += v[i] * v[i];
                Scalar acc = Scalar(0);
                for (int64_t j = 0; j < beta; ++j) {
                    acc += block[i * beta + j] * v[j];
                }
                result[i] = demand_sq * acc;
            }
            const Scalar norm = std::sqrt(norm_sq);
            for (int64_t i = 0; i < beta; ++i) {
                v[i] = result[i] / norm;
            }
        }

        Scalar numerator = Scalar(0);
        Scalar denominator = Scalar(0);
        for (int64_t i = 0; i < beta; ++i) {
            Scalar acc = Scalar(0);
            for (int64_t j = 0; j < beta; ++j) {
                acc += block[i * beta + j] * v[j];
            }
            result[i] = demand_sq * acc;
            numerator += result[i] * v[i];
            denominator += v[i] * v[i];
        }
        out[k] = numerator / denominator;
    }
}

template void estimate_lipschitz(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<float> gram_values,
    ConstArray1<float> demands,
    int iterations,
    Array1<float> output,
    int num_threads
);
template void estimate_lipschitz(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<double> gram_values,
    ConstArray1<double> demands,
    int iterations,
    Array1<double> output,
    int num_threads
);