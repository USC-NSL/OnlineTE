#include "path_kernels.hpp"

template <typename Scalar>
double assignment_delta_norm_squared(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<Scalar> gram_values,
    ConstArray1<Scalar> current,
    ConstArray1<Scalar> previous,
    ConstArray1<Scalar> demands,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *q_offsets = static_cast<const int64_t *>(gram_offsets.data());
    const auto *q = static_cast<const Scalar *>(gram_values.data());
    const auto *y = static_cast<const Scalar *>(current.data());
    const auto *y_old = static_cast<const Scalar *>(previous.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);
    std::vector<double> contributions(static_cast<size_t>(commodities), 0.0);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        const int64_t start = commodity_offsets[k];
        const int64_t beta = commodity_offsets[k + 1] - start;
        const Scalar *block = q + q_offsets[k];
        double quadratic = 0.0;
        for (int64_t i = 0; i < beta; ++i) {
            const double delta_i = static_cast<double>(y[start + i]) -
                                   static_cast<double>(y_old[start + i]);
            double row_product = 0.0;
            for (int64_t j = 0; j < beta; ++j) {
                const double delta_j = static_cast<double>(y[start + j]) -
                                       static_cast<double>(y_old[start + j]);
                row_product += static_cast<double>(block[i * beta + j]) * delta_j;
            }
            quadratic += delta_i * row_product;
        }
        const double demand = static_cast<double>(d[k]);
        contributions[static_cast<size_t>(k)] = demand * demand * quadratic;
    }

    double total = 0.0;
    for (const double contribution : contributions) {
        total += contribution;
    }
    return total;
}

template <typename Scalar>
std::vector<double> _cheapest_path_per_commodity(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> edge_duals,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *edge_offsets = static_cast<const int64_t *>(path_edge_offsets.data());
    const auto *edges = static_cast<const int32_t *>(path_edges.data());
    const auto *scales = static_cast<const Scalar *>(edge_scale.data());
    const auto *duals = static_cast<const Scalar *>(edge_duals.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);
    std::vector<double> contributions(static_cast<size_t>(commodities), 0.0);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        double minimum = std::numeric_limits<double>::infinity();
        for (int64_t path = commodity_offsets[k]; path < commodity_offsets[k + 1]; ++path) {
            double path_value = 0.0;
            for (int64_t i = edge_offsets[path]; i < edge_offsets[path + 1]; ++i) {
                const int32_t edge = edges[i];
                path_value += static_cast<double>(scales[edge]) *
                              static_cast<double>(duals[edge]);
            }
            minimum = std::min(minimum, path_value);
        }
        contributions[static_cast<size_t>(k)] = minimum;
    }
    return contributions;
}

template <typename Scalar>
double mlu_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> edge_duals,
    ConstArray1<Scalar> demands,
    int num_threads
) {
    auto contributions = _cheapest_path_per_commodity(
        commodity_path_offsets,
        path_edge_offsets,
        path_edges,
        edge_scale,
        edge_duals,
        num_threads
    );
    const auto *d = static_cast<const Scalar *>(demands.data());
    
    double total = 0.0;
    for (size_t k = 0; k < contributions.size(); ++k) {
        total += contributions[k] * d[k];
    }
    return total;
}

template <typename Scalar>
double maxflow_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> edge_duals,
    ConstArray1<Scalar> demands,
    int total_number_of_commodities,
    int num_threads
) {
    auto contributions = _cheapest_path_per_commodity(
        commodity_path_offsets,
        path_edge_offsets,
        path_edges,
        edge_scale,
        edge_duals,
        num_threads
    );
    const auto *d = static_cast<const Scalar *>(demands.data());
    
    double total = 0.0;
    double K_recip = 1.0 / total_number_of_commodities;
    for (size_t k = 0; k < contributions.size(); ++k) {
        total += std::max(K_recip - contributions[k], 0.0) * d[k];
    }
    return total;
}

template double assignment_delta_norm_squared(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<float> gram_values,
    ConstArray1<float> current,
    ConstArray1<float> previous,
    ConstArray1<float> demands,
    int num_threads
);
template double assignment_delta_norm_squared(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<double> gram_values,
    ConstArray1<double> current,
    ConstArray1<double> previous,
    ConstArray1<double> demands,
    int num_threads
);

template std::vector<double> _cheapest_path_per_commodity(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> edge_scale,
    ConstArray1<float> edge_duals,
    int num_threads
);
template std::vector<double> _cheapest_path_per_commodity(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> edge_scale,
    ConstArray1<double> edge_duals,
    int num_threads
);

template double mlu_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> edge_scale,
    ConstArray1<float> edge_duals,
    ConstArray1<float> demands,
    int num_threads
);
template double mlu_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> edge_scale,
    ConstArray1<double> edge_duals,
    ConstArray1<double> demands,
    int num_threads
);

template double maxflow_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> edge_scale,
    ConstArray1<float> edge_duals,
    ConstArray1<float> demands,
    int total_number_of_commodities,
    int num_threads
);
template double maxflow_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> edge_scale,
    ConstArray1<double> edge_duals,
    ConstArray1<double> demands,
    int total_number_of_commodities,
    int num_threads
);