#include "path_kernels.hpp"

template <typename Scalar>
void linear_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> sharing_bias,
    ConstArray1<Scalar> demands,
    Scalar maxflow_shift,
    Array1<Scalar> output,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *edge_offsets = static_cast<const int64_t *>(path_edge_offsets.data());
    const auto *edges = static_cast<const int32_t *>(path_edges.data());
    const auto *scales = static_cast<const Scalar *>(edge_scale.data());
    const auto *bias = static_cast<const Scalar *>(sharing_bias.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    auto *out = static_cast<Scalar *>(output.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
        for (int64_t path = commodity_offsets[k]; path < commodity_offsets[k + 1]; ++path) {
            Scalar acc = -maxflow_shift;
            for (int64_t i = edge_offsets[path]; i < edge_offsets[path + 1]; ++i) {
                const int32_t edge = edges[i];
                acc += scales[edge] * bias[edge];
            }
            out[path] = d[k] * acc;
        }
    }
}

template <typename Scalar>
static void project_simplex(
    std::vector<Scalar> &values,
    Scalar *output
) {
    std::vector<Scalar> sorted(values);
    std::sort(sorted.begin(), sorted.end(), std::greater<Scalar>());
    Scalar cumulative = Scalar(0);
    int64_t rho = -1;
    Scalar rho_sum = Scalar(0);
    for (int64_t i = 0; i < static_cast<int64_t>(sorted.size()); ++i) {
        cumulative += sorted[i];
        if (sorted[i] * Scalar(i + 1) > cumulative - Scalar(1)) {
            rho = i;
            rho_sum = cumulative;
        }
    }
    const Scalar theta = (rho_sum - Scalar(1)) / Scalar(rho + 1);
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
        output[i] = std::max(values[i] - theta, Scalar(0));
    }
}

template <typename Scalar>
void projected_qp_step(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<Scalar> gram_values,
    ConstArray1<Scalar> z,
    ConstArray1<Scalar> y_old,
    ConstArray1<Scalar> linear,
    ConstArray1<Scalar> demands,
    ConstArray1<Scalar> step_sizes,
    bool orthant,
    Array1<Scalar> output,
    Array1<double> relative_gaps,
    I64Array unconverged_commodities,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *q_offsets = static_cast<const int64_t *>(gram_offsets.data());
    const auto *q = static_cast<const Scalar *>(gram_values.data());
    const auto *z_data = static_cast<const Scalar *>(z.data());
    const auto *old_data = static_cast<const Scalar *>(y_old.data());
    const auto *c = static_cast<const Scalar *>(linear.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
    const auto *steps = static_cast<const Scalar *>(step_sizes.data());
    auto *out = static_cast<Scalar *>(output.data());
    auto *gaps = static_cast<double *>(relative_gaps.data());
    const auto *indices = static_cast<const int64_t *>(unconverged_commodities.data());
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t i = 0; i < unconverged_commodities.size(); i++) {
        const int64_t k = indices[i];
        const int64_t start = commodity_offsets[k];
        const int64_t beta = commodity_offsets[k + 1] - start;
        const Scalar *block = q + q_offsets[k];
        const Scalar demand_sq = d[k] * d[k];
        std::vector<Scalar> trial(static_cast<size_t>(beta));
        Scalar positive_sum = Scalar(0);

        for (int64_t i = 0; i < beta; ++i) {
            Scalar qx = Scalar(0);
            for (int64_t j = 0; j < beta; ++j) {
                qx += block[i * beta + j] * (z_data[start + j] - old_data[start + j]);
            }
            trial[i] = z_data[start + i] - steps[k] * (demand_sq * qx + c[start + i]);
            if (orthant) {
                trial[i] = std::max(trial[i], Scalar(0));
                positive_sum += trial[i];
            }
        }

        if (orthant && positive_sum <= Scalar(1)) {
            std::copy(trial.begin(), trial.end(), out + start);
        } else {
            project_simplex(trial, out + start);
        }

        // Evaluate a rigorous Wolfe dual-gap certificate at the projected
        // candidate.  The Nesterov extrapolate above need not be feasible, so
        // its first-order residual cannot be used as a stopping certificate.
        // Accumulate in double precision even for the float32 specialization.
        double quadratic = 0.0;
        double linear_objective = 0.0;
        double gradient_dot_candidate = 0.0;
        double minimum_gradient = std::numeric_limits<double>::infinity();
        const double demand_sq_double = static_cast<double>(d[k]) * static_cast<double>(d[k]);

        for (int64_t i = 0; i < beta; ++i) {
            double gram_product = 0.0;
            for (int64_t j = 0; j < beta; ++j) {
                const double delta_j = static_cast<double>(out[start + j]) -
                                       static_cast<double>(old_data[start + j]);
                gram_product += static_cast<double>(block[i * beta + j]) * delta_j;
            }
            const double quadratic_gradient = demand_sq_double * gram_product;
            const double gradient_i = quadratic_gradient + static_cast<double>(c[start + i]);
            const double candidate_i = static_cast<double>(out[start + i]);
            const double delta_i = candidate_i - static_cast<double>(old_data[start + i]);
            quadratic += delta_i * quadratic_gradient;
            linear_objective += static_cast<double>(c[start + i]) * candidate_i;
            gradient_dot_candidate += gradient_i * candidate_i;
            minimum_gradient = std::min(minimum_gradient, gradient_i);
        }

        const double primal_objective = 0.5 * quadratic + linear_objective;
        const double linear_minimum = orthant
            ? std::min(0.0, minimum_gradient)
            : minimum_gradient;
        double gap = gradient_dot_candidate - linear_minimum;
        const double roundoff_tolerance =
            64.0 * static_cast<double>(std::numeric_limits<Scalar>::epsilon()) *
            std::max({1.0, std::abs(gradient_dot_candidate), std::abs(linear_minimum)});

        if (gap < -roundoff_tolerance ||
            !std::isfinite(primal_objective) ||
            !std::isfinite(gap)) {
            gaps[k] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        gap = std::max(0.0, gap);
        const double lower_bound = primal_objective - gap;
        if (!std::isfinite(lower_bound)) {
            gaps[k] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double denominator = std::max({
            1.0,
            std::abs(primal_objective),
            std::abs(lower_bound),
        });
        gaps[k] = gap / denominator;
    }
}

template void linear_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<float> edge_scale,
    ConstArray1<float> sharing_bias,
    ConstArray1<float> demands,
    float maxflow_shift,
    Array1<float> output,
    int num_threads
);
template void linear_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<double> edge_scale,
    ConstArray1<double> sharing_bias,
    ConstArray1<double> demands,
    double maxflow_shift,
    Array1<double> output,
    int num_threads
);

template static void project_simplex(
    std::vector<float> &values,
    float *output
);
template static void project_simplex(
    std::vector<double> &values,
    double *output
);

template void projected_qp_step(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<float> gram_values,
    ConstArray1<float> z,
    ConstArray1<float> y_old,
    ConstArray1<float> linear,
    ConstArray1<float> demands,
    ConstArray1<float> step_sizes,
    bool orthant,
    Array1<float> output,
    Array1<double> relative_gaps,
    I64Array unconverged_commodities,
    int num_threads
);
template void projected_qp_step(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<double> gram_values,
    ConstArray1<double> z,
    ConstArray1<double> y_old,
    ConstArray1<double> linear,
    ConstArray1<double> demands,
    ConstArray1<double> step_sizes,
    bool orthant,
    Array1<double> output,
    Array1<double> relative_gaps,
    I64Array unconverged_commodities,
    int num_threads
);