#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace nb = nanobind;

template <typename T>
using Array1 = nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using ConstArray1 = nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <typename T>
using Array2 = nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

using I64Array = ConstArray1<int64_t>;
using I32Array = ConstArray1<int32_t>;

static int effective_threads(int requested) {
    return requested > 0 ? requested : 1;
}

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
static void project_simplex(std::vector<Scalar> &values, Scalar *output) {
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
    const int64_t commodities = static_cast<int64_t>(commodity_path_offsets.shape(0)) - 1;
    const int threads = effective_threads(num_threads);

#pragma omp parallel for schedule(static) num_threads(threads) if(commodities > 1)
    for (int64_t k = 0; k < commodities; ++k) {
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
double dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> edge_duals,
    ConstArray1<Scalar> demands,
    int num_threads
) {
    const auto *commodity_offsets = static_cast<const int64_t *>(commodity_path_offsets.data());
    const auto *edge_offsets = static_cast<const int64_t *>(path_edge_offsets.data());
    const auto *edges = static_cast<const int32_t *>(path_edges.data());
    const auto *scales = static_cast<const Scalar *>(edge_scale.data());
    const auto *duals = static_cast<const Scalar *>(edge_duals.data());
    const auto *d = static_cast<const Scalar *>(demands.data());
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
        contributions[static_cast<size_t>(k)] = static_cast<double>(d[k]) * minimum;
    }

    double total = 0.0;
    for (const double contribution : contributions) {
        total += contribution;
    }
    return total;
}

template <typename Scalar>
void bind_float_kernels(nb::module_ &m, const std::string &suffix) {
    m.def(("build_gram_" + suffix).c_str(), &build_gram<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("block_gram_matvec_" + suffix).c_str(), &block_gram_matvec<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("linear_term_" + suffix).c_str(), &linear_term<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("projected_qp_step_" + suffix).c_str(), &projected_qp_step<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("estimate_lipschitz_" + suffix).c_str(), &estimate_lipschitz<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("initialize_splits_" + suffix).c_str(), &initialize_splits<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("paths_to_edge_" + suffix).c_str(), &paths_to_edge<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("paths_to_edge_mean_" + suffix).c_str(), &paths_to_edge_mean<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("total_flow_" + suffix).c_str(), &total_flow<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("assignment_delta_norm_squared_" + suffix).c_str(), &assignment_delta_norm_squared<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("dual_objective_term_" + suffix).c_str(), &dual_objective_term<Scalar>, nb::call_guard<nb::gil_scoped_release>());
}

NB_MODULE(_path_kernels, m) {
    m.doc() = "Packed native kernels for the distributed path-based solver";
    bind_float_kernels<float>(m, "float32");
    bind_float_kernels<double>(m, "float64");
}
