#ifndef ONLINETE_PATH_KERNELS_H
#define ONLINETE_PATH_KERNELS_H

/**
 * These are custom kernels for path-based operations.
 * These replace the original OnlineTE `Numba` kernels. They
 * are faster, leaner and no longer impose an extra JIT overhead.
 * 
 * All kernels here use `nanobind`.
 * 
 * IMPORTANT NOTE
 * --------------
 * Unlike the original Numba kernels, these kernels treat arrays
 * as ragged 1D lists. In particular, the original path split
 * matrix `Y_{tk}` was a padded 2D array, where padding was added
 * in case the number of paths for some commodity was less than
 * the maximum (`T`).
 * Here, all ragged arrays (i.e. `Y_{tk}` and `A^{(k)}`) are
 * flat 1D arrays with an offset matrix provided that tells us where
 * each array begins and ends on the flat list.
 */

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

/**
 * 1D array.
 */
template <typename T>
using Array1 = nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/**
 * Constant 1D array.
 */
template <typename T>
using ConstArray1 = nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

/**
 * 2D array.
 */
template <typename T>
using Array2 = nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/**
 * Constant int64 array.
 */
using I64Array = ConstArray1<int64_t>;
/**
 * Constant int32 array.
 */
using I32Array = ConstArray1<int32_t>;

/**
 * Utility for handling thread counts. Less than or
 * equal to 0 numbers are treated as just 1 thread.
 */
static int effective_threads(int requested) {
    return requested > 0 ? requested : 1;
}

/**
 * Kernels operate on flat 2D ragged arrays.
 * In general, we have a consistent way to refer to arguments for all
 * of these kernels.
 * 
 * @see PackedPathBatch.from_path_provider to see how each input argument
 * is calculated. In short:
 * 
 * @param commodity_path_offsets Offset for paths per commodity. This is
 * the cumulative sum of the original `beta` vector from the path provider.
 * Take heed that for `K` commodities, this vector has a length of `K+1` with
 * the first entry being 0.
 * @param path_edges Each path is given a globally unique index like `i` and
 * this 1D array contains all edges used by these paths. This naturally requires
 * its own offset array.
 * @param path_edge_offsets Offset for `path_edges` as described above.
 * @param gram_offsets The Gram matrix is also ragged (as a square matrix,
 * it only covers a `beta` by `beta` square). It is thus equivalently just
 * the cumulative sum of `beta^2`.
 * @param edge_scale A scaling factor per edge (usually, the inverse of edge
 * capacity).
 * @param num_threads Number of threads for OpenMP parallel loop.
 * @param demands Simple 1D array of demands per commodity.
 */

/**
 * Build the Gram matrix (i.e. `A^{(k)} := (alpha^{(k)}^T alpha^{(k)})).
 * 
 * @param output The 1D output vector to write the flattened Gram matrix.
 */
template <typename Scalar> void build_gram(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    I64Array gram_offsets,
    Array1<Scalar> output,
    int num_threads
);

/**
 * Implements multiplication between a ragged vector and ragged
 * Gram matrix.
 * 
 * @param output The 1D output vector to write the product result.
 */
template <typename Scalar> void block_gram_matvec(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<Scalar> gram_values,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    Array1<Scalar> output,
    int num_threads
);

/**
 * Implements a linear term that can be added directly to the flat
 * representation of a ragged array.
 * This function computes:
 * 
 *      -maxflow_shift + d_k . (alpha^{(k)}^T sharing_bias) .* edge_scale
 * 
 * This is the linear term that constant term that we get for computing
 * the gradient of the switch-side objective.
 * 
 * @param sharing_bias The sharing problem bias. A vector with one etnry
 * per edge.
 * @param maxflow_shift A single scalar to **subtract** from each entry
 * in the result.
 * @param output The 1D flat output vector.
 */
template <typename Scalar> void linear_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> sharing_bias,
    ConstArray1<Scalar> demands,
    Scalar maxflow_shift,
    Array1<Scalar> output,
    int num_threads
);

/**
 * Project a single vector on to the simplex.
 * 
 * @param values Reference to the vector to project.
 * @param output Output array to write the result in.
 * 
 * @note Upon completion, `output` will be a non-negative
 * vector and its entries sum up to 1.
 */
template <typename Scalar> static void project_simplex(
    std::vector<Scalar> &values,
    Scalar *output
);

/**
 * A single, fused kernel for all PGD operations for solving
 * the switch-side QP.
 * 
 * PGD OPERATION
 * -------------
 * In parallel, we solve the quadratic program:
 * 
 *      minimize 1/2 d_k^2 (Y_k - Y_k^{(old)})^T G (Y_k - Y_k^{(old)}) + Y_k^T L
 * 
 * This admits the gradient:
 * 
 *      d_k^2 G (Y_k - Y_k^{(old)}) + L
 * 
 * And we get the next candidate as:
 * 
 *      Z_k - step_k (d_k^2 G (Y_k - Y_k^{(old)}) + L)
 * 
 * Here, `Z` is the current iterate vector (which is often a Nesterov extrapolated
 * vector, and hence not ncessarily primal feasible).
 * 
 * The candidate is then finally projected and written to the output buffer. The
 * projection is just a plain simplex projection if `orthant` is false, or will be
 * skipped if the candidate already lies inside the simplex and `orthant` is true.
 * 
 * OBJECTIVE GAP CERTIFICATE
 * -------------------------
 * To certify optimality, we use a Frank-Wolfe gap on the current iterate.
 * Note that we cannot pre-compute this, as `Z` is not necessarily primal feasible
 * in case it is Nesterov extrapolated.
 * Now, assume that the gradient evaluated on `out` is `g`. Then the gap is:
 * 
 *      minimum g^T (out - y)
 * 
 * Over all feasible `y`. For the case of the simplex, `min g^T y` is just the
 * element-wise minimum of `g` (or 0 if the interior is allowed). This gives the
 * relation:
 * 
 *      gap = g^T out - min_i g_i            (if `orthant` is false)
 *      gap = g^T out - min(min_i g_i, 0)    (if `orthant` is true )
 * 
 * The relative gap is then calculated by evaluating the primal objective and then
 * reporting `gap / obj`.
 * 
 * @param z The current iterate (and often accelerated).
 * @param y_old The previous iterate. This is an artifact of the ADMM sharing form,
 * not from the acceleration.
 * @param step_sizes Lipschitz adjusted step sizes ( @see estimate_lipschitz )
 * @param orthant Set to true if sub-probability vectors are feasible (i.e. the
 * interior of the simplex is allowed). Only for MLU problems do we set this to
 * false.
 * @param relative_gaps 1D output array for writing the current relative gap of
 * individual commodities.
 * @param unconverged_commodities 1D array of commodities from previous iterations
 * that have not yet converged. Since each commodity converges at a different rate,
 * we need to selectively call them, lest we end up wasting cycles for commodities
 * that are already very close to optimality.
 * @param output Output buffer to write the new feasible iterate into. It is up to
 * the caller code to implement acceleration.
 */
template <typename Scalar> void projected_qp_step(
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
);

/**
 * Estimate the Lipschitz constant of the each commodity. This is quite
 * simply the dominant eigenvalue of the matrix:
 * 
 *      d_k^2 G_k
 * 
 * We estimate this with a few power method iterations.
 * 
 * @param iterations Number of power method iterations.
 * @param output The output buffer that holds the Lipschitz constant for
 * each commodity. The step size is calculated as `gamma / L_k` where `gamma`
 * is some value less than or equal to 1 and `L_k` is the estimated constant
 * for commodity `k`.
 */
template <typename Scalar> void estimate_lipschitz(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<Scalar> gram_values,
    ConstArray1<Scalar> demands,
    int iterations,
    Array1<Scalar> output,
    int num_threads
);

/**
 * Initialize path splits in proportion to path capacity.
 * 
 * @param output Flat ragged array of path split assignments.
 */
template <typename Scalar> void initialize_splits(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> capacities,
    Array1<Scalar> output,
    int num_threads
);

/**
 * Convert path-based splits to edge-based assignment, kept nicely
 * in an output 2D array.
 * 
 * @param output 2D array to write the edge-based assignment into.
 */
template <typename Scalar> void paths_to_edge(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    Array2<Scalar> output,
    int num_threads
);

/**
 * Returns the mean flow over all commodities per edge by just
 * using the path-based assignment.
 * 
 * @param output Simple 1D array to write the results into.
 */
template <typename Scalar> void paths_to_edge_mean(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    Array1<Scalar> output,
    int num_threads
);

/**
 * Returns the total flow of the current path-based assignment.
 * 
 * @returns Calculated total flow.
 */
template <typename Scalar> Scalar total_flow(
    I64Array commodity_path_offsets,
    ConstArray1<Scalar> values,
    ConstArray1<Scalar> demands,
    int num_threads
);

/**
 * Returns the sum of squared norm for:
 * 
 *      d_k alpha^{(k)} (current - previous)
 * 
 * This is used to calculate ADMM infeasibilities and is basically
 * just the Forbenius norm of a ragged array.
 * 
 * @param current Current path-based iterate.
 * @param previous Previous path-based iterate.
 * @returns The Frobenius norm.
 */
template <typename Scalar> double assignment_delta_norm_squared(
    I64Array commodity_path_offsets,
    I64Array gram_offsets,
    ConstArray1<Scalar> gram_values,
    ConstArray1<Scalar> current,
    ConstArray1<Scalar> previous,
    ConstArray1<Scalar> demands,
    int num_threads
);

/**
 * Given a vector `edge_duals`, returns for each `k`:
 * 
 *      min_t alpha^{(k)}_t^T (edge_duals * edge_scale)
 * 
 * This is needed for calculating the dual objective.
 * 
 * @param edge_duals The dual vector for edge prices.
 * @returns Vector of length `K`, one value per commodity.
 */
template <typename Scalar> std::vector<double> _cheapest_path_per_commodity(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> edge_duals,
    int num_threads
);

/**
 * Derive the dual objective bound for MLU.
 * 
 * @param edge_duals Dual feasible variable for the capacity constraint.
 * @returns Dual objective bound for this batch of commodities.
 */
template <typename Scalar> double mlu_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> edge_duals,
    ConstArray1<Scalar> demands,
    int num_threads
);

/**
 * Derive the dual objective bound for Max-Flow.
 * 
 * @param edge_duals Dual feasible variable for the capacity constraint.
 * @param total_number_of_commodities Total number of commodities (`K`).
 * @returns Dual objective bound for this batch of commodities.
 */
template <typename Scalar> double maxflow_dual_objective_term(
    I64Array commodity_path_offsets,
    I64Array path_edge_offsets,
    I32Array path_edges,
    ConstArray1<Scalar> edge_scale,
    ConstArray1<Scalar> edge_duals,
    ConstArray1<Scalar> demands,
    int total_number_of_commodities,
    int num_threads
);

#endif // ONLINETE_PATH_KERNELS_H