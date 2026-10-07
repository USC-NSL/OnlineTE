#include "path_kernels.hpp"

// Module binders

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
    m.def(("mlu_dual_objective_term_" + suffix).c_str(), &mlu_dual_objective_term<Scalar>, nb::call_guard<nb::gil_scoped_release>());
    m.def(("maxflow_dual_objective_term_" + suffix).c_str(), &maxflow_dual_objective_term<Scalar>, nb::call_guard<nb::gil_scoped_release>());
}

NB_MODULE(_path_kernels, m) {
    m.doc() = "Packed native kernels for the distributed path-based solver";
    bind_float_kernels<float>(m, "float32");
    bind_float_kernels<double>(m, "float64");
}
