#include "mannwhitneyu.hpp"
#include "tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <vector>

namespace py = pybind11;
using namespace hpdexc::tensor;

// ========== numpy <-> Vector ==========
inline Vector<int32_t> numpy_to_vec_i32(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.request();
    const int32_t* src = static_cast<const int32_t*>(buf.ptr);
    std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    std::vector<int32_t> data(src, src + n);
    return Vector<int32_t>::from_std(data);
}

// ========== Ndarray -> numpy ==========
template <typename T>
py::array ndarray_to_numpy(const Ndarray<T>& arr) {
    auto shp = arr.shape();
    std::vector<py::ssize_t> pyshape(shp.begin(), shp.end());
    py::array_t<T> out(pyshape);
    if (arr.size() > 0) {
        std::memcpy(out.mutable_data(), arr.data(), arr.size() * sizeof(T));
    }
    return out;
}

// 使用 tensor_py 的零拷贝转换
using hpdexc::tensor_py::from_scipy_csc_t;

// ========== mannwhitneyu binding ==========
py::tuple mannwhitneyu_bind(
    py::object csc_matrix,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> group_id,
    std::size_t n_targets,
    int threads,
    // --- options ---
    bool ref_sorted,
    bool tar_sorted,
    bool tie_correction,
    bool use_continuity,
    int sparse_type,
    double sparse_value,
    int alternative,
    int method
) {
    Vector<int32_t> gid = numpy_to_vec_i32(group_id);
    auto A = from_scipy_csc_t<double,int32_t>(csc_matrix);

    // 填充选项
    hpdexc::MannWhitneyuOption<double> opt{};
    opt.ref_sorted     = ref_sorted;
    opt.tar_sorted     = tar_sorted;
    opt.tie_correction = tie_correction;
    opt.use_continuity = use_continuity;
    opt.sparse_type    = static_cast<hpdexc::MannWhitneyuOption<double>::SparseValueMinmax>(sparse_type);
    opt.sparse_value   = sparse_value;
    opt.alternative    = static_cast<hpdexc::MannWhitneyuOption<double>::Alternative>(alternative);
    opt.method         = static_cast<hpdexc::MannWhitneyuOption<double>::Method>(method);

    // 算法调用
    Ndarray<double> U, P;
    std::tie(U, P) = hpdexc::mannwhitneyu<double,int32_t>(A, opt, gid, n_targets, threads, nullptr);

    return py::make_tuple(ndarray_to_numpy(U), ndarray_to_numpy(P));
}

// ========== group_mean binding ==========
py::array group_mean_bind(
    py::object csc_matrix,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> group_id,
    std::size_t n_groups,
    int threads) {
    Vector<int32_t> gid = numpy_to_vec_i32(group_id);
    auto A = from_scipy_csc_t<double,int32_t>(csc_matrix);

    Ndarray<double> result = hpdexc::group_mean(A, gid, n_groups, threads, nullptr);
    return ndarray_to_numpy(result);
}

// ========== 模块入口 ==========
PYBIND11_MODULE(kernel, m) {
    m.doc() = "Bindings for Mann-Whitney U test and group mean";

    m.def("mannwhitneyu", &mannwhitneyu_bind,
          py::arg("csc_matrix"),
          py::arg("group_id"),
          py::arg("n_targets"),
          py::arg("threads") = -1,
          py::arg("ref_sorted") = false,
          py::arg("tar_sorted") = false,
          py::arg("tie_correction") = true,
          py::arg("use_continuity") = true,
          py::arg("sparse_type") = 0,     // none
          py::arg("sparse_value") = 0.0,
          py::arg("alternative") = 2,     // two_sided
          py::arg("method") = 0           // automatic
    );

    m.def("group_mean", &group_mean_bind,
          py::arg("csc_matrix"),
          py::arg("group_id"),
          py::arg("n_groups"),
          py::arg("threads") = -1);
}
