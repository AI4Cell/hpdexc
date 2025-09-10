#include "common.hpp"
#include "view.hpp"
#include "mannwhitney.hpp"
#include "progress.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace hpdexc;

// 将 Python 侧的 sparse_value 宽容地转换为模板类型 T：
// - 若 T 为整型：允许 Python int，亦允许形如 0.0 的“整数值”浮点；否则报错
// - 若 T 为浮点：接受 Python int 或 float
template<typename T>
static inline T coerce_sparse_value_strict(py::object sparse_value) {
    if (sparse_value.is_none()) {
        return T{};
    }

    if constexpr (std::is_integral_v<T>) {
        if (py::isinstance<py::int_>(sparse_value)) {
            long long v = sparse_value.cast<long long>();
            return static_cast<T>(v);
        }
        if (py::isinstance<py::float_>(sparse_value)) {
            double d = sparse_value.cast<double>();
            // 仅接受“小数部分为 0”的浮点值
            double r = std::round(d);
            if (std::abs(d - r) > 0.0) {
                throw std::runtime_error("sparse_value must be integral when group_id is integral");
            }
            long long v = static_cast<long long>(r);
            return static_cast<T>(v);
        }
        throw std::runtime_error("unsupported sparse_value type for integral group_id");
    } else {
        // 浮点模板：直接按 T 转
        return sparse_value.cast<T>();
    }
}

// SciPy 稀疏矩阵到 CscView 的转换（零拷贝方案）
CscView sparse_to_csc_view(py::object sparse_matrix) {
    auto indptr = sparse_matrix.attr("indptr").cast<py::array>();
    auto indices = sparse_matrix.attr("indices").cast<py::array>();
    auto data = sparse_matrix.attr("data").cast<py::array>();
    auto shape = sparse_matrix.attr("shape").cast<std::pair<int64_t, int64_t>>();
    
    // 检测索引数据类型，支持 int32 和 int64
    bool index_is_i64 = false;
    if (py::dtype::of<int64_t>().is(indptr.dtype()) && py::dtype::of<int64_t>().is(indices.dtype())) {
        index_is_i64 = true;
    } else if (py::dtype::of<int32_t>().is(indptr.dtype()) && py::dtype::of<int32_t>().is(indices.dtype())) {
        index_is_i64 = false;
    } else {
        throw std::runtime_error("indptr and indices must both be int32 or int64");
    }
    
    // 获取缓冲区指针（零拷贝）
    auto indptr_buf = indptr.request();
    auto indices_buf = indices.request();
    auto data_buf = data.request();
    
    // 创建 shared_ptr 来管理生命周期
    auto indptr_owner = std::make_shared<py::array>(indptr);
    auto indices_owner = std::make_shared<py::array>(indices);
    auto data_owner = std::make_shared<py::array>(data);
    
    return hpdexc::CscView::from_triplets_with_owners(
        indptr_buf.ptr,
        indices_buf.ptr,
        data_buf.ptr,
        static_cast<std::size_t>(shape.first),   // rows
        static_cast<std::size_t>(shape.second),  // cols
        static_cast<std::size_t>(indices_buf.size), // nnz
        index_is_i64,
        indptr_owner,
        indices_owner,
        data_owner
    );
}

hpdexc::DenseView sparse_to_dense_view(py::object sparse_matrix) {
    auto data = sparse_matrix.attr("data").cast<py::array>();
    auto shape = sparse_matrix.attr("shape").cast<std::pair<int64_t, int64_t>>();
    auto data_buf = data.request();
    
    // 检查数据类型是否为 int32_t
    if (!py::dtype::of<int32_t>().is(data.dtype())) {
        data = data.attr("astype")(py::dtype::of<int32_t>());
        data_buf = data.request();
    }
    
    return hpdexc::DenseView(
        data_buf.ptr, 
        static_cast<std::size_t>(shape.first), 
        static_cast<std::size_t>(shape.second)
    );
}

hpdexc::DenseViewMut sparse_to_dense_view_mut(py::object sparse_matrix) {
    auto data = sparse_matrix.attr("data").cast<py::array>();
    auto shape = sparse_matrix.attr("shape").cast<std::pair<int64_t, int64_t>>();
    auto data_buf = data.request(true);  // 要求可写缓冲区
    
    // 检查数据类型是否为 int32_t
    if (!py::dtype::of<int32_t>().is(data.dtype())) {
        data = data.attr("astype")(py::dtype::of<int32_t>());
        data_buf = data.request(true);
    }
    
    // 检查缓冲区是否可写
    if (data_buf.readonly) {
        throw std::runtime_error("sparse_to_dense_view_mut requires writable data buffer");
    }
    
    return hpdexc::DenseViewMut(
        data_buf.ptr, 
        static_cast<std::size_t>(shape.first), 
        static_cast<std::size_t>(shape.second)
    );
}

template<typename T>
py::array dense_view_to_array(hpdexc::DenseView view) {
    return py::array_t<T>({view.rows, view.cols}, {view.stride * sizeof(T), sizeof(T)}, static_cast<const T*>(view.data));
}
template<typename T>
py::array dense_view_mut_to_array(hpdexc::DenseViewMut view) {
    return py::array_t<T>({view.rows, view.cols}, {view.stride * sizeof(T), sizeof(T)}, static_cast<T*>(view.data));
}

PYBIND11_MODULE(kernel, m) {
    namespace py = pybind11;

    // ========== 绑定 Progress ==========
    py::class_<ProgressTracker>(m, "Progress")
        .def(py::init<size_t, size_t, bool>(), py::arg("total"), py::arg("nthreads"), py::arg("show_progress") = false)
        .def_property_readonly("total", &ProgressTracker::total)
        .def_property_readonly("nthreads", &ProgressTracker::nthreads)
        .def("buffer", [](ProgressTracker& self){
            return py::array_t<size_t>(
                static_cast<py::ssize_t>(self.nthreads()),
                self.ptr(),
                py::cast(&self, py::return_value_policy::reference)
            );
        })
        .def("aggregate", [](const ProgressTracker& self){
            size_t sum = 0;
            self.aggregate_and_notify([&](size_t s){ sum = s; });
            return sum;
        })
        .def("start_progress_display", &ProgressTracker::start_progress_display)
        .def("stop_progress_display", &ProgressTracker::stop_progress_display)
        .def("complete", &ProgressTracker::complete);

    // ========== 绑定 MannWhitneyResult ==========
    py::class_<hpdexc::MannWhitneyResult>(m, "MannWhitneyResult")
        .def(py::init<>())
        .def_property_readonly("U", [](const hpdexc::MannWhitneyResult& self) {
            return py::array_t<double>({self.U.rows, self.U.cols}, 
                                     {self.U.stride * sizeof(double), sizeof(double)}, 
                                     static_cast<double*>(self.U.data));
        })
        .def_property_readonly("P", [](const hpdexc::MannWhitneyResult& self) {
            return py::array_t<double>({self.P.rows, self.P.cols}, 
                                     {self.P.stride * sizeof(double), sizeof(double)}, 
                                     static_cast<double*>(self.P.data));
        })
        .def("set", &hpdexc::MannWhitneyResult::set);

    // ========== 绑定 mannwhitney ==========
    m.def(
        "mannwhitney",
        [](py::object sparse_matrix,
           py::array group_id,
           size_t n_targets,
           bool tie_correction,
           bool use_continuity,
           int alternative,
           int method,
           bool ref_sorted,
           bool tar_sorted,
           bool use_sparse_value,
           bool is_sparse_minmax,
           py::object sparse_value,
           bool use_histogram,
           uint64_t max_bins,
           uint64_t mem_budget_bytes,
           int threads,
           py::object progress,
           bool show_progress) 
        {
            // 1. CSC 视图
            hpdexc::CscView A = sparse_to_csc_view(sparse_matrix);

            // 2. 获取稀疏矩阵的 data dtype
            auto data_array = sparse_matrix.attr("data").cast<py::array>();
            auto data_dtype = data_array.dtype();

            // 3. group_id 类型检查（限制为 int32 或 int64）
            if (!py::dtype::of<int32_t>().is(group_id.dtype()) && 
                !py::dtype::of<int64_t>().is(group_id.dtype())) {
                throw std::runtime_error("group_id must be int32 or int64");
            }
            auto group_id_buf = group_id.request();
            hpdexc::DenseView group_ids(group_id_buf.ptr, group_id_buf.shape[0], 1);

            // 4. 可选进度缓冲
            void* progress_ptr = nullptr;
            ProgressTracker* tracker_ptr = nullptr;
            std::unique_ptr<ProgressTracker> auto_tracker = nullptr;
            
            if (!progress.is_none()) {
                if (py::isinstance<ProgressTracker>(progress)) {
                    auto& tracker = progress.cast<ProgressTracker&>();
                    progress_ptr = static_cast<void*>(tracker.ptr());
                    tracker_ptr = &tracker;
                } else {
                    auto arr = progress.cast<py::array>();
                    auto buf = arr.request(true);
                    progress_ptr = buf.ptr;
                }
            } else if (show_progress) {
                // 如果show_progress=True但没有传递progress参数，自动创建ProgressTracker
                auto_tracker = std::make_unique<ProgressTracker>(n_targets, threads, true);
                progress_ptr = static_cast<void*>(auto_tracker->ptr());
                tracker_ptr = auto_tracker.get();
            }

            // 5. dtype 分派（根据稀疏矩阵的 data dtype）
            #define MANNWHITNEY_DO(T) \
                if (py::dtype::of<T>().is(data_dtype)) { \
                    using Opt = hpdexc::MannWhitneyOption<T>; \
                    bool sparse_value_is_none = sparse_value.is_none(); \
                    auto alt = static_cast<typename Opt::Alternative>(alternative); \
                    auto meth = static_cast<typename Opt::Method>(method); \
                    auto sparse_val = sparse_value_is_none ? T{} : coerce_sparse_value_strict<T>(sparse_value); \
                    auto sparse_minmax = typename Opt::SparseValueMinmax(is_sparse_minmax); \
                    Opt opt; \
                    opt.ref_sorted = ref_sorted; \
                    opt.tar_sorted = tar_sorted; \
                    opt.tie_correction = tie_correction; \
                    opt.use_continuity = use_continuity; \
                    opt.use_sparse_value = use_sparse_value && !sparse_value_is_none; \
                    opt.is_spare_minmax = sparse_minmax; \
                    opt.sparse_value = sparse_val; \
                    opt.alternative = alt; \
                    opt.method = meth; \
                    if (tracker_ptr) { \
                        tracker_ptr->start_progress_display(); \
                    } \
                    auto result = hpdexc::mannwhitney<T>(A, opt, group_ids, n_targets, threads, progress_ptr); \
                    if (tracker_ptr) { \
                        tracker_ptr->complete(); \
                    } \
                    return result; \
                }

            HPDEXC_DTYPE_DISPATCH(MANNWHITNEY_DO)  // 只分派 int32 / int64

            #undef MANNWHITNEY_DO

            throw std::runtime_error("unsupported dtype for sparse matrix data");
        },
        py::arg("csc_matrix"),
        py::arg("group_id"),
        py::arg("n_targets"),
        py::arg("tie_correction") = true,
        py::arg("use_continuity") = true,
        py::arg("alternative") = 2,   // two-sided
        py::arg("method") = 0,        // auto
        py::arg("ref_sorted") = false,
        py::arg("tar_sorted") = false,
        py::arg("use_sparse_value") = false,
        py::arg("is_sparse_minmax") = false,
        py::arg("sparse_value") = py::none(),
        py::arg("use_histogram") = false,
        py::arg("max_bins") = (1ull<<16),
        py::arg("mem_budget_bytes") = (1ull<<30),
        py::arg("threads") = -1,
        py::arg("progress") = py::none(),
        py::arg("show_progress") = false
    );
}
