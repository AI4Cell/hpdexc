#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// 在这里包含你的算子声明头文件，例如：
// #include "core/rank_sum.hpp"
// #include "core/wilcoxon.hpp"

namespace py = pybind11;

PYBIND11_MODULE(hpdexc, m) {
    m.doc() = "High-performance DE backends (C++17, pybind11)";

    // 示例：导出一个占位函数，确认构建链路
    m.def("ping", []() { return std::string("hpdexc ok"); });

    // TODO: 这里 m.def 绑定你的 C++算子函数，例如：
    // m.def("wilcoxon_rank_sum_csc", &wilcoxon_rank_sum_csc, "...");
}
