// reader.hpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hpdexc {

struct CscOwned {
    std::vector<int64_t> indptr64;
    std::vector<int64_t> indices64;
    std::vector<double>  data;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t nnz  = 0;
};

// 读取 .h5ad 的 /X（假定为 CSR 或 CSC），输出统一为 CSC（int64 索引 + double 数据）。
// layout_code: 0 表示 /X 为 CSR，需要转换；1 表示 /X 已为 CSC，直接拷贝。
CscOwned read_total_csc(const std::string& path, int layout_code);

}  // namespace hpdexc


