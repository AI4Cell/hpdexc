#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>

// 前向声明，避免在头文件中包含 pybind11
namespace pybind11 {
    class array;
}

namespace hpdexc {

/**
 * @brief CSC (Compressed Sparse Column) 稀疏矩阵视图
 *
 * 用于描述稀疏矩阵的压缩列存储（CSC）格式，支持 int32/int64 索引的零拷贝访问。
 * 支持生命周期管理，可以保存 py::array 引用以防止数据被垃圾回收。
 */
 struct CscView {
    const void* indptr;      ///< 列指针数组 [cols+1]，类型为 int32* 或 int64*
    const void* indices;     ///< 行索引数组 [nnz]，类型为 int32* 或 int64*
    const void* data;        ///< 数据数组 [nnz]
    std::size_t rows;        ///< 行数
    std::size_t cols;        ///< 列数
    std::size_t nnz;         ///< 非零元素个数
    bool index_is_i64;       ///< true 表示索引为 int64，false 表示 int32

    // 生命周期管理：保存 py::array 引用以防止数据被垃圾回收
    std::shared_ptr<pybind11::array> indptr_owner;
    std::shared_ptr<pybind11::array> indices_owner;
    std::shared_ptr<pybind11::array> data_owner;

    /**
     * @brief 构造函数（不保存引用，用于临时视图）
     */
    CscView(const void* indptr_, const void* indices_, const void* data_,
            std::size_t rows_, std::size_t cols_, std::size_t nnz_,
            bool index_is_i64_ = false)
        : indptr(indptr_), indices(indices_), data(data_),
          rows(rows_), cols(cols_), nnz(nnz_), index_is_i64(index_is_i64_) {}

    /**
     * @brief 构造函数（保存引用，用于生命周期管理）
     */
    CscView(const void* indptr_, const void* indices_, const void* data_,
            std::size_t rows_, std::size_t cols_, std::size_t nnz_,
            bool index_is_i64_,
            std::shared_ptr<pybind11::array> indptr_owner_,
            std::shared_ptr<pybind11::array> indices_owner_,
            std::shared_ptr<pybind11::array> data_owner_)
        : indptr(indptr_), indices(indices_), data(data_),
          rows(rows_), cols(cols_), nnz(nnz_), index_is_i64(index_is_i64_),
          indptr_owner(indptr_owner_), indices_owner(indices_owner_), data_owner(data_owner_) {}

    /**
     * @brief 从三元组构造 CSC 视图（不保存引用）
     */
    static inline CscView from_triplets(const void* indptr_, const void* indices_, const void* data_,
                                 std::size_t rows_, std::size_t cols_, std::size_t nnz_,
                                 bool index_is_i64_ = false) {
        return CscView(indptr_, indices_, data_, rows_, cols_, nnz_, index_is_i64_);
    }

    /**
     * @brief 从三元组构造 CSC 视图（保存引用）
     */
    static inline CscView from_triplets_with_owners(const void* indptr_, const void* indices_, const void* data_,
                                                    std::size_t rows_, std::size_t cols_, std::size_t nnz_,
                                                    bool index_is_i64_,
                                                    std::shared_ptr<pybind11::array> indptr_owner_,
                                                    std::shared_ptr<pybind11::array> indices_owner_,
                                                    std::shared_ptr<pybind11::array> data_owner_) {
        return CscView(indptr_, indices_, data_, rows_, cols_, nnz_, index_is_i64_,
                      indptr_owner_, indices_owner_, data_owner_);
    }

    /**
     * @brief 获取数据指针（模板化）
     */
    template<typename T>
    inline const T* data_ptr() const { return static_cast<const T*>(data); }

    /**
     * @brief 获取 indptr[j]（根据类型分派）
     */
    inline int64_t indptr_at(std::size_t j) const {
        return index_is_i64
            ? reinterpret_cast<const int64_t*>(indptr)[j]
            : static_cast<int64_t>(reinterpret_cast<const int32_t*>(indptr)[j]);
    }

    /**
     * @brief 获取 indices[k]（根据类型分派）
     */
    inline int64_t row_at(std::size_t k) const {
        return index_is_i64
            ? reinterpret_cast<const int64_t*>(indices)[k]
            : static_cast<int64_t>(reinterpret_cast<const int32_t*>(indices)[k]);
    }

    /**
     * @brief 获取总的非零元素数
     */
    inline int64_t nnz_total() const {
        return indptr_at(static_cast<std::size_t>(cols));
    }
};

/**
 * @brief CSR (Compressed Sparse Row) 稀疏矩阵视图
 *
 * 用于描述稀疏矩阵的压缩行存储（CSR）格式，索引统一为 int64。
 */
struct CsrView {
    const int64_t* indptr;   ///< 行指针数组 [rows+1]
    const int64_t* indices;  ///< 列索引数组 [nnz]
    const void* data;        ///< 数据数组 [nnz]
    std::size_t rows;        ///< 行数
    std::size_t cols;        ///< 列数
    std::size_t nnz;         ///< 非零元素个数

    /**
     * @brief 构造函数
     */
    CsrView(const int64_t* indptr_, const int64_t* indices_, const void* data_,
            std::size_t rows_, std::size_t cols_, std::size_t nnz_)
        : indptr(indptr_), indices(indices_), data(data_),
          rows(rows_), cols(cols_), nnz(nnz_) {}

    /**
     * @brief 获取数据指针（模板化）
     */
    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data); }
};

/**
 * @brief 稠密矩阵只读视图
 */
struct DenseView {
    const void* data;           ///< 数据指针
    std::size_t rows;        ///< 行数
    std::size_t cols;        ///< 列数
    std::size_t stride;      ///< 行步长（默认为 cols）

    DenseView(const void* data_, std::size_t rows_, std::size_t cols_,
              std::size_t stride_ = 0)
        : data(data_), rows(rows_), cols(cols_),
          stride(stride_ ? stride_ : cols_) {}
};

/**
 * @brief 稠密矩阵可写视图
 */
struct DenseViewMut {
    void* data;                 ///< 数据指针
    std::size_t rows;        ///< 行数
    std::size_t cols;        ///< 列数
    std::size_t stride;      ///< 行步长（默认为 cols）

    DenseViewMut(void* data_, std::size_t rows_, std::size_t cols_,
                 std::size_t stride_ = 0)
        : data(data_), rows(rows_), cols(cols_),
          stride(stride_ ? stride_ : cols_) {}
};

}