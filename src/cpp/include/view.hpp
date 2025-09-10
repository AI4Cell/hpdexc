#pragma once
#include <cstdint>
#include <cstddef>

namespace hpdexc {

/**
 * @brief CSC (Compressed Sparse Column) 稀疏矩阵视图
 *
 * 用于描述稀疏矩阵的压缩列存储（CSC）格式，支持 int32 / int64 索引。
 */
 struct CscView {
    const void* indptr;      ///< 列指针数组 [cols+1]，类型为 int32* 或 int64*
    const void* indices;     ///< 行索引数组 [nnz]，类型为 int32* 或 int64*
    const void* data;        ///< 数据数组 [nnz]
    std::size_t rows;        ///< 行数
    std::size_t cols;        ///< 列数
    std::size_t nnz;         ///< 非零元素个数
    bool index_is_i64 = false; ///< true 表示索引为 int64，false 表示 int32

    /**
     * @brief 构造函数
     */
    CscView(const void* indptr_, const void* indices_, const void* data_,
            std::size_t rows_, std::size_t cols_, std::size_t nnz_,
            bool index_is_i64_ = false)
        : indptr(indptr_), indices(indices_), data(data_),
          rows(rows_), cols(cols_), nnz(nnz_), index_is_i64(index_is_i64_) {}

    /**
     * @brief 从三元组构造 CSC 视图
     */
    static inline CscView from_triplets(const void* indptr_, const void* indices_, const void* data_,
                                 std::size_t rows_, std::size_t cols_, std::size_t nnz_,
                                 bool index_is_i64_ = false,
                                 bool indptr_owned = false,
                                 bool indices_owned = false,
                                 bool data_owned = false,
                                 bool data_writable = false) {
        return CscView(indptr_, indices_, data_, rows_, cols_, nnz_, index_is_i64_);
    }

    /**
     * @brief 获取数据指针（模板化）
     */
    template<typename T>
    inline const T* data_ptr() const { return static_cast<const T*>(data); }

    /**
     * @brief 获取 indptr[j]
     */
    inline int64_t indptr_at(std::size_t j) const {
        return index_is_i64
            ? reinterpret_cast<const int64_t*>(indptr)[j]
            : static_cast<int64_t>(reinterpret_cast<const int32_t*>(indptr)[j]);
    }

    /**
     * @brief 获取 indices[k]
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