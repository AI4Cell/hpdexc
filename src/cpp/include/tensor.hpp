#pragma once
#include "common.hpp"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <cstring> // for std::memcpy

namespace hpdexc::tensor {

// 索引类型定义
using index32_t = int32_t;
using index_t   = int64_t;

// ========================== Vector / Ndarray ==========================

template<class T>
class Vector {
public:
    using value_type = T;

    FORCE_INLINE Vector(const T* data, std::size_t size, std::shared_ptr<void> owner = nullptr) noexcept
      : data_(data), size_(size), owner_(std::move(owner)) {}

    FORCE_INLINE const T* data()  const noexcept { return data_; }
    FORCE_INLINE T*       data()        noexcept { return const_cast<T*>(data_); }
    FORCE_INLINE std::size_t size() const noexcept { return size_; }

    // ---- 静态构造（使用 AlignedVector 作为 owner）----
    FORCE_INLINE static Vector<T> zeros(std::size_t n) {
        auto buf = std::make_shared<AlignedVector<T>>(n, T(0));
        return Vector<T>(buf->data(), n, buf);
    }
    FORCE_INLINE static Vector<T> ones(std::size_t n) {
        auto buf = std::make_shared<AlignedVector<T>>(n, T(1));
        return Vector<T>(buf->data(), n, buf);
    }
    FORCE_INLINE static Vector<T> full(std::size_t n, T value) {
        auto buf = std::make_shared<AlignedVector<T>>(n, value);
        return Vector<T>(buf->data(), n, buf);
    }
    FORCE_INLINE static Vector<T> from_std(const std::vector<T>& v) {
        auto buf = std::make_shared<AlignedVector<T>>(v.size());
        std::memcpy(buf->data(), v.data(), v.size() * sizeof(T));
        return Vector<T>(buf->data(), buf->size(), buf);
    }

private:
    const T* data_;
    std::size_t size_;
    std::shared_ptr<void> owner_;
};

template<class T>
class Ndarray {
public:
    using value_type = T;

    FORCE_INLINE Ndarray() noexcept
      : data_(nullptr),
        shape_({}),
        stride_({}),
        size_(0),
        owner_(nullptr) {}

    FORCE_INLINE Ndarray(const T* data,
            std::vector<std::size_t> shape,
            std::vector<std::size_t> stride = {},
            std::shared_ptr<void> owner = nullptr)
      : data_(data),
        shape_(std::move(shape)),
        stride_(stride.empty() ? default_stride(shape_) : std::move(stride)),
        size_(compute_size(shape_)),
        owner_(std::move(owner)) {}

    FORCE_INLINE const T* data() const noexcept { return data_; }
    FORCE_INLINE T*       data()       noexcept { return const_cast<T*>(data_); }

    FORCE_INLINE const std::vector<std::size_t>& shape()  const noexcept { return shape_;  }
    FORCE_INLINE const std::vector<std::size_t>& stride() const noexcept { return stride_; }
    FORCE_INLINE std::size_t ndim() const noexcept { return shape_.size(); }
    FORCE_INLINE std::size_t size() const noexcept { return size_; }

    // ---- 静态构造（AlignedVector owner）----
    FORCE_INLINE static Ndarray<T> zeros(const std::vector<std::size_t>& shape) {
        auto buf = std::make_shared<AlignedVector<T>>(compute_size(shape), T(0));
        return Ndarray<T>(buf->data(), shape, {}, buf);
    }
    FORCE_INLINE static Ndarray<T> ones(const std::vector<std::size_t>& shape) {
        auto buf = std::make_shared<AlignedVector<T>>(compute_size(shape), T(1));
        return Ndarray<T>(buf->data(), shape, {}, buf);
    }
    FORCE_INLINE static Ndarray<T> full(const std::vector<std::size_t>& shape, T value) {
        auto buf = std::make_shared<AlignedVector<T>>(compute_size(shape), value);
        return Ndarray<T>(buf->data(), shape, {}, buf);
    }
    FORCE_INLINE static Ndarray<T> from_std(const std::vector<T>& v, const std::vector<std::size_t>& shape) {
        if (compute_size(shape) != v.size()) {
            throw std::runtime_error("Ndarray::from_std: size mismatch");
        }
        auto buf = std::make_shared<AlignedVector<T>>(v.size());
        std::memcpy(buf->data(), v.data(), v.size() * sizeof(T));
        return Ndarray<T>(buf->data(), shape, {}, buf);
    }

private:
    static std::vector<std::size_t> default_stride(const std::vector<std::size_t>& shape) {
        std::vector<std::size_t> s(shape.size());
        std::size_t step = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            s[i] = step;
            step *= shape[i];
        }
        return s;
    }
    static std::size_t compute_size(const std::vector<std::size_t>& shape) {
        std::size_t total = 1;
        for (auto d : shape) total *= d;
        return total;
    }

    const T* data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;
    std::size_t size_;
    std::shared_ptr<void> owner_;
};

// ========================== CSC ==========================

template<class T, class Idx>
class Csc {
public:
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "Csc: Idx must be int32_t or int64_t");

    using value_type = T;
    using index_type = Idx;

    FORCE_INLINE Csc(const T* data, const Idx* indptr, const Idx* indices,
        std::size_t rows, std::size_t cols, std::size_t nnz,
        std::shared_ptr<void> indptr_owner = nullptr,
        std::shared_ptr<void> indices_owner = nullptr,
        std::shared_ptr<void> data_owner = nullptr) noexcept
      : data_(data), indptr_(indptr), indices_(indices),
        rows_(rows), cols_(cols), nnz_(nnz),
        indptr_owner_(std::move(indptr_owner)),
        indices_owner_(std::move(indices_owner)),
        data_owner_(std::move(data_owner)) {}

    FORCE_INLINE const T*   data()    const noexcept { return data_; }
    FORCE_INLINE const Idx* indptr()  const noexcept { return indptr_; }
    FORCE_INLINE const Idx* indices() const noexcept { return indices_; }
    FORCE_INLINE T*         data()          noexcept { return const_cast<T*>(data_); }
    FORCE_INLINE Idx*       indptr()        noexcept { return const_cast<Idx*>(indptr_); }
    FORCE_INLINE Idx*       indices()       noexcept { return const_cast<Idx*>(indices_); }

    FORCE_INLINE Idx indptr_at(std::size_t j)  const noexcept { return indptr_[j]; }
    FORCE_INLINE Idx indices_at(std::size_t k) const noexcept { return indices_[k]; }

    FORCE_INLINE Idx          nnz_total() const noexcept { return indptr_[cols_]; }
    FORCE_INLINE std::size_t  rows()      const noexcept { return rows_; }
    FORCE_INLINE std::size_t  cols()      const noexcept { return cols_; }
    FORCE_INLINE std::size_t  nnz()       const noexcept { return nnz_; }
    static constexpr bool is_index32() noexcept { return std::is_same_v<Idx, int32_t>; }

    FORCE_INLINE const std::shared_ptr<void>& indptr_owner() const noexcept { return indptr_owner_; }
    FORCE_INLINE const std::shared_ptr<void>& indices_owner() const noexcept { return indices_owner_; }
    FORCE_INLINE const std::shared_ptr<void>& data_owner()    const noexcept { return data_owner_; }

    // ---- 从稠密矩阵构造（仅支持 2D，行主序）----
    static Csc<T,Idx> from_dense(const Ndarray<T>& arr, T sparse_value = T(0));

private:
    const T*   data_;
    const Idx* indptr_;
    const Idx* indices_;
    std::size_t rows_, cols_, nnz_;
    std::shared_ptr<void> indptr_owner_;
    std::shared_ptr<void> indices_owner_;
    std::shared_ptr<void> data_owner_;
};

// ========================== CSR ==========================

template<class T, class Idx>
class Csr {
public:
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "Csr: Idx must be int32_t or int64_t");

    using value_type = T;
    using index_type = Idx;

    FORCE_INLINE Csr(const T* data, const Idx* indptr, const Idx* indices,
        std::size_t rows, std::size_t cols, std::size_t nnz,
        std::shared_ptr<void> indptr_owner = nullptr,
        std::shared_ptr<void> indices_owner = nullptr,
        std::shared_ptr<void> data_owner = nullptr) noexcept
      : data_(data), indptr_(indptr), indices_(indices),
        rows_(rows), cols_(cols), nnz_(nnz),
        indptr_owner_(std::move(indptr_owner)),
        indices_owner_(std::move(indices_owner)),
        data_owner_(std::move(data_owner)) {}

    FORCE_INLINE const T*   data()    const noexcept { return data_; }
    FORCE_INLINE const Idx* indptr()  const noexcept { return indptr_; }
    FORCE_INLINE const Idx* indices() const noexcept { return indices_; }
    FORCE_INLINE T*         data()          noexcept { return const_cast<T*>(data_); }
    FORCE_INLINE Idx*       indptr()        noexcept { return const_cast<Idx*>(indptr_); }
    FORCE_INLINE Idx*       indices()       noexcept { return const_cast<Idx*>(indices_); }

    FORCE_INLINE Idx indptr_at(std::size_t i)  const noexcept { return indptr_[i]; }
    FORCE_INLINE Idx indices_at(std::size_t k) const noexcept { return indices_[k]; }

    FORCE_INLINE Idx          nnz_total() const noexcept { return indptr_[rows_]; }
    FORCE_INLINE std::size_t  rows()      const noexcept { return rows_; }
    FORCE_INLINE std::size_t  cols()      const noexcept { return cols_; }
    FORCE_INLINE std::size_t  nnz()       const noexcept { return nnz_; }
    static constexpr bool is_index32() noexcept { return std::is_same_v<Idx, int32_t>; }

    FORCE_INLINE const std::shared_ptr<void>& indptr_owner() const noexcept { return indptr_owner_; }
    FORCE_INLINE const std::shared_ptr<void>& indices_owner() const noexcept { return indices_owner_; }
    FORCE_INLINE const std::shared_ptr<void>& data_owner()    const noexcept { return data_owner_; }

    // ---- 从稠密矩阵构造（仅支持 2D，行主序）----
    static Csr<T,Idx> from_dense(const Ndarray<T>& arr, T sparse_value = T(0));

private:
    const T*   data_;
    const Idx* indptr_;
    const Idx* indices_;
    std::size_t rows_, cols_, nnz_;
    std::shared_ptr<void> indptr_owner_;
    std::shared_ptr<void> indices_owner_;
    std::shared_ptr<void> data_owner_;
};

// ========================== COO ==========================

template<class T, class Idx>
class Coo {
public:
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "Coo: Idx must be int32_t or int64_t");

    using value_type = T;
    using index_type = Idx;

    FORCE_INLINE Coo(const T* data, const Idx* row, const Idx* col,
        std::size_t rows, std::size_t cols, std::size_t nnz,
        std::shared_ptr<void> row_owner = nullptr,
        std::shared_ptr<void> col_owner = nullptr,
        std::shared_ptr<void> data_owner = nullptr) noexcept
      : data_(data), row_(row), col_(col),
        rows_(rows), cols_(cols), nnz_(nnz),
        row_owner_(std::move(row_owner)),
        col_owner_(std::move(col_owner)),
        data_owner_(std::move(data_owner)) {}

    FORCE_INLINE const T*   data() const noexcept { return data_; }
    FORCE_INLINE const Idx* row()  const noexcept { return row_;  }
    FORCE_INLINE const Idx* col()  const noexcept { return col_;  }
    FORCE_INLINE T*         data()       noexcept { return const_cast<T*>(data_); }
    FORCE_INLINE Idx*       row()        noexcept { return const_cast<Idx*>(row_); }
    FORCE_INLINE Idx*       col()        noexcept { return const_cast<Idx*>(col_); }

    FORCE_INLINE Idx row_at(std::size_t k) const noexcept { return row_[k]; }
    FORCE_INLINE Idx col_at(std::size_t k) const noexcept { return col_[k]; }

    FORCE_INLINE std::size_t rows() const noexcept { return rows_; }
    FORCE_INLINE std::size_t cols() const noexcept { return cols_; }
    FORCE_INLINE std::size_t nnz()  const noexcept { return nnz_; }

    FORCE_INLINE const std::shared_ptr<void>& row_owner()  const noexcept { return row_owner_; }
    FORCE_INLINE const std::shared_ptr<void>& col_owner()  const noexcept { return col_owner_; }
    FORCE_INLINE const std::shared_ptr<void>& data_owner() const noexcept { return data_owner_; }

    // ---- 从稠密矩阵构造（仅支持 2D，尊重 stride）----
    template<class U = T>
    FORCE_INLINE static Coo<T,Idx> from_dense(const Ndarray<U>& arr, U sparse_value = U(0)) {
        if (arr.ndim() != 2) {
            throw std::runtime_error("Coo::from_dense: only 2D supported");
        }
        const auto& shp = arr.shape();
        const auto& str = arr.stride();
        const std::size_t R = shp[0], C = shp[1];

        auto row_buf = std::make_shared<AlignedVector<Idx>>();
        auto col_buf = std::make_shared<AlignedVector<Idx>>();
        auto data_buf= std::make_shared<AlignedVector<T>>();
        const std::size_t est = (R * C) / 10 + 1;
        row_buf->reserve(est);
        col_buf->reserve(est);
        data_buf->reserve(est);

        const U* d = arr.data();
        for (std::size_t i = 0; i < R; ++i) {
            for (std::size_t j = 0; j < C; ++j) {
                const std::size_t off = i * str[0] + j * str[1];
                U v = d[off];
                if (v != sparse_value) {
                    row_buf->push_back(static_cast<Idx>(i));
                    col_buf->push_back(static_cast<Idx>(j));
                    data_buf->push_back(static_cast<T>(v));
                }
            }
        }

        return Coo<T,Idx>(data_buf->data(), row_buf->data(), col_buf->data(),
                          R, C, data_buf->size(), row_buf, col_buf, data_buf);
    }

private:
    const T*   data_;
    const Idx* row_;
    const Idx* col_;
    std::size_t rows_, cols_, nnz_;
    std::shared_ptr<void> row_owner_;
    std::shared_ptr<void> col_owner_;
    std::shared_ptr<void> data_owner_;
};

// ====================== 便捷别名（可选） ======================
template<class T> using Csc32 = Csc<T, int32_t>;
template<class T> using Csc64 = Csc<T, int64_t>;
template<class T> using Csr32 = Csr<T, int32_t>;
template<class T> using Csr64 = Csr<T, int64_t>;
template<class T> using Coo32 = Coo<T, int32_t>;
template<class T> using Coo64 = Coo<T, int64_t>;

// ==================== 稀疏格式互转（header-only） ====================

// COO -> CSR
template<class T, class Idx>
Csr<T,Idx> coo_to_csr(const Coo<T,Idx>& coo) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "coo_to_csr: Idx must be int32_t or int64_t");

    const std::size_t R = coo.rows(), C = coo.cols(), NNZ = coo.nnz();

    auto indptr_vec  = std::make_shared<AlignedVector<Idx>>(R + 1, (Idx)0);
    auto indices_vec = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto data_vec    = std::make_shared<AlignedVector<T>>(NNZ);

    const Idx* row = coo.row();
    const Idx* col = coo.col();
    const T*   val = coo.data();

    for (std::size_t k = 0; k < NNZ; ++k) {
        if (row[k] < 0 || static_cast<std::size_t>(row[k]) >= R)
            throw std::runtime_error("coo_to_csr: row out of range");
        ++(*indptr_vec)[ static_cast<std::size_t>(row[k]) + 1 ];
    }
    for (std::size_t r = 0; r < R; ++r) {
        (*indptr_vec)[r+1] = (*indptr_vec)[r+1] + (*indptr_vec)[r];
    }

    AlignedVector<Idx> next(indptr_vec->size());
    std::memcpy(next.data(), indptr_vec->data(), indptr_vec->size() * sizeof(Idx));

    for (std::size_t k = 0; k < NNZ; ++k) {
        if (row[k] < 0 || static_cast<std::size_t>(row[k]) >= R)
            throw std::runtime_error("coo_to_csr: row out of range (fill)");
        const auto r = static_cast<std::size_t>(row[k]);
        const auto dst = static_cast<std::size_t>(next[r]++);
        (*indices_vec)[dst] = col[k];
        (*data_vec)[dst]    = val[k];
    }

    (void)C;
    return Csr<T,Idx>(data_vec->data(), indptr_vec->data(), indices_vec->data(),
                      R, C, NNZ, indptr_vec, indices_vec, data_vec);
}

// COO -> CSC
template<class T, class Idx>
Csc<T,Idx> coo_to_csc(const Coo<T,Idx>& coo) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "coo_to_csc: Idx must be int32_t or int64_t");

    const std::size_t R = coo.rows(), C = coo.cols(), NNZ = coo.nnz();

    auto indptr_vec  = std::make_shared<AlignedVector<Idx>>(C + 1, (Idx)0);
    auto indices_vec = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto data_vec    = std::make_shared<AlignedVector<T>>(NNZ);

    const Idx* row = coo.row();
    const Idx* col = coo.col();
    const T*   val = coo.data();

    for (std::size_t k = 0; k < NNZ; ++k) {
        if (col[k] < 0 || static_cast<std::size_t>(col[k]) >= C)
            throw std::runtime_error("coo_to_csc: col out of range");
        ++(*indptr_vec)[ static_cast<std::size_t>(col[k]) + 1 ];
    }
    for (std::size_t c = 0; c < C; ++c) {
        (*indptr_vec)[c+1] = (*indptr_vec)[c+1] + (*indptr_vec)[c];
    }

    AlignedVector<Idx> next(indptr_vec->size());
    std::memcpy(next.data(), indptr_vec->data(), indptr_vec->size() * sizeof(Idx));

    for (std::size_t k = 0; k < NNZ; ++k) {
        if (col[k] < 0 || static_cast<std::size_t>(col[k]) >= C)
            throw std::runtime_error("coo_to_csc: col out of range (fill)");
        const auto c = static_cast<std::size_t>(col[k]);
        const auto dst = static_cast<std::size_t>(next[c]++);
        (*indices_vec)[dst] = row[k];
        (*data_vec)[dst]    = val[k];
    }

    return Csc<T,Idx>(data_vec->data(), indptr_vec->data(), indices_vec->data(),
                      R, C, NNZ, indptr_vec, indices_vec, data_vec);
}

// CSR -> CSC
template<class T, class Idx>
Csc<T,Idx> csr_to_csc(const Csr<T,Idx>& csr) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "csr_to_csc: Idx must be int32_t or int64_t");

    const std::size_t R = csr.rows(), C = csr.cols(), NNZ = csr.nnz();

    auto indptr_vec  = std::make_shared<AlignedVector<Idx>>(C + 1, (Idx)0);
    auto indices_vec = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto data_vec    = std::make_shared<AlignedVector<T>>(NNZ);

    const Idx* rp = csr.indptr();
    const Idx* ci = csr.indices();
    const T*   dv = csr.data();

    for (std::size_t r = 0; r < R; ++r) {
        const std::size_t start = static_cast<std::size_t>(rp[r]);
        const std::size_t end   = static_cast<std::size_t>(rp[r+1]);
        for (std::size_t k = start; k < end; ++k) {
            if (ci[k] < 0 || static_cast<std::size_t>(ci[k]) >= C)
                throw std::runtime_error("csr_to_csc: col out of range");
            ++(*indptr_vec)[ static_cast<std::size_t>(ci[k]) + 1 ];
        }
    }
    for (std::size_t c = 0; c < C; ++c) {
        (*indptr_vec)[c+1] = (*indptr_vec)[c+1] + (*indptr_vec)[c];
    }

    AlignedVector<Idx> next(indptr_vec->size());
    std::memcpy(next.data(), indptr_vec->data(), indptr_vec->size() * sizeof(Idx));

    for (std::size_t r = 0; r < R; ++r) {
        const std::size_t start = static_cast<std::size_t>(rp[r]);
        const std::size_t end   = static_cast<std::size_t>(rp[r+1]);
        for (std::size_t k = start; k < end; ++k) {
            const auto c = static_cast<std::size_t>(ci[k]);
            const std::size_t dst = static_cast<std::size_t>(next[c]++);
            (*indices_vec)[dst] = static_cast<Idx>(r);
            (*data_vec)[dst]    = dv[k];
        }
    }

    return Csc<T,Idx>(data_vec->data(), indptr_vec->data(), indices_vec->data(),
                      R, C, NNZ, indptr_vec, indices_vec, data_vec);
}

// CSC -> CSR
template<class T, class Idx>
Csr<T,Idx> csc_to_csr(const Csc<T,Idx>& csc) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "csc_to_csr: Idx must be int32_t or int64_t");

    const std::size_t R = csc.rows(), C = csc.cols(), NNZ = csc.nnz();

    auto indptr_vec  = std::make_shared<AlignedVector<Idx>>(R + 1, (Idx)0);
    auto indices_vec = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto data_vec    = std::make_shared<AlignedVector<T>>(NNZ);

    const Idx* cp = csc.indptr();
    const Idx* ri = csc.indices();
    const T*   dv = csc.data();

    for (std::size_t c = 0; c < C; ++c) {
        const std::size_t start = static_cast<std::size_t>(cp[c]);
        const std::size_t end   = static_cast<std::size_t>(cp[c+1]);
        for (std::size_t k = start; k < end; ++k) {
            if (ri[k] < 0 || static_cast<std::size_t>(ri[k]) >= R)
                throw std::runtime_error("csc_to_csr: row out of range");
            ++(*indptr_vec)[ static_cast<std::size_t>(ri[k]) + 1 ];
        }
    }
    for (std::size_t r = 0; r < R; ++r) {
        (*indptr_vec)[r+1] = (*indptr_vec)[r+1] + (*indptr_vec)[r];
    }

    AlignedVector<Idx> next(indptr_vec->size());
    std::memcpy(next.data(), indptr_vec->data(), indptr_vec->size() * sizeof(Idx));

    for (std::size_t c = 0; c < C; ++c) {
        const std::size_t start = static_cast<std::size_t>(cp[c]);
        const std::size_t end   = static_cast<std::size_t>(cp[c+1]);
        for (std::size_t k = start; k < end; ++k) {
            if (ri[k] < 0 || static_cast<std::size_t>(ri[k]) >= R)
                throw std::runtime_error("csc_to_csr: row out of range (fill)");
            const auto r = static_cast<std::size_t>(ri[k]);
            const std::size_t dst = static_cast<std::size_t>(next[r]++);
            (*indices_vec)[dst] = static_cast<Idx>(c);
            (*data_vec)[dst]    = dv[k];
        }
    }

    return Csr<T,Idx>(data_vec->data(), indptr_vec->data(), indices_vec->data(),
                      R, C, NNZ, indptr_vec, indices_vec, data_vec);
}

// CSR -> COO
template<class T, class Idx>
Coo<T,Idx> csr_to_coo(const Csr<T,Idx>& csr) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "csr_to_coo: Idx must be int32_t or int64_t");

    const std::size_t R = csr.rows(), C = csr.cols(), NNZ = csr.nnz();

    auto row_vec  = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto col_vec  = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto data_vec = std::make_shared<AlignedVector<T>>(NNZ);

    const Idx* rp = csr.indptr();
    const Idx* ci = csr.indices();
    const T*   dv = csr.data();

    for (std::size_t r = 0; r < R; ++r) {
        const std::size_t start = static_cast<std::size_t>(rp[r]);
        const std::size_t end   = static_cast<std::size_t>(rp[r+1]);
        for (std::size_t k = start; k < end; ++k) {
            (*row_vec)[k]  = static_cast<Idx>(r);
            (*col_vec)[k]  = ci[k];
            (*data_vec)[k] = dv[k];
        }
    }

    (void)C;
    return Coo<T,Idx>(data_vec->data(), row_vec->data(), col_vec->data(),
                      R, C, NNZ, row_vec, col_vec, data_vec);
}

// CSC -> COO
template<class T, class Idx>
Coo<T,Idx> csc_to_coo(const Csc<T,Idx>& csc) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "csc_to_coo: Idx must be int32_t or int64_t");

    const std::size_t R = csc.rows(), C = csc.cols(), NNZ = csc.nnz();

    auto row_vec  = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto col_vec  = std::make_shared<AlignedVector<Idx>>(NNZ);
    auto data_vec = std::make_shared<AlignedVector<T>>(NNZ);

    const Idx* cp = csc.indptr();
    const Idx* ri = csc.indices();
    const T*   dv = csc.data();

    std::size_t k = 0;
    for (std::size_t c = 0; c < C; ++c) {
        const std::size_t start = static_cast<std::size_t>(cp[c]);
        const std::size_t end   = static_cast<std::size_t>(cp[c+1]);
        for (std::size_t p = start; p < end; ++p, ++k) {
            (*row_vec)[k]  = ri[p];
            (*col_vec)[k]  = static_cast<Idx>(c);
            (*data_vec)[k] = dv[p];
        }
    }

    (void)R;
    return Coo<T,Idx>(data_vec->data(), row_vec->data(), col_vec->data(),
                      R, C, NNZ, row_vec, col_vec, data_vec);
}

// ==================== 稀疏 from_dense 类外定义 ====================
template<class T, class Idx>
FORCE_INLINE Csr<T,Idx>
Csr<T,Idx>::from_dense(const Ndarray<T>& arr, T sparse_value) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "Csr::from_dense: Idx must be int32_t or int64_t");
    auto coo = Coo<T,Idx>::from_dense(arr, sparse_value);
    return coo_to_csr<T,Idx>(coo);
}

template<class T, class Idx>
FORCE_INLINE Csc<T,Idx>
Csc<T,Idx>::from_dense(const Ndarray<T>& arr, T sparse_value) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "Csc::from_dense: Idx must be int32_t or int64_t");
    auto coo = Coo<T,Idx>::from_dense(arr, sparse_value);
    return coo_to_csc<T,Idx>(coo);
}

} // namespace hpdexc::tensor

// ========================= Python 互操作 =========================
#if HPDEXC_PYBIND
namespace hpdexc::tensor_py {

namespace py = pybind11;
using namespace hpdexc::tensor;

// GIL 安全持有 PyObject*：避免跨线程析构导致崩溃
inline std::shared_ptr<void> keep_alive_pyobject(const py::object& o) {
    PyObject* p = o.ptr();
    Py_INCREF(p);
    return std::shared_ptr<void>(
        p,
        [](void* vp){
            py::gil_scoped_acquire gil;
            Py_DECREF(reinterpret_cast<PyObject*>(vp));
        }
    );
}

// ---- dtype helpers ----
FORCE_INLINE bool is_index32(const py::array& a) noexcept {
    return py::dtype::of<int32_t>().is(a.dtype());
}
template<class T>
FORCE_INLINE bool is_value_dtype(const py::dtype& d) noexcept {
    return py::dtype::of<T>().is(d);
}

// -------------------- from_scipy_*（强类型） --------------------
template<class T, class Idx>
Csc<T,Idx> from_scipy_csc_t(py::object m) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "from_scipy_csc_t: Idx must be int32_t or int64_t");

    auto indptr  = m.attr("indptr").cast<py::array>();
    auto indices = m.attr("indices").cast<py::array>();
    auto data    = m.attr("data").cast<py::array>();
    auto shape   = m.attr("shape").cast<std::pair<std::size_t,std::size_t>>();

    if (!is_value_dtype<T>(data.dtype()))   throw std::runtime_error("Csc: data dtype mismatch");
    if (!py::dtype::of<Idx>().is(indptr.dtype()) ||
        !py::dtype::of<Idx>().is(indices.dtype())) {
        throw std::runtime_error("Csc: index dtype mismatch");
    }

    auto ib = indptr.request(); auto jb = indices.request(); auto db = data.request();
    return Csc<T,Idx>(
        static_cast<const T*>(db.ptr),
        static_cast<const Idx*>(ib.ptr),
        static_cast<const Idx*>(jb.ptr),
        shape.first, shape.second,
        static_cast<std::size_t>(db.size),
        keep_alive_pyobject(indptr),
        keep_alive_pyobject(indices),
        keep_alive_pyobject(data)
    );
}

template<class T, class Idx>
Csr<T,Idx> from_scipy_csr_t(py::object m) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "from_scipy_csr_t: Idx must be int32_t or int64_t");

    auto indptr  = m.attr("indptr").cast<py::array>();
    auto indices = m.attr("indices").cast<py::array>();
    auto data    = m.attr("data").cast<py::array>();
    auto shape   = m.attr("shape").cast<std::pair<std::size_t,std::size_t>>();

    if (!is_value_dtype<T>(data.dtype()))   throw std::runtime_error("Csr: data dtype mismatch");
    if (!py::dtype::of<Idx>().is(indptr.dtype()) ||
        !py::dtype::of<Idx>().is(indices.dtype())) {
        throw std::runtime_error("Csr: index dtype mismatch");
    }

    auto ib = indptr.request(); auto jb = indices.request(); auto db = data.request();
    return Csr<T,Idx>(
        static_cast<const T*>(db.ptr),
        static_cast<const Idx*>(ib.ptr),
        static_cast<const Idx*>(jb.ptr),
        shape.first, shape.second,
        static_cast<std::size_t>(db.size),
        keep_alive_pyobject(indptr),
        keep_alive_pyobject(indices),
        keep_alive_pyobject(data)
    );
}

template<class T, class Idx>
Coo<T,Idx> from_scipy_coo_t(py::object m) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "from_scipy_coo_t: Idx must be int32_t or int64_t");

    auto row   = m.attr("row").cast<py::array>();
    auto col   = m.attr("col").cast<py::array>();
    auto data  = m.attr("data").cast<py::array>();
    auto shape = m.attr("shape").cast<std::pair<std::size_t,std::size_t>>();

    if (!is_value_dtype<T>(data.dtype())) throw std::runtime_error("Coo: data dtype mismatch");
    if (!py::dtype::of<Idx>().is(row.dtype()) || !py::dtype::of<Idx>().is(col.dtype()))
        throw std::runtime_error("Coo: index dtype mismatch");

    auto rb = row.request(); auto cb = col.request(); auto db = data.request();
    return Coo<T,Idx>(
        static_cast<const T*>(db.ptr),
        static_cast<const Idx*>(rb.ptr),
        static_cast<const Idx*>(cb.ptr),
        shape.first, shape.second,
        static_cast<std::size_t>(db.size),
        keep_alive_pyobject(row),
        keep_alive_pyobject(col),
        keep_alive_pyobject(data)
    );
}

// -------------------- to_scipy_*（强类型，零拷贝且绑定生命周期） --------------------
template<class T, class Idx>
py::object to_scipy_csc(const Csc<T,Idx>& A) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "to_scipy_csc: Idx must be int32_t or int64_t");

    auto make_base = [](const std::shared_ptr<void>& owner){
        auto* holder = new std::shared_ptr<void>(owner);
        return py::capsule(holder, [](void* p){ delete static_cast<std::shared_ptr<void>*>(p); });
    };
    py::capsule base_indptr  = make_base(A.indptr_owner());
    py::capsule base_indices = make_base(A.indices_owner());
    py::capsule base_data    = make_base(A.data_owner());

    auto indptr = py::array(
        py::buffer_info(const_cast<Idx*>(A.indptr()), sizeof(Idx),
                        py::format_descriptor<Idx>::format(), 1,
                        {A.cols() + 1}, {sizeof(Idx)}),
        base_indptr
    );
    auto indices = py::array(
        py::buffer_info(const_cast<Idx*>(A.indices()), sizeof(Idx),
                        py::format_descriptor<Idx>::format(), 1,
                        {A.nnz()}, {sizeof(Idx)}),
        base_indices
    );
    auto data = py::array(
        py::buffer_info(const_cast<T*>(A.data()), sizeof(T),
                        py::format_descriptor<T>::format(), 1,
                        {A.nnz()}, {sizeof(T)}),
        base_data
    );

    auto sp = py::module_::import("scipy.sparse");
    return sp.attr("csc_matrix")(py::make_tuple(data, indices, indptr),
                                 py::make_tuple(A.rows(), A.cols()));
}

template<class T, class Idx>
py::object to_scipy_csr(const Csr<T,Idx>& A) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "to_scipy_csr: Idx must be int32_t or int64_t");

    auto make_base = [](const std::shared_ptr<void>& owner){
        auto* holder = new std::shared_ptr<void>(owner);
        return py::capsule(holder, [](void* p){ delete static_cast<std::shared_ptr<void>*>(p); });
    };
    py::capsule base_indptr  = make_base(A.indptr_owner());
    py::capsule base_indices = make_base(A.indices_owner());
    py::capsule base_data    = make_base(A.data_owner());

    auto indptr = py::array(
        py::buffer_info(const_cast<Idx*>(A.indptr()), sizeof(Idx),
                        py::format_descriptor<Idx>::format(), 1,
                        {A.rows() + 1}, {sizeof(Idx)}),
        base_indptr
    );
    auto indices = py::array(
        py::buffer_info(const_cast<Idx*>(A.indices()), sizeof(Idx),
                        py::format_descriptor<Idx>::format(), 1,
                        {A.nnz()}, {sizeof(Idx)}),
        base_indices
    );
    auto data = py::array(
        py::buffer_info(const_cast<T*>(A.data()), sizeof(T),
                        py::format_descriptor<T>::format(), 1,
                        {A.nnz()}, {sizeof(T)}),
        base_data
    );

    auto sp = py::module_::import("scipy.sparse");
    return sp.attr("csr_matrix")(py::make_tuple(data, indices, indptr),
                                 py::make_tuple(A.rows(), A.cols()));
}

template<class T, class Idx>
py::object to_scipy_coo(const Coo<T,Idx>& A) {
    static_assert(std::is_same_v<Idx,int32_t> || std::is_same_v<Idx,int64_t>,
                  "to_scipy_coo: Idx must be int32_t or int64_t");

    auto make_base = [](const std::shared_ptr<void>& owner){
        auto* holder = new std::shared_ptr<void>(owner);
        return py::capsule(holder, [](void* p){ delete static_cast<std::shared_ptr<void>*>(p); });
    };
    py::capsule base_row   = make_base(A.row_owner());
    py::capsule base_col   = make_base(A.col_owner());
    py::capsule base_data  = make_base(A.data_owner());

    auto row = py::array(
        py::buffer_info(const_cast<Idx*>(A.row()), sizeof(Idx),
                        py::format_descriptor<Idx>::format(), 1,
                        {A.nnz()}, {sizeof(Idx)}),
        base_row
    );
    auto col = py::array(
        py::buffer_info(const_cast<Idx*>(A.col()), sizeof(Idx),
                        py::format_descriptor<Idx>::format(), 1,
                        {A.nnz()}, {sizeof(Idx)}),
        base_col
    );
    auto data = py::array(
        py::buffer_info(const_cast<T*>(A.data()), sizeof(T),
                        py::format_descriptor<T>::format(), 1,
                        {A.nnz()}, {sizeof(T)}),
        base_data
    );

    auto sp = py::module_::import("scipy.sparse");
    return sp.attr("coo_matrix")(py::make_tuple(data, row, col),
                                 py::make_tuple(A.rows(), A.cols()));
}
} // namespace hpdexc::tensor_py
#endif
