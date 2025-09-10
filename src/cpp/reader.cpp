#include "reader.hpp"

#include <hdf5.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace hpdexc {

template <class T>
static std::vector<T> h5_read_1d(hid_t file, const char* path, hid_t memtype) {
    if (H5Lexists(file, path, H5P_DEFAULT) <= 0) throw std::runtime_error(std::string("missing: ")+path);
    hid_t dset = H5Dopen(file, path, H5P_DEFAULT);
    if (dset < 0) throw std::runtime_error(std::string("open failed: ")+path);
    hid_t sp = H5Dget_space(dset);
    hssize_t n = H5Sget_simple_extent_npoints(sp);
    std::vector<T> out((size_t)n);
    if (H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) {
        H5Sclose(sp); H5Dclose(dset);
        throw std::runtime_error(std::string("read failed: ")+path);
    }
    H5Sclose(sp); H5Dclose(dset);
    return out;
}

static void csr_to_csc(std::size_t rows, std::size_t cols,
                       const std::vector<int64_t>& rp,
                       const std::vector<int64_t>& ci,
                       const std::vector<double>& xv,
                       std::vector<int64_t>& cp,
                       std::vector<int64_t>& ri,
                       std::vector<double>& cv) {
    const int64_t nnz = (int64_t)xv.size();
    cp.assign(cols + 1, 0);
    ri.resize((size_t)nnz);
    cv.resize((size_t)nnz);

    for (int64_t i = 0; i < (int64_t)rows; ++i)
        for (int64_t k = rp[i]; k < rp[i+1]; ++k)
            cp[(size_t)ci[(size_t)k] + 1]++;
    for (size_t j = 0; j < cols; ++j) cp[j+1] += cp[j];
    std::vector<int64_t> nxt = cp;
    for (int64_t i = 0; i < (int64_t)rows; ++i) {
        for (int64_t k = rp[i]; k < rp[i+1]; ++k) {
            int64_t j = ci[(size_t)k];
            int64_t dst = nxt[(size_t)j]++;
            ri[(size_t)dst] = i;
            cv[(size_t)dst] = xv[(size_t)k];
        }
    }
}

CscOwned read_total_csc(const std::string& path, int layout_code) {
    CscOwned out;
    hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) throw std::runtime_error("cannot open: " + path);

    // 读取 /X/shape
    auto shape = h5_read_1d<long long>(file, "/X/shape", H5T_NATIVE_LLONG);
    if (shape.size() != 2) { H5Fclose(file); throw std::runtime_error("bad /X/shape"); }
    out.rows = (std::size_t)shape[0];
    out.cols = (std::size_t)shape[1];

    // 读取 indptr/indices（提升为 int64）
    auto indptr64  = h5_read_1d<long long>(file, "/X/indptr",  H5T_NATIVE_LLONG);
    auto indices64 = h5_read_1d<long long>(file, "/X/indices", H5T_NATIVE_LLONG);

    // 读取 data 为 double
    hid_t dset = H5Dopen(file, "/X/data", H5P_DEFAULT);
    if (dset < 0) { H5Fclose(file); throw std::runtime_error("open /X/data failed"); }
    hid_t sp  = H5Dget_space(dset);
    hssize_t n = H5Sget_simple_extent_npoints(sp);
    hid_t dt  = H5Dget_type(dset);
    size_t sz = H5Tget_size(dt);
    std::vector<double> data64((size_t)n);
    if (H5Tget_class(dt) != H5T_FLOAT) {
        H5Tclose(dt); H5Sclose(sp); H5Dclose(dset); H5Fclose(file);
        throw std::runtime_error("/X/data not float");
    }
    if (sz == 4) {
        std::vector<float> tmp((size_t)n);
        if (H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, tmp.data()) < 0) {
            H5Tclose(dt); H5Sclose(sp); H5Dclose(dset); H5Fclose(file);
            throw std::runtime_error("read /X/data f32 failed");
        }
        for (size_t i = 0; i < (size_t)n; ++i) data64[i] = (double)tmp[i];
    } else if (sz == 8) {
        if (H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data64.data()) < 0) {
            H5Tclose(dt); H5Sclose(sp); H5Dclose(dset); H5Fclose(file);
            throw std::runtime_error("read /X/data f64 failed");
        }
    } else {
        H5Tclose(dt); H5Sclose(sp); H5Dclose(dset); H5Fclose(file);
        throw std::runtime_error("unsupported float width");
    }
    H5Tclose(dt); H5Sclose(sp); H5Dclose(dset);

    out.nnz = data64.size();

    if (layout_code == 1) {
        // 已是 CSC
        if (indptr64.size() != out.cols + 1) { H5Fclose(file); throw std::runtime_error("CSC indptr size mismatch"); }
        if (indices64.size() != out.nnz) { H5Fclose(file); throw std::runtime_error("CSC indices size mismatch"); }
        out.indptr64.assign(indptr64.begin(), indptr64.end());
        out.indices64.assign(indices64.begin(), indices64.end());
        out.data.assign(data64.begin(), data64.end());
    } else {
        // CSR -> CSC
        if (indptr64.size() != out.rows + 1) { H5Fclose(file); throw std::runtime_error("CSR indptr size mismatch"); }
        if (indices64.size() != out.nnz) { H5Fclose(file); throw std::runtime_error("CSR indices size mismatch"); }
        csr_to_csc(out.rows, out.cols, indptr64, indices64, data64, out.indptr64, out.indices64, out.data);
    }

    H5Fclose(file);
    return out;
}

} // namespace hpdexc


