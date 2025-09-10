from __future__ import annotations
import h5py
import numpy as np
import scipy.sparse as sp
from .backen import _c_kernel  # keep kernel import path available
import importlib
csc_total = importlib.import_module('hpdex.csc_total')


def inspect_h5ad_header(path: str) -> dict:
    with h5py.File(path, "r") as f:
        X = f["/X"]
        if isinstance(X, h5py.Dataset):
            n_obs, n_vars = map(int, X.shape)
            layout = "dense"
            nnz = -1
            return dict(n_obs=n_obs, n_vars=n_vars, nnz=nnz, layout=layout)
        shape = tuple(np.array(X["shape"][...], dtype=np.int64))
        n_obs, n_vars = shape
        enc = X.attrs.get("encoding-type", "")
        if isinstance(enc, bytes):
            enc = enc.decode()
        layout = "csc" if enc == "csc_matrix" else "csr"
        nnz = int(X["data"].shape[0])
    return dict(n_obs=int(n_obs), n_vars=int(n_vars), nnz=nnz, layout=layout)


def h5ad_reader_csc(path: str, return_scipy: bool = True):
    h = inspect_h5ad_header(path)
    if h["layout"] == "dense":
        raise RuntimeError(
            f"检测到 /X 为稠密数据集（shape={h['n_obs']}x{h['n_vars']}）。当前最小实现仅支持稀疏编码的 /X（/X/indptr, /X/indices, /X/data）。"
        )
    blobs = csc_total.read_csc_total(
        path,
        h["n_obs"], h["n_vars"], h["nnz"],
        8,  # out_data_itemsize
        8,  # out_index_itemsize
        0 if h["layout"] == "csr" else 1,
    )
    if not return_scipy:
        return blobs
    indptr = np.asarray(blobs["indptr"], dtype=np.int64, order="C")
    indices = np.asarray(blobs["indices"], dtype=np.int64, order="C")
    data = np.asarray(blobs["data"], dtype=np.float64, order="C")
    n_obs, n_vars = blobs["shape"]
    return sp.csc_matrix((data, indices, indptr), shape=(n_obs, n_vars))


