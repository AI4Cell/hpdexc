from hpdex import parallel_differential_expression
import pandas as pd
import scanpy as sc

adata = sc.read_h5ad("/Volumes/Wzq/Datasets/scperturb/ReplogleWeissman2022_rpe1.h5ad")

df = parallel_differential_expression(
    adata,
    groupby_key="gene",
    reference="non-targeting",
    threads=8
)

print(df.head())
df.to_csv("ReplogleWeissman2022_rpe1.csv", index=False)