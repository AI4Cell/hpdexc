from hpdex import parallel_differential_expression
import pandas as pd
import anndata as ad
import time

print("Loading data...")
adata = ad.read_h5ad("/Volumes/Wzq/Datasets/scperturb/ReplogleWeissman2022_rpe1.h5ad")

print("Differential expression analysis...")
start_time = time.time()
df = parallel_differential_expression(
    adata,
    groupby_key="gene",
    reference="non-targeting",
    threads=8,
    show_progress=True
)
end_time = time.time()
print("🎉 Differential expression analysis completed")
print(f"Time taken: {end_time - start_time} s")

print(df.head())
df.to_csv("ReplogleWeissman2022_rpe1.csv", index=False)