from typing import Any, List
import importlib

__all__: List[str] = ["ping", "mwu_v1_stats_csc", "sparse_wilcoxon_de", "wilcoxon_fdr_csc"]

def __getattr__(name: str) -> Any:
    if name in __all__:
        if name in ("ping", "mwu_v1_stats_csc"):
            mod = importlib.import_module(".backen", __name__)
            return getattr(mod, name)
        if name == "sparse_wilcoxon_de":
            mod = importlib.import_module(".sparse_de", __name__)
            return getattr(mod, name)
        if name == "wilcoxon_fdr_csc":
            mod = importlib.import_module(".simple", __name__)
            return getattr(mod, name)
    raise AttributeError(name)

def __dir__() -> list[str]:
    return list(__all__)


