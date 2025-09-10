from typing import Any, List
import importlib

__all__: List[str] = ["adaptive_sort", "mannwhitney"]

def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = importlib.import_module(".kernel", __package__)
        return getattr(mod, name)
    raise AttributeError(name)

def __dir__() -> list[str]:
    return list(__all__)