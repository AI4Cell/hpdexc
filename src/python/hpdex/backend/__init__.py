from typing import Any, List
import importlib

__all__: List[str] = ["kernel"]

def __getattr__(name: str) -> Any:
    if name in __all__:
        if name in ("kernel"):
            mod = importlib.import_module(".kernel", __name__)
            return getattr(mod, name)
    raise AttributeError(name)

def __dir__() -> list[str]:
    return list(__all__)


# from .kernel import mannwhitney  # 移除直接导入以避免类型重复定义