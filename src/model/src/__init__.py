# src/model/src/__init__.py
from typing import Callable, Dict
import torch.nn as nn

# ---------- registry ----------
_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

def register_model(name: str):
    def _decorator(builder: Callable[..., nn.Module]):
        _MODEL_REGISTRY[name] = builder
        return builder
    return _decorator

def get_model(name: str, **kwargs) -> nn.Module:
    if name not in _MODEL_REGISTRY:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "(none registered)"
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name](**kwargs)

# ---------- auto-import all submodules so they register ----------
import importlib, pkgutil, pathlib
_pkg_path = pathlib.Path(__file__).parent
for m in [m.name for m in pkgutil.iter_modules([str(_pkg_path)])]:
    if m == "__init__":
        continue
    importlib.import_module(f"{__name__}.{m}")
