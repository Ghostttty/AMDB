# -*- coding: utf-8 -*-
"""Движки матричных вычислений: NumPy (BLAS/OpenMP) и PyTorch (CUDA)."""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np


class Engine(Protocol):
    name: str

    def einsum(self, spec: str, *ops: np.ndarray, path: Any = None) -> np.ndarray: ...


class NumpyEngine:
    """Основной путь: einsum поверх оптимизированного BLAS."""

    name = "numpy"

    def einsum(self, spec: str, *ops: np.ndarray, path: Any = None) -> np.ndarray:
        return np.einsum(spec, *ops, optimize=path if path is not None else True)

    def __repr__(self) -> str:
        return "NumpyEngine()"


class TorchEngine:
    """GPU-путь. Перенос данных учитывается при выборе движка, а не игнорируется."""

    name = "torch"

    def __init__(self, device: str = "cuda", dtype: Any = None):
        import torch  # локальный импорт: torch — необязательная зависимость

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна; используйте NumpyEngine")
        self.device = device
        self.dtype = dtype or torch.float32

    def einsum(self, spec: str, *ops: np.ndarray, path: Any = None) -> np.ndarray:
        t = [self.torch.as_tensor(np.ascontiguousarray(o), device=self.device,
                                  dtype=self.dtype) for o in ops]
        return self.torch.einsum(spec, *t).detach().cpu().numpy()

    def __repr__(self) -> str:
        return f"TorchEngine(device={self.device!r})"


def blas_info() -> dict[str, Any]:
    """Сведения о сборке BLAS — от неё производительность зависит на порядок."""
    info: dict[str, Any] = {"numpy": np.__version__}
    try:
        cfg = np.__config__.CONFIG  # numpy >= 2
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        info["blas"] = blas.get("name", "unknown")
        info["blas_version"] = blas.get("version", "unknown")
    except Exception:  # pragma: no cover
        info["blas"] = "unknown"
    return info


def warn_if_reference_blas() -> str | None:
    """Предупреждение, если NumPy собран с эталонной netlib-BLAS."""
    info = blas_info()
    name = str(info.get("blas", "")).lower()
    if name in ("blas", "netlib", "unknown"):
        return (
            f"NumPy использует BLAS '{info.get('blas')}': производительность свёрток "
            "может быть на порядок ниже, чем с OpenBLAS/MKL"
        )
    return None


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pick_engine(flops: float, bytes_moved: float, prefer: str = "auto") -> Engine:
    """Выбор движка. Порог калибруется бенчмарком bench/bench_gpu.py.

    GPU включается только при достаточной арифметической интенсивности: иначе
    выигрыш съедается переносом данных через PCIe.
    """
    if prefer == "numpy":
        return NumpyEngine()
    if prefer == "torch":
        return TorchEngine()
    intensity = flops / max(bytes_moved, 1.0)
    if flops > 5e8 and intensity > 4 and gpu_available():
        try:
            return TorchEngine()
        except Exception:  # pragma: no cover
            return NumpyEngine()
    return NumpyEngine()
