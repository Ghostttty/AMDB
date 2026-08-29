# -*- coding: utf-8 -*-
"""Движки матричных вычислений: NumPy (BLAS/OpenMP) и PyTorch (CUDA).

Движок отвечает за одно: выполнить einsum-спецификацию над набором массивов.
Всё остальное — выбор порядка свёрток, представление, разреженные ядра —
решается выше, в оптимизаторе и исполнителе.
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from ..core.convolve import batched_from_spec


class Engine(Protocol):
    name: str

    def einsum(self, spec: str, *ops: np.ndarray, path: Any = None) -> np.ndarray: ...


class NumpyEngine:
    """Основной путь: einsum поверх оптимизированного BLAS.

    Парные (λ, μ)-произведения при λ >= 1 и μ >= 1 исполняются разложением на
    сечения с пакетным матричным умножением (§4.6 статьи): einsum не всегда
    находит этот путь сам, и зависит это от версии библиотеки. Если
    спецификация к произведению не сводится, вызывается обычный einsum.
    """

    name = "numpy"

    def __init__(self, batched: bool = True):
        self.batched = batched

    def einsum(self, spec: str, *ops: np.ndarray, path: Any = None) -> np.ndarray:
        if self.batched and len(ops) == 2:
            out = batched_from_spec(spec, np.asarray(ops[0]), np.asarray(ops[1]))
            if out is not None:
                return out
        return np.einsum(spec, *ops, optimize=path if path is not None else True)

    def __repr__(self) -> str:
        return "NumpyEngine()"


class TorchEngine:
    """GPU-путь.

    Две особенности, существенные для замеров.

    **Точность.** По умолчанию считает во float64, как и остальная система:
    в §4.8 показано, что одинарной точности для денежных мер недостаточно.
    На потребительских картах двойная точность выполняется на малой доле
    скорости одинарной, поэтому переход на float32 возможен, но он —
    осознанное решение, а не умолчание.

    **Резидентные операнды.** Перенос через PCIe сопоставим по времени с самой
    свёрткой на кубах в единицы мегабайт, поэтому единственный режим, в котором
    ускоритель может дать выигрыш, — когда куб загружен на карту один раз и
    остаётся там между запросами. Для этого служит :meth:`upload`: полученный
    дескриптор можно передавать в :meth:`einsum` вместо массива.
    """

    name = "torch"

    def __init__(self, device: str = "cuda", dtype: Any = None,
                 resident: bool = False):
        import torch  # локальный импорт: torch — необязательная зависимость

        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна; используйте NumpyEngine")
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float64
        self.resident = resident
        # Ключ — идентичность массива; вместе с тензором хранится ссылка на сам
        # массив, иначе после сборки мусора идентификатор мог бы достаться
        # другому объекту и кэш вернул бы чужие данные.
        self._cache: dict[int, tuple[np.ndarray, Any]] = {}

    # -- резидентные операнды ------------------------------------------------
    def upload(self, array: np.ndarray):
        """Переносит массив на устройство и возвращает дескриптор."""
        return self.torch.as_tensor(np.ascontiguousarray(array),
                                    device=self.device, dtype=self.dtype)

    def download(self, tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def clear_resident(self) -> None:
        """Освобождает память устройства от закреплённых операндов."""
        self._cache.clear()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def resident_bytes(self) -> int:
        return sum(t.element_size() * t.nelement() for _, t in self._cache.values())

    def _as_tensor(self, operand):
        if not isinstance(operand, np.ndarray):
            return (operand.to(dtype=self.dtype)
                    if operand.dtype != self.dtype else operand)
        if not self.resident:
            return self.upload(operand)
        key = id(operand)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is operand:
            return hit[1]
        tensor = self.upload(operand)
        self._cache[key] = (operand, tensor)
        return tensor

    # -- вычисление ------------------------------------------------------------
    def einsum_device(self, spec: str, *ops, path: Any = None):
        """Свёртка без выгрузки результата: он остаётся на устройстве."""
        return self.torch.einsum(spec, *[self._as_tensor(o) for o in ops])

    def einsum(self, spec: str, *ops, path: Any = None) -> np.ndarray:
        return self.download(self.einsum_device(spec, *ops, path=path))

    def synchronize(self) -> None:
        """Ждёт завершения работы устройства — обязательно перед замером времени."""
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def __repr__(self) -> str:
        return f"TorchEngine(device={self.device!r}, dtype={self.dtype})"


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


_GPU_AVAILABLE: bool | None = None


def gpu_available() -> bool:
    """Есть ли пригодный ускоритель. Результат запоминается: импорт torch дорог."""
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is None:
        try:
            import torch

            _GPU_AVAILABLE = bool(torch.cuda.is_available())
        except Exception:
            _GPU_AVAILABLE = False
    return _GPU_AVAILABLE


def gpu_info() -> dict[str, Any]:
    """Описание ускорителя — для протокола замера."""
    if not gpu_available():
        return {"available": False}
    import torch

    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    return {
        "available": True,
        "name": props.name,
        "memory_gib": round(props.total_memory / 2**30, 1),
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


#: Ниже этого числа операций ускоритель заведомо не окупает переноса.
GPU_MIN_FLOPS = 5e8
#: Отношение операций к перемещённым байтам, ниже которого решает не счёт, а шина.
GPU_MIN_INTENSITY = 4.0


def spec_cost(spec: str, *ops: np.ndarray) -> tuple[float, float]:
    """Оценка (число операций, перемещено байт) для einsum-спецификации.

    Число операций — размер полного индексного пространства спецификации;
    перемещённые байты — суммарный объём операндов и результата. Оценка груба,
    но её достаточно для решения «стоит ли переносить данные на устройство»,
    где разница между вариантами составляет порядки.
    """
    if "->" not in spec:
        return 0.0, 0.0
    lhs, out = spec.split("->")
    sizes: dict[str, int] = {}
    for sub, arr in zip(lhs.split(","), ops):
        for letter, length in zip(sub, np.shape(arr)):
            sizes[letter] = length
    flops = 1.0
    for length in sizes.values():
        flops *= length
    itemsize = max((np.asarray(o).itemsize for o in ops), default=8)
    moved = float(sum(np.size(o) for o in ops))
    out_cells = 1.0
    for letter in out:
        out_cells *= sizes.get(letter, 1)
    return flops, (moved + out_cells) * itemsize


def pick_engine(flops: float, bytes_moved: float, prefer: str = "auto") -> Engine:
    """Выбор движка.

    GPU включается только при достаточной арифметической интенсивности: иначе
    выигрыш съедается переносом данных через PCIe. Пороги калибруются стендом
    bench/bench_gpu.py.
    """
    if prefer == "numpy":
        return NumpyEngine()
    if prefer == "torch":
        return TorchEngine()
    intensity = flops / max(bytes_moved, 1.0)
    if flops > GPU_MIN_FLOPS and intensity > GPU_MIN_INTENSITY and gpu_available():
        try:
            return TorchEngine()
        except Exception:  # pragma: no cover
            return NumpyEngine()
    return NumpyEngine()
