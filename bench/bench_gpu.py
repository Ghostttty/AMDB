# -*- coding: utf-8 -*-
"""B6: CPU против GPU с учётом переноса данных.

Определяет порог, при котором имеет смысл включать TorchEngine (см.
amdb.exec.engine.pick_engine). Без CUDA бенчмарк корректно завершается.

    python bench/bench_gpu.py
"""
from __future__ import annotations

import argparse
import time

import numpy as np

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb.core import build_einsum
from amdb.exec.engine import NumpyEngine, TorchEngine, gpu_available


def timeit(fn, repeat: int = 3) -> float:
    fn()
    return min(_once(fn) for _ in range(repeat))


def _once(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()

    if not gpu_available():
        print("CUDA недоступна (нет torch или нет GPU) — бенчмарк B6 пропущен.")
        print("Порог переключения в pick_engine остаётся значением по умолчанию.")
        return

    cpu, gpu = NumpyEngine(), TorchEngine()
    spec = build_einsum(3, 3, 1, 1)
    print(f"{'сторона':<12}{'ячеек':>14}{'CPU, мс':>12}{'GPU, мс':>12}{'ускорение':>12}")
    print("-" * 62)
    for side in (50, 100, 200, 300):
        rng = np.random.default_rng(0)
        A = rng.random((side, side, side)).astype(np.float32)
        B = rng.random((side, side, side)).astype(np.float32)
        t_cpu = timeit(lambda: cpu.einsum(spec, A, B), a.repeat)
        t_gpu = timeit(lambda: gpu.einsum(spec, A, B), a.repeat)
        print(f"{side:<12}{side ** 3:>14,}{t_cpu * 1e3:>12.2f}{t_gpu * 1e3:>12.2f}"
              f"{t_cpu / t_gpu:>11.2f}×")
    print("\nGPU-время включает перенос через PCIe в обе стороны — именно поэтому")
    print("pick_engine переключается на GPU только при высокой арифметической")
    print("интенсивности, а не по одному лишь объёму данных.")


if __name__ == "__main__":
    main()
