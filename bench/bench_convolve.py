# -*- coding: utf-8 -*-
"""B1/B2: производительность (λ, μ)-свёртки.

B1 воспроизводит таблицу 1 статьи Симакова: (1,1)-свёртка для 2D и 3D матриц.
B2 измеряет масштабирование по рангу при фиксированном числе ячеек.

    python bench/bench_convolve.py
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from importlib import import_module

import numpy as np

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb.core import build_einsum, convolve
from amdb.core import convolve as _convolve_module_marker  # noqa: F401
from amdb.exec.engine import blas_info, warn_if_reference_blas

_CONVOLVE = import_module("amdb.core.convolve")

#: Опубликованные в статье времена (нс/операция), AMD Ryzen 7 PRO 3700.
ARTICLE_NS = {
    "2D 10×10": {"Go посл.": 1_686, "Go парал.": 8_281, "NumPy": 4_465},
    "2D 100×100": {"Go посл.": 139_068, "Go парал.": 55_040, "NumPy": 18_323},
    "3D 10×10×10": {"Go посл.": 553_848, "Go парал.": 127_343, "NumPy": 20_646},
    "3D 100×100×100": {"Go посл.": 5_357_979_300, "Go парал.": 644_387_150,
                       "NumPy": 65_128_484},
}


@contextmanager
def einsum_only():
    """Временно отключает путь пакетного gemm, оставляя чистый einsum.

    Нужно ровно для одного: сверка с опубликованным замером должна вестись
    тем же способом, каким она получена в статье, иначе сравнивались бы разные
    операции. Ускорение от gemm показывается отдельным столбцом.
    """
    original = _CONVOLVE._batched_matmul
    _CONVOLVE._batched_matmul = lambda *args, **kwargs: None
    try:
        yield
    finally:
        _CONVOLVE._batched_matmul = original


def timeit(fn, repeat: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_b1(repeat: int) -> None:
    print("B1. Воспроизведение таблицы 1 статьи — (1,1)-свёрнутое произведение\n")
    cases = [
        ("2D 10×10", (10, 10), 0, 1),
        ("2D 100×100", (100, 100), 0, 1),
        ("3D 10×10×10", (10, 10, 10), 1, 1),
        ("3D 100×100×100", (100, 100, 100), 1, 1),
    ]
    print(f"{'размер':<18}{'einsum, нс':>15}{'gemm, нс':>15}"
          f"{'статья NumPy':>15}{'einsum/gemm':>13}{'gemm к Go':>12}")
    print("-" * 92)
    for name, shape, lam, mu in cases:
        rng = np.random.default_rng(0)
        A, B = rng.random(shape), rng.random(shape)
        spec = build_einsum(len(shape), len(shape), lam, mu)
        with einsum_only():
            ns_einsum = timeit(lambda: convolve(A, B, lam, mu), repeat) * 1e9
        ns_gemm = timeit(lambda: convolve(A, B, lam, mu), repeat) * 1e9
        ref = ARTICLE_NS[name]
        print(f"{name:<18}{ns_einsum:>15,.0f}{ns_gemm:>15,.0f}{ref['NumPy']:>15,}"
              f"{ns_einsum / ns_gemm:>12.1f}×{ref['Go посл.'] / ns_gemm:>11.1f}×"
              f"   {spec}")
    print("\nСтолбец einsum — тот же способ вычисления, что в статье: сверка "
          "должна вестись сопоставимым путём. Столбец gemm — путь настоящей "
          "работы, разлагающий λ-произведение на сечения. При λ = 0 "
          "(двумерные строки) пути совпадают, и времена равны.")


def bench_b2(repeat: int) -> None:
    print("\n\nB2. Масштабирование по рангу при ~10⁶ ячеек на операнд\n")
    print(f"{'ранг':<8}{'форма':<26}{'ячеек':>12}{'время, мс':>14}{'спецификация':>22}")
    print("-" * 82)
    for rank in range(2, 8):
        side = max(2, int(round(10 ** (6 / rank))))
        shape = (side,) * rank
        cells = side ** rank
        if cells > 4_000_000:
            side -= 1
            shape = (side,) * rank
            cells = side ** rank
        rng = np.random.default_rng(rank)
        A, B = rng.random(shape), rng.random(shape)
        lam, mu = rank - 2, 1
        spec = build_einsum(rank, rank, lam, mu)
        ms = timeit(lambda: convolve(A, B, lam, mu), repeat) * 1e3
        print(f"{rank:<8}{str(shape):<26}{cells:>12,}{ms:>14.2f}   {spec}")
    print("\nПри постоянном числе ячеек время от ранга почти не зависит: начиная "
          "с ранга 3 появляется скоттов индекс, свёртка идёт пакетным gemm, и "
          "накладные расходы планировщика einsum перестают доминировать.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--only", choices=["b1", "b2"], help="запустить один бенчмарк")
    a = p.parse_args()

    info = blas_info()
    print(f"NumPy {info['numpy']}, BLAS {info.get('blas')} "
          f"{info.get('blas_version', '')}\n")
    warn = warn_if_reference_blas()
    if warn:
        print(f"ВНИМАНИЕ: {warn}\n")

    if a.only != "b2":
        bench_b1(a.repeat)
    if a.only != "b1":
        bench_b2(a.repeat)


if __name__ == "__main__":
    main()
