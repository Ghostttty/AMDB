# -*- coding: utf-8 -*-
"""Самопроверка окружения: что готово к работе и к замерам, а чего не хватает.

Пишется для одного случая: арендована машина с ускорителем, время идёт, и надо
за несколько секунд понять, можно ли запускать стенды, — а не выяснять это по
их сообщениям об ошибках.

Проверяется четыре вещи, и каждая проверяется **замером**, а не наличием
пакета: считает ли BLAS в несколько потоков; доступен ли ускоритель; с какой
скоростью он выполняет двойную точность, которой требует модель; и при каком
размере куба автовыбор движка вообще включит ускоритель.

    python -m amdb doctor
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from ..exec.engine import (
    GPU_MIN_FLOPS,
    GPU_MIN_INTENSITY,
    blas_info,
    gpu_available,
    warn_if_reference_blas,
)

OK, WARN, BAD = "  ок  ", " важно", "  нет "


def _best(fn, repeat: int = 3) -> float:
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _line(status: str, title: str, detail: str = "") -> None:
    print(f"[{status}] {title}" + (f"\n         {detail}" if detail else ""))


def _check_python() -> None:
    import platform
    import sys

    version = ".".join(str(x) for x in sys.version_info[:3])
    cores = os.cpu_count() or 0
    _line(OK if sys.version_info >= (3, 10) else BAD,
          f"Python {version}, ядер {cores}, {platform.system()}")


def _check_numpy() -> None:
    info = blas_info()
    name = info.get("blas", "неизвестно")
    detail = f"BLAS: {name} {info.get('blas_version', '')}".strip()

    # Достигнутая производительность важнее названия сборки: эталонная netlib
    # может называться как угодно, а считать на порядок медленнее.
    size = 1200
    rng = np.random.default_rng(0)
    a, b = rng.random((size, size)), rng.random((size, size))
    seconds = _best(lambda: a @ b)
    gflops = 2 * size ** 3 / seconds / 1e9

    warn = warn_if_reference_blas()
    if warn:
        _line(BAD, f"NumPy {info['numpy']} — {gflops:.0f} Гфлопс на умножении {size}²",
              warn)
    elif gflops < 20:
        _line(WARN, f"NumPy {info['numpy']} — {gflops:.0f} Гфлопс на умножении {size}²",
              f"{detail}; для многопоточной сборки это мало — проверьте, не задан "
              "ли OMP_NUM_THREADS")
    else:
        _line(OK, f"NumPy {info['numpy']} — {gflops:.0f} Гфлопс на умножении {size}²",
              detail)


def _check_optional() -> None:
    from importlib.util import find_spec

    wanted = [("pandas", "загрузка таблиц и выдача результата", 'pip install -e ".[pandas]"'),
              ("duckdb", "стенды сравнения", 'pip install -e ".[bench]"'),
              ("chdb", "столбец ClickHouse в стендах (только Linux)",
               'pip install -e ".[clickhouse]"')]
    for module, why, how in wanted:
        if find_spec(module) is not None:
            _line(OK, f"{module} — {why}")
        else:
            _line(WARN, f"{module} не установлен — {why}", how)


def _check_gpu() -> dict[str, Any]:
    from importlib.util import find_spec

    if find_spec("torch") is None:
        _line(WARN, "torch не установлен — стенды ускорителя будут пропущены",
              'pip install -e ".[gpu]"  (на Windows нужен индекс с сайта PyTorch)')
        return {}

    import torch

    if not torch.cuda.is_available():
        _line(BAD, f"torch {torch.__version__} собран без CUDA либо карта не видна",
              "проверьте nvidia-smi и сборку torch: у колеса по умолчанию под "
              "Windows поддержки CUDA нет")
        return {}

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    _line(OK, f"{props.name}, {props.total_memory / 2**30:.0f} ГиБ, "
              f"способность {props.major}.{props.minor}",
          f"torch {torch.__version__}, CUDA {torch.version.cuda}")

    # Сразу проверяем, что ядра под эту карту в колесе есть: у совсем свежих
    # карт cuda.is_available() бывает истинно, а операция падает.
    try:
        probe = torch.ones(64, 64, device="cuda", dtype=torch.float64)
        (probe @ probe).cpu()
    except Exception as exc:  # pragma: no cover — зависит от железа
        _line(BAD, "карта видна, но операция не выполняется",
              f"{type(exc).__name__}: {exc}. Обычно значит, что в сборке torch "
              "нет ядер под эту вычислительную способность")
        return {}
    return {"torch": torch}


def _check_precision(env: dict[str, Any]) -> None:
    """Измеряет отношение скоростей одинарной и двойной точности.

    Определяет, годится ли карта для расчётов, которых требует модель: у
    потребительских серий двойная точность выполняется на малой доле скорости
    одинарной, и замер во float64 на них покажет не свойства подхода, а
    свойства карты.
    """
    torch = env.get("torch")
    if torch is None:
        return
    size = 2048
    times = {}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        x = torch.randn(size, size, device="cuda", dtype=dtype)

        def run():
            torch.matmul(x, x)
            torch.cuda.synchronize()

        times[name] = _best(run)
        del x
    torch.cuda.empty_cache()

    ratio = times["float64"] / times["float32"]
    gflops64 = 2 * size ** 3 / times["float64"] / 1e9
    detail = (f"float64 {gflops64:.0f} Гфлопс, вдвое-втрое медленнее одинарной — "
              "карта пригодна для расчётов в двойной точности")
    if ratio <= 4:
        _line(OK, f"двойная точность медленнее одинарной в {ratio:.1f} раза", detail)
    elif ratio <= 16:
        _line(WARN, f"двойная точность медленнее одинарной в {ratio:.0f} раз",
              f"float64 даёт {gflops64:.0f} Гфлопс; замеры в двойной точности "
              "будут занижены — снимайте и --dtype float32 тоже")
    else:
        _line(BAD, f"двойная точность медленнее одинарной в {ratio:.0f} раз",
              f"float64 даёт {gflops64:.0f} Гфлопс: это потребительская карта, и "
              "замер во float64 покажет её ограничение, а не свойства подхода. "
              "Пользуйтесь --dtype float32 и оценивайте погрешность по §4.8")


def _check_dispatch() -> None:
    """Показывает, при каком размере куба автовыбор включит ускоритель."""
    print()
    print("Порог автовыбора движка: операций > "
          f"{GPU_MIN_FLOPS:.0e}, интенсивность > {GPU_MIN_INTENSITY}")
    for side in (100, 150, 200, 300):
        flops = float(side) ** 4                      # abc,bcd->abd при равных осях
        moved = (2 * side ** 3 + side ** 3) * 8.0
        fires = flops > GPU_MIN_FLOPS and flops / moved > GPU_MIN_INTENSITY
        print(f"    куб {side}³ × куб {side}³: {flops:.1e} операций, "
              f"интенсивность {flops / moved:.1f} -> "
              f"{'ускоритель' if fires else 'процессор'}")
    print("    На звёздной схеме с кубом 100³ ускоритель не включится: шаги плана")
    print("    дают порядка 10⁶ операций. Для замеров берите крупные измерения.")


def run() -> int:
    print("Самопроверка окружения AMDB\n")
    _check_python()
    _check_numpy()
    _check_optional()
    env = _check_gpu()
    _check_precision(env)
    _check_dispatch()

    print()
    if env:
        print("Готово к запуску: bench/bench_gpu.py и bench/bench_engines.py")
    else:
        print("Стенды ускорителя запустятся, но столбцы GPU останутся пустыми.")
        print("Остальные стенды работают: bench_duckdb.py, bench_parallel.py.")
    return 0
