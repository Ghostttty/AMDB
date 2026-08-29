# -*- coding: utf-8 -*-
"""B6: CPU против GPU на ядре — с учётом переноса и без него.

Отвечает на два разных вопроса, которые легко перепутать.

**С переносом.** Операнды каждый раз загружаются на карту и результат
скачивается обратно. Так выглядит разовый запрос по холодным данным. Куб на
10^6 ячеек во float64 — это 8 МиБ; перенос через шину занимает порядка
миллисекунды, а сама свёртка на таких размерах — единицы миллисекунд, поэтому
здесь ускоритель окупается далеко не всегда.

**Резидентный режим.** Куб загружен на карту один раз и остаётся там между
запросами; скачивается только результат, который на порядки меньше. Это
рабочий режим витрины, и именно он показывает, есть ли выигрыш по существу.

Точность измеряется отдельно: система считает во float64, но у потребительских
карт двойная точность выполняется на малой доле скорости одинарной, и разница
между режимами может оказаться больше разницы между CPU и GPU.

Результаты **сверяются с CPU**: расхождение считается провалом замера.

    python bench/bench_gpu.py
    python bench/bench_gpu.py --dtype float32 --sides 100,200,300,400

Без CUDA стенд корректно завершается, сообщив, чего не хватает.
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
from amdb.exec.engine import (
    GPU_MIN_FLOPS,
    GPU_MIN_INTENSITY,
    NumpyEngine,
    TorchEngine,
    gpu_available,
    gpu_info,
    spec_cost,
)


def timeit(fn, repeat: int, after=None) -> float:
    fn()
    if after is not None:
        after()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if after is not None:
            after()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--sides", default="50,100,200,300",
                   help="стороны кубов через запятую")
    p.add_argument("--dtype", choices=["float64", "float32", "both"], default="both")
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="допуск сверки для float32")
    a = p.parse_args()

    if not gpu_available():
        print("CUDA недоступна — стенд B6 пропущен.")
        print("Нужны: карта NVIDIA, драйвер и torch со сборкой под CUDA.")
        print("Порог переключения в pick_engine остаётся значением по умолчанию:")
        print(f"  операций > {GPU_MIN_FLOPS:.0e}, интенсивность > {GPU_MIN_INTENSITY}")
        return 0

    info = gpu_info()
    print(f"GPU: {info['name']}, {info['memory_gib']} ГиБ, вычислительная "
          f"способность {info['capability']}; torch {info['torch']}, "
          f"CUDA {info['cuda']}")

    import torch

    dtypes = ({"float64": [("float64", np.float64, torch.float64)],
               "float32": [("float32", np.float32, torch.float32)]}
              .get(a.dtype, [("float64", np.float64, torch.float64),
                             ("float32", np.float32, torch.float32)]))
    sides = [int(x) for x in a.sides.split(",")]
    spec = build_einsum(3, 3, 1, 1)
    cpu = NumpyEngine()
    failures = 0

    for name, np_dtype, torch_dtype in dtypes:
        gpu = TorchEngine(dtype=torch_dtype)
        print(f"\nТочность {name}; свёртка {spec}")
        head = (f"{'сторона':<9}{'ячеек':>13}{'МиБ':>8}{'CPU, мс':>11}"
                f"{'GPU+перенос':>13}{'ускор.':>9}{'GPU резид.':>12}{'ускор.':>9}"
                f"{'сверка':>9}")
        print(head)
        print("-" * len(head))
        for side in sides:
            rng = np.random.default_rng(0)
            A = rng.random((side, side, side)).astype(np_dtype)
            B = rng.random((side, side, side)).astype(np_dtype)

            ref = cpu.einsum(spec, A, B)
            t_cpu = timeit(lambda: cpu.einsum(spec, A, B), a.repeat)

            t_move = timeit(lambda: gpu.einsum(spec, A, B), a.repeat)

            dev_a, dev_b = gpu.upload(A), gpu.upload(B)
            got = gpu.download(gpu.einsum_device(spec, dev_a, dev_b))
            t_res = timeit(lambda: gpu.download(gpu.einsum_device(spec, dev_a, dev_b)),
                           a.repeat, after=gpu.synchronize)

            tol = 1e-12 if name == "float64" else a.tolerance
            ok = np.allclose(got, ref, rtol=tol, atol=tol * max(abs(ref).max(), 1.0))
            failures += not ok
            del dev_a, dev_b
            torch.cuda.empty_cache()

            print(f"{side:<9}{side ** 3:>13,}{A.nbytes / 2**20:>8.0f}"
                  f"{t_cpu * 1e3:>11.2f}{t_move * 1e3:>13.2f}{t_cpu / t_move:>8.2f}×"
                  f"{t_res * 1e3:>12.2f}{t_cpu / t_res:>8.2f}×"
                  f"{'совпало' if ok else 'РАСХОЖД.':>9}".replace(",", " "))

    print("\nЧто с этим делать. Столбец «GPU+перенос» отвечает на вопрос о разовом")
    print("запросе, «GPU резид.» — о витрине, где куб живёт на карте. Порог в")
    print("pick_engine следует калибровать по первому столбцу: он определяет, с")
    print("какого размера перенос окупается. Сейчас порог таков:")
    flops, moved = spec_cost(spec, np.zeros((100, 100, 100)), np.zeros((100, 100, 100)))
    print(f"  операций > {GPU_MIN_FLOPS:.0e}, интенсивность > {GPU_MIN_INTENSITY}; "
          f"для куба 100^3 получается {flops:.1e} и {flops / moved:.1f}")

    if failures:
        print(f"\nВНИМАНИЕ: расхождение с CPU на {failures} замерах — "
              "результаты недействительны.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
