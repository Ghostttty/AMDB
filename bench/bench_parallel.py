# -*- coding: utf-8 -*-
"""B13: параллельное исполнение по скоттовым сечениям против однопоточного.

Реализуется первое разбиение из работы Мунермана и Мунермана (2022): при λ > 0
результат составлен из сечений ориентации скоттовых индексов, и сечения
независимы — обмена между процессами не требуется.

Стенд отвечает на один вопрос: **когда разбиение по сечениям выгоднее, чем
отдать ту же работу одному вызову.** Ответ зависит от того, с чем сравнивать,
и потому замер ведётся в двух режимах:

* против **многопоточного BLAS** — то, как система работает по умолчанию:
  einsum внутри уже занимает все ядра;
* против **однопоточного BLAS** — то, как она работала бы на сборке NumPy без
  многопоточности либо при явном ограничении потоков.

Разница между этими двумя столбцами и есть содержание замера: разбиение по
сечениям **заменяет** внутриблочный параллелизм, а не добавляется к нему.

    python bench/bench_parallel.py
    python bench/bench_parallel.py --workers 8 --repeat 5

Замер требует запуска дочерних процессов, поэтому стенд обязан вызываться как
самостоятельная программа (``if __name__ == "__main__"``), иначе на Windows
порождение процессов уйдёт в рекурсию.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb.exec.engine import NumpyEngine
from amdb.exec.parallel import ParallelEngine

#: Формы для развёртки: (спецификация, форма A, форма B).
#: Первый индекс — скоттов, по нему и идёт разбиение.
CASES = [
    ("slc,scm->slm", (32, 300, 200), (32, 200, 300)),
    ("slc,scm->slm", (128, 300, 200), (128, 200, 300)),
    ("slc,scm->slm", (512, 150, 100), (512, 100, 150)),
    ("slc,scm->slm", (2000, 60, 40), (2000, 40, 60)),
    ("slc,scm->slm", (8000, 25, 20), (8000, 20, 25)),
    ("slc,sc->sl", (256, 400, 300), (256, 300)),
]

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def timeit(fn, repeat: int) -> float:
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def measure(workers_list, repeat: int) -> list[tuple]:
    """Прогон развёртки в текущем окружении (число потоков BLAS задано снаружи)."""
    cpu = NumpyEngine()
    rows = []
    for spec, shape_a, shape_b in CASES:
        rng = np.random.default_rng(0)
        A, B = rng.random(shape_a), rng.random(shape_b)
        reference = cpu.einsum(spec, A, B)
        t_one = timeit(lambda: cpu.einsum(spec, A, B), repeat)

        best_time, best_workers, verified = float("inf"), 0, True
        for workers in workers_list:
            engine = ParallelEngine(workers=workers, min_cells=0)
            try:
                verified &= bool(np.allclose(engine.einsum(spec, A, B), reference))
                elapsed = timeit(lambda: engine.einsum(spec, A, B), repeat)
            finally:
                engine.close()
            if elapsed < best_time:
                best_time, best_workers = elapsed, workers
        megabytes = (A.nbytes + B.nbytes) / 2**20
        rows.append((spec, shape_a, megabytes, t_one, best_time, best_workers,
                     verified))
    return rows


def report(title: str, rows) -> None:
    print(f"\n{title}")
    head = (f"{'сечений × форма':<24}{'МиБ':>7}{'один вызов':>13}"
            f"{'по сечениям':>14}{'ускорение':>12}{'проц.':>7}{'сверка':>9}")
    print(head)
    print("-" * len(head))
    for spec, shape_a, mib, t_one, t_par, workers, ok in rows:
        label = f"{shape_a[0]} × {shape_a[1]}×{shape_a[2]}"
        print(f"{label:<24}{mib:>7.0f}{t_one:>12.1f}м{t_par:>13.1f}м"
              f"{t_one / t_par:>11.2f}×{workers:>7}"
              f"{'совпало' if ok else 'РАСХОЖД.':>9}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=0,
                   help="фиксированное число процессов; 0 — перебрать 4, 8, 16")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--single-blas", action="store_true",
                   help="служебный режим: прогон уже в окружении с одним потоком BLAS")
    a = p.parse_args()

    workers_list = [a.workers] if a.workers else [4, 8, 16]
    rows = measure(workers_list, a.repeat)

    if a.single_blas:                       # дочерний прогон, печатает только таблицу
        report("BLAS в один поток", rows)
        return 0

    print(f"ядер: {os.cpu_count()}; процессов в переборе: {workers_list}")
    report("BLAS многопоточный — так система работает по умолчанию", rows)

    # Второй режим требует, чтобы ограничение потоков было выставлено до импорта
    # NumPy, поэтому он выполняется отдельным процессом.
    env = dict(os.environ)
    for var in _THREAD_VARS:
        env[var] = "1"
    child = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--single-blas",
         "--repeat", str(a.repeat)] + (["--workers", str(a.workers)] if a.workers else []),
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(child.stdout.rstrip())
    if child.returncode != 0:
        print(child.stderr[-800:])
        return child.returncode

    print("""
Как это читать. Разбиение по сечениям и многопоточность BLAS занимают одни и те
же ядра, поэтому они не складываются, а заменяют друг друга. Верхняя таблица
сравнивает разбиение с уже распараллеленным BLAS, нижняя — с однопоточным.
Практический вывод следует из их сопоставления, а не из каждой по отдельности.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
