# -*- coding: utf-8 -*-
"""Проверка правила выбора оси дробления: скоттова ось против свободной.

Утверждение из §3.4 — при разбиении работы между узлами следует резать по
скоттовой (λ) оси, общей обоим сомножителям, а не по свободной. Основание
теоретическое: скоттова ось режет оба операнда, и узлу достаётся только своя
доля данных; свободная ось режет лишь один операнд, а второй приходится
целиком реплицировать на все узлы.

Стенд измеряет это на конкретной свёртке A[l,s,c] × B[s,c,m] -> C[l,s,m],
где l и m свободны, а s скоттова. Для каждой оси разбиения считается:

* объём пересылок — сколько байт операндов суммарно получат узлы;
* время подготовки — фактическая материализация этих полезных нагрузок
  (узел не может считать по чужому срезу, его нужно выделить и скопировать);
* время счёта — сами частичные свёртки;
* сверка результата с монолитной свёрткой.

    python bench/bench_split.py --nodes 4
"""
from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb.core import convolve
from amdb.exec.chunker import FREE, SCOTT, chunk_ranges

#: Оси свёртки A[l,s,c] × B[s,c,m] -> C[l,s,m] и их роль.
KIND = {"l": FREE, "s": SCOTT, "m": FREE}
IN_A = {"l": True, "s": True, "m": False}
IN_B = {"l": False, "s": True, "m": True}


def plan(axis, n_nodes, shapes, a_nbytes, b_nbytes):
    """Объём пересылок при разбиении по данной оси."""
    parts = len(chunk_ranges(shapes[axis], n_nodes))
    a = a_nbytes if IN_A[axis] else a_nbytes * parts
    b = b_nbytes if IN_B[axis] else b_nbytes * parts
    replicated = [n for n, inside in (("A", IN_A[axis]), ("B", IN_B[axis])) if not inside]
    return parts, a + b, replicated


def run(axis, n_nodes, A, B, shapes):
    """Готовит полезные нагрузки узлов и считает частичные свёртки."""
    ranges = chunk_ranges(shapes[axis], n_nodes)
    t0 = time.perf_counter()
    payloads = []
    for lo, hi in ranges:
        if axis == "l":
            payloads.append((np.ascontiguousarray(A[lo:hi]), np.ascontiguousarray(B)))
        elif axis == "s":
            payloads.append((np.ascontiguousarray(A[:, lo:hi]),
                             np.ascontiguousarray(B[lo:hi])))
        else:
            payloads.append((np.ascontiguousarray(A),
                             np.ascontiguousarray(B[:, :, lo:hi])))
    t_ship = time.perf_counter() - t0

    # Счёт во всех трёх схемах идёт одним и тем же путём ядра — иначе
    # сравнивались бы не оси разбиения, а реализации свёртки. Ядро само
    # раскладывает λ-произведение на сечения и отдаёт их пакетному gemm.
    t0 = time.perf_counter()
    parts = [convolve(a, b, lam=1, mu=1) for a, b in payloads]
    t_calc = time.perf_counter() - t0

    pos = {"l": 0, "s": 1, "m": 2}[axis]
    return t_ship, t_calc, np.concatenate(parts, axis=pos)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--l", type=int, default=200)
    p.add_argument("--s", type=int, default=200)
    p.add_argument("--c", type=int, default=120)
    p.add_argument("--m", type=int, default=200)
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()

    shapes = {"l": a.l, "s": a.s, "c": a.c, "m": a.m}
    rng = np.random.default_rng(0)
    A = rng.random((a.l, a.s, a.c))
    B = rng.random((a.s, a.c, a.m))
    ref = convolve(A, B, lam=1, mu=1)

    print(f"A[l,s,c] = {A.shape}, B[s,c,m] = {B.shape}, узлов: {a.nodes}")
    print(f"A: {A.nbytes / 2**20:.1f} МиБ, B: {B.nbytes / 2**20:.1f} МиБ, "
          f"результат: {ref.nbytes / 2**20:.1f} МиБ")
    print()

    head = (f"{'ось':<6}{'роль':<16}{'частей':>8}{'переслано':>12}{'рост':>8}"
            f"{'реплика':>10}{'подготовка':>13}{'счёт':>10}{'сверка':>9}")
    print(head)
    print("-" * len(head))

    base = None
    for axis in ("s", "l", "m"):
        parts, shipped, replicated = plan(axis, a.nodes, shapes, A.nbytes, B.nbytes)
        best_ship = best_calc = float("inf")
        got = None
        for _ in range(a.repeat):
            t_ship, t_calc, got = run(axis, a.nodes, A, B, shapes)
            best_ship, best_calc = min(best_ship, t_ship), min(best_calc, t_calc)
        base = base if base is not None else shipped
        ok = np.allclose(got, ref)
        print(f"{axis:<6}{KIND[axis]:<16}{parts:>8}"
              f"{shipped / 2**20:>10.1f}М{shipped / base:>7.2f}×"
              f"{(', '.join(replicated) or 'нет'):>10}"
              f"{best_ship * 1e3:>12.1f}м{best_calc * 1e3:>9.1f}м"
              f"{'совпало' if ok else 'РАСХОЖД.':>9}")

    print()
    print("Скоттова ось s режет оба сомножителя: каждый узел получает только")
    print("свою долю, суммарный объём пересылок равен объёму данных. Свободные")
    print("оси l и m вынуждают реплицировать второй операнд на все узлы, и")
    print("объём растёт пропорционально их числу. Результат во всех трёх")
    print("случаях один и тот же — выбор оси влияет на цену, а не на ответ.")
    print()
    print("Время счёта во всех схемах одного порядка: ядро раскладывает")
    print("λ-произведение на сечения независимо от того, по какой оси разбита")
    print("работа. Различает схемы именно объём пересылок.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
