# -*- coding: utf-8 -*-
"""Расширенное сравнение AMDB и DuckDB: развёртки по трём параметрам.

Одна точка замера говорит мало. Здесь измеряется, как ведёт себя преимущество
матричного подхода при изменении трёх величин, каждая из которых меняет условия
принципиально:

* **объём фактов** — при фиксированном кубе рост числа строк повышает
  заполненность, но не размер куба: свёртка не дорожает вовсе;
* **размер куба** — произведение мощностей измерений. Растёт объём вычислений и
  память; здесь ищется граница применимости;
* **перекос распределения** — реальные продажи неравномерны; перекос понижает
  заполненность и переводит куб в разреженный режим.

Каждый замер сверяется с DuckDB: расхождение делает замер недействительным.

    python bench/bench_scaling.py                    # все три развёртки
    python bench/bench_scaling.py --only rows        # одна
    python bench/bench_scaling.py --max-cells 3e7    # ограничить память
"""
from __future__ import annotations

import argparse
import gc
import time

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bench.workload import build_amdb, build_duckdb, make_workload, normalize

#: Подмножество запросов для развёрток: по одному представителю на класс.
PROBES = [
    ("свёртка 2 осей", "SELECT customer, SUM(quantity) AS v FROM sales GROUP BY customer",
     "SELECT customer, SUM(quantity) AS v FROM sales GROUP BY customer", 1),
    ("соединение+иерархия",
     "SELECT customer, month, SUM(quantity * price) AS v FROM sales "
     "JOIN product ON sales.product = product.product GROUP BY customer, month",
     "SELECT s.customer, d.month, SUM(s.quantity * p.price) AS v FROM sales s "
     "JOIN products p ON s.product = p.product JOIN dates d ON s.date = d.date "
     "GROUP BY s.customer, d.month", 2),
    ("широкая выдача",
     "SELECT customer, product, date, SUM(quantity) AS v FROM sales "
     "GROUP BY customer, product, date",
     "SELECT customer, product, date, SUM(quantity) AS v FROM sales "
     "GROUP BY customer, product, date", 3),
]


def timeit(fn, repeat: int) -> tuple[float, object]:
    out = fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def measure(rows: int, customers: int, products: int, days: int, skew: float,
            repeat: int, layout=None, seed: int = 42) -> dict | None:
    """Один замер: строит обе базы, гоняет пробы, сверяет результаты."""
    work = make_workload(rows, customers, products, days, seed, skew)

    t0 = time.perf_counter()
    kwargs = {"layout": layout} if layout else {}
    db = build_amdb(work, **kwargs)
    load = time.perf_counter() - t0
    con = build_duckdb(work)
    con.execute("SELECT COUNT(*) FROM sales").fetchall()

    stats = db.stats("sales")
    mem = sum(db.stats(n)["bytes_in_memory"] for n in db.cubes) / 2**20
    src = work.sales.memory_usage(deep=True).sum() / 2**20

    probes = {}
    for name, sql_a, sql_d, nkeys in PROBES:
        t_a, res_a = timeit(lambda: db.sql(sql_a).to_pandas(), repeat)
        t_d, res_d = timeit(lambda: con.execute(sql_d).fetchdf(), repeat)
        got = normalize(list(res_a.itertuples(index=False, name=None)), nkeys)
        ref = normalize(list(res_d.itertuples(index=False, name=None)), nkeys)
        ok = len(got) == len(ref) and all(
            g[:-1] == r[:-1] and abs(g[-1] - r[-1]) <= 1e-9 * max(abs(r[-1]), 1.0)
            for g, r in zip(got, ref))
        probes[name] = {"amdb": t_a, "duck": t_d, "ratio": t_d / t_a,
                        "ok": ok, "rows": len(got)}

    con.close()
    result = {
        "rows": rows, "cells": work.cells, "fill": stats["fill_factor"],
        "layout": db.cube("sales").layout, "load": load, "mem": mem, "src": src,
        "probes": probes,
    }
    del db, work
    gc.collect()
    return result


def table(title: str, note: str, results: list[dict], first_col: str,
          first_val) -> None:
    print(f"\n{title}")
    print(note)
    names = [n for n, *_ in PROBES]
    head = (f"{first_col:<18}{'заполн.':>9}{'куб, МиБ':>10}{'загрузка, мс':>14}"
            + "".join(f"{n:>24}" for n in names))
    print(head)
    print("-" * len(head))
    for r, val in zip(results, first_val):
        cells = "".join(
            f"{r['probes'][n]['ratio']:>22.2f}×" if r["probes"][n]["ok"]
            else f"{'РАСХОЖД.':>23}" for n in names)
        print(f"{val:<18}{r['fill']:>8.1%}{r['mem']:>10.0f}"
              f"{r['load'] * 1e3:>14.0f}{cells}")
    print("Числа — во сколько раз AMDB быстрее DuckDB (меньше 1 — медленнее).")


def sweep_rows(args) -> list[dict]:
    sizes = [100_000, 500_000, 2_000_000, 5_000_000]
    if args.heavy:
        sizes.append(10_000_000)
    out = []
    for n in sizes:
        print(f"  ... {n:,} строк".replace(",", " "), flush=True)
        out.append(measure(n, 100, 100, 100, 0.0, args.repeat))
    table("РАЗВЁРТКА 1. Объём фактов при неизменном кубе 100×100×100",
          "Куб не растёт, растёт лишь его заполненность — свёртка не дорожает,\n"
          "тогда как DuckDB обрабатывает всё больше строк.",
          out, "строк фактов", [f"{s:,}".replace(",", " ") for s in sizes])
    return out


def sweep_cells(args) -> list[dict]:
    sides = [50, 100, 150, 200]
    if args.heavy:
        sides.append(260)
    sides = [s for s in sides if s ** 3 <= args.max_cells]
    out = []
    for s in sides:
        print(f"  ... куб {s}×{s}×{s} = {s**3:,} ячеек".replace(",", " "), flush=True)
        out.append(measure(2_000_000, s, s, s, 0.0, args.repeat))
    table("РАЗВЁРТКА 2. Размер куба при неизменных 2 млн фактов",
          "Число ячеек растёт кубически, заполненность падает. Здесь проходит\n"
          "граница применимости: куб перестаёт помещаться в память раньше,\n"
          "чем у DuckDB кончаются возможности.",
          out, "сторона куба", [f"{s}³ = {s**3:,}".replace(",", " ") for s in sides])
    return out


def sweep_skew(args) -> list[dict]:
    skews = [0.0, 0.6, 1.0, 1.4]
    out = []
    for k in skews:
        print(f"  ... перекос {k}", flush=True)
        out.append(measure(2_000_000, 200, 200, 100, k, args.repeat))
    table("РАЗВЁРТКА 3. Перекос распределения при кубе 200×200×100",
          "Степенной закон: немногие клиенты и товары дают большую часть продаж.\n"
          "Заполненность падает — проверяется, что даёт разреженность.",
          out, "перекос", [f"{k}" for k in skews])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--only", choices=["rows", "cells", "skew"], action="append")
    p.add_argument("--heavy", action="store_true",
                   help="добавить самые крупные точки (дольше и больше памяти)")
    p.add_argument("--max-cells", type=float, default=2.5e7,
                   help="потолок числа ячеек плотного куба")
    a = p.parse_args()
    a.max_cells = int(a.max_cells)

    try:
        import duckdb
    except ImportError:
        print("DuckDB не установлен: pip install duckdb")
        return 1

    from amdb.exec.engine import blas_info
    info = blas_info()
    print(f"NumPy {info['numpy']}, BLAS {info.get('blas')}, DuckDB {duckdb.__version__}")
    print(f"Выдача — кадр данных; повторов на замер: {a.repeat}")

    chosen = a.only or ["rows", "cells", "skew"]
    started = time.perf_counter()
    if "rows" in chosen:
        print("\nРазвёртка по объёму фактов:")
        sweep_rows(a)
    if "cells" in chosen:
        print("\nРазвёртка по размеру куба:")
        sweep_cells(a)
    if "skew" in chosen:
        print("\nРазвёртка по перекосу:")
        sweep_skew(a)
    print(f"\nВсего {time.perf_counter() - started:.0f} с.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
