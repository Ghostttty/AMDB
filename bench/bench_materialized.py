# -*- coding: utf-8 -*-
"""Симметричное сравнение: гиперкуб против материализованного агрегата DuckDB.

Возражение против обычного сравнения справедливо: AMDB пользуется заранее
построенным гиперкубом, а DuckDB каждый раз обходит сырые строки. Сравнивались
не движки, а **модели хранения**.

Этот стенд снимает возражение: DuckDB получает таблицу той же гранулярности,
что и гиперкуб, — факт, свёрнутый до (customer, product, date) и снабжённый
счётчиком строк, то есть точный аналог спутникового счётного куба. Теперь обе
системы стартуют из предпосчитанной сводки, и сравнение становится сравнением
**исполнителей**, а не моделей.

Приводятся все три режима, чтобы разница была видна:

* DuckDB по сырым строкам — исходный вариант;
* DuckDB по материализованному агрегату — симметричный вариант;
* AMDB по гиперкубу.

    python bench/bench_materialized.py --rows 2000000
"""
from __future__ import annotations

import argparse
import time

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.workload import (
    MATERIALIZED_SQL,
    QUERIES,
    build_amdb,
    build_duckdb,
    build_duckdb_materialized,
    make_workload,
    normalize,
)


def timeit(fn, repeat: int) -> tuple[float, object]:
    out = fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def rows_of(df, nkeys):
    return normalize(list(df.itertuples(index=False, name=None)), nkeys)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=2_000_000)
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--products", type=int, default=100)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    try:
        import duckdb
    except ImportError:
        print("DuckDB не установлен: pip install duckdb")
        return 1

    work = make_workload(a.rows, a.customers, a.products, a.days, a.seed)
    print(f"DuckDB {duckdb.__version__}; {a.rows:,} строк фактов, гиперкуб "
          f"{a.customers}×{a.products}×{a.days}\n".replace(",", " "))

    t0 = time.perf_counter()
    db = build_amdb(work)
    t_cube = time.perf_counter() - t0
    con_raw = build_duckdb(work)
    con_raw.execute("SELECT COUNT(*) FROM sales").fetchall()
    con_mat, t_mat = build_duckdb_materialized(work)

    cells = con_mat.execute("SELECT COUNT(*) FROM sales").fetchall()[0][0]
    print("Построение сводок:")
    print(f"  AMDB, гиперкуб                     {t_cube * 1e3:8.0f} мс")
    print(f"  DuckDB, материализованный агрегат  {t_mat * 1e3:8.0f} мс "
          f"({cells:,} строк вместо {a.rows:,})".replace(",", " "))
    print()

    head = (f"{'запрос':<38}{'AMDB':>10}{'DuckDB сырой':>14}"
            f"{'DuckDB агрегат':>16}{'AMDB / агрегат':>16}{'сверка':>9}")
    print(head)
    print("-" * len(head))

    wins = losses = 0
    for q in QUERIES:
        mat_sql = MATERIALIZED_SQL.get(q.name)
        if mat_sql is None:
            continue
        t_a, res_a = timeit(lambda: db.sql(q.amdb).to_pandas(), a.repeat)
        t_raw, _ = timeit(lambda: con_raw.execute(q.duck).fetchdf(), a.repeat)
        t_m, res_m = timeit(lambda: con_mat.execute(mat_sql).fetchdf(), a.repeat)

        got = rows_of(res_a, len(q.keys))
        ref = rows_of(res_m, len(q.keys))
        ok = len(got) == len(ref) and all(
            g[:-1] == r[:-1] and abs(g[-1] - r[-1]) <= q.tolerance * max(abs(r[-1]), 1.0)
            for g, r in zip(got, ref))
        ratio = t_m / t_a
        if ok:
            wins += ratio > 1.05
            losses += ratio < 0.95
        print(f"{q.name:<38}{t_a * 1e3:>9.2f}м{t_raw * 1e3:>13.2f}м"
              f"{t_m * 1e3:>15.2f}м{ratio:>15.2f}×"
              f"{'совпало' if ok else 'РАСХОЖД.':>9}")

    print()
    print(f"Против материализованного агрегата: AMDB быстрее на {wins} запросах, "
          f"медленнее на {losses}.")
    print("\nЧто показывает этот стенд: часть преимущества AMDB объясняется не")
    print("алгеброй, а тем, что сводка предпосчитана. Столбец «AMDB / агрегат»")
    print("отделяет вклад собственно матричного исполнения от вклада")
    print("предварительной агрегации.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
