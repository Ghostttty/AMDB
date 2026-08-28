# -*- coding: utf-8 -*-
"""B11: три исполнителя на одном наборе — AMDB, DuckDB и ClickHouse.

DuckDB отвечает на возражение «сравните с колоночной СУБД», ClickHouse — на
возражение «возьмите признанный OLAP-движок». Обе взяты во встраиваемом виде и
работают в том же процессе, что и AMDB: так в замер не попадают ни сетевой
обмен, ни клиент-серверные накладные расходы.

Условия сравнения выровнены по четырём пунктам, каждый из которых иначе исказил
бы результат:

* данные лежат в собственных таблицах каждой системы, а не читаются из кадра
  pandas при каждом запросе;
* число потоков у обоих движков одинаково (по умолчанию — все ядра);
* столбцы объявлены не-nullable: иначе ClickHouse не примет их в ключ
  сортировки MergeTree и станет считать медленнее;
* у chdb измеряются и вычитаются постоянные накладные расходы на запрос
  (порядка 0.7 мс) — на запросах в единицы миллисекунд это заметная доля.

Результаты **сверяются построчно** со всеми тремя системами; расхождение
считается провалом запроса.

    python bench/bench_clickhouse.py --rows 5000000
    python bench/bench_clickhouse.py --rows 500000 --materialized

Требуется chdb (встраиваемый ClickHouse); под Windows его нет, стенд
рассчитан на Linux — см. раздел «Проверка в Linux» в README.
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
    clickhouse_overhead,
    make_workload,
    normalize,
    to_clickhouse,
)


def timeit(fn, repeat: int) -> tuple[float, object]:
    out = fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def ch_rows(session, sql: str) -> list[tuple]:
    """Результат ClickHouse в виде кортежей — для построчной сверки."""
    import io

    import pandas as pd

    text = str(session.query(sql, "CSV"))
    if not text.strip():
        return []
    frame = pd.read_csv(io.StringIO(text), header=None)
    return list(frame.itertuples(index=False, name=None))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--products", type=int, default=100)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", choices=["Memory", "MergeTree"], default="Memory",
                   help="представление таблицы фактов в ClickHouse")
    p.add_argument("--materialized", action="store_true",
                   help="симметричный режим: у всех трёх предпосчитанная сводка")
    a = p.parse_args()

    try:
        import chdb
    except ImportError:
        print("chdb не установлен. Под Windows его нет; стенд рассчитан на Linux — "
              "см. раздел «Проверка в Linux» в README.")
        return 1
    import duckdb

    from bench.workload import build_clickhouse, build_clickhouse_materialized

    work = make_workload(a.rows, a.customers, a.products, a.days, a.seed)
    print(f"DuckDB {duckdb.__version__}; ClickHouse через chdb {chdb.__version__}; "
          f"{a.rows:,} строк фактов, гиперкуб "
          f"{a.customers}×{a.products}×{a.days}".replace(",", " "))

    t0 = time.perf_counter()
    db = build_amdb(work)
    t_cube = time.perf_counter() - t0

    if a.materialized:
        con, t_duck = build_duckdb_materialized(work)
        session, t_ch = build_clickhouse_materialized(work)
        sql_of = lambda q: MATERIALIZED_SQL.get(q.name)
        mode = "все три системы стартуют из предпосчитанной сводки"
    else:
        t0 = time.perf_counter()
        con = build_duckdb(work)
        t_duck = time.perf_counter() - t0
        t0 = time.perf_counter()
        session = build_clickhouse(work, engine=a.engine)
        t_ch = time.perf_counter() - t0
        sql_of = lambda q: q.duck
        mode = f"колоночные СУБД работают по сырым фактам; ClickHouse: {a.engine}"

    floor = clickhouse_overhead(session)
    print(f"\n{mode}")
    print(f"подготовка данных: AMDB {t_cube * 1e3:.0f} мс, DuckDB {t_duck * 1e3:.0f} мс, "
          f"ClickHouse {t_ch * 1e3:.0f} мс")
    print(f"постоянные накладные chdb: {floor * 1e3:.2f} мс на запрос (вычитаются)\n")

    head = (f"{'запрос':<38}{'AMDB':>9}{'DuckDB':>10}{'ClickHouse':>12}"
            f"{'AMDB/DuckDB':>13}{'AMDB/CH':>10}{'сверка':>9}")
    print(head)
    print("-" * len(head))

    failures = 0
    for q in QUERIES:
        sql = sql_of(q)
        if sql is None:
            continue
        ta, res_a = timeit(lambda: db.sql(q.amdb).to_pandas(), a.repeat)
        td, res_d = timeit(lambda: con.execute(sql).fetchdf(), a.repeat)
        ch_sql = to_clickhouse(sql)
        tc, _ = timeit(lambda: session.query(ch_sql, "Arrow"), a.repeat)
        tc = max(tc - floor, 0.0)

        ref = normalize(list(res_d.itertuples(index=False, name=None)), len(q.keys))
        got = normalize(list(res_a.itertuples(index=False, name=None)), len(q.keys))
        chv = normalize(ch_rows(session, ch_sql), len(q.keys))

        def same(rows):
            return len(rows) == len(ref) and all(
                g[:-1] == r[:-1]
                and abs(g[-1] - r[-1]) <= q.tolerance * max(abs(r[-1]), 1.0)
                for g, r in zip(rows, ref))

        ok = same(got) and same(chv)
        failures += not ok
        print(f"{q.name:<38}{ta * 1e3:>8.2f}м{td * 1e3:>9.2f}м{tc * 1e3:>11.2f}м"
              f"{td / ta:>12.2f}×{tc / ta:>9.2f}×"
              f"{'совпало' if ok else 'РАСХОЖД.':>9}")

    print()
    if failures:
        print(f"ВНИМАНИЕ: расхождение результатов на {failures} запросах — "
              "замер недействителен.")
        return 2
    print("Все результаты совпали во всех трёх системах.")
    print("Выигрыш AMDB дают запросы, где соединение, отбор и свёртка по иерархии")
    print("складываются в одно стягивание; на одиночной агрегации его нет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
