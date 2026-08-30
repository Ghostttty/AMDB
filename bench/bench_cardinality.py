# -*- coding: utf-8 -*-
"""B15: рост мощности измерения — где схема выводит гиперкуб за границу.

Критерий §4.10 статьи сформулирован через число ячеек, а число ячеек есть
произведение мощностей измерений. Стенд переводит критерий в требование к
схеме: мощность одного измерения растёт на четыре порядка при неизменных
числе фактов и мощностях остальных осей.

Меряются четыре величины, а не одна: выбранное представление, время и пиковая
память построения, время запроса. Ограничение по памяти наступает раньше
ограничения по времени, и без её замера картина неполна.

Сравнение симметричное: обе колоночные СУБД получают сводку гранулярности
гиперкуба, то есть тот же предпосчёт, что и матричная модель. Иначе стенд
измерял бы предварительную агрегацию, а не исполнение.

    python bench/bench_cardinality.py
    python bench/bench_cardinality.py --rows 2000000 --cards 100,1000,10000
    python bench/bench_cardinality.py --extreme     # 10^6 x 10^5 ячеек

ClickHouse требует chdb и потому доступен только под Linux; без него столбец
остаётся пустым, а остальные — осмысленны.
"""
from __future__ import annotations

import argparse
import gc
import os
import time
import tracemalloc

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from amdb import Database
from amdb.storage.policy import choose_layout, estimate_bytes
from bench.workload import clickhouse_overhead

QUERY = "SELECT customer, SUM(quantity) AS s FROM sales GROUP BY customer"
SUMMARY_SQL = "SELECT customer, SUM(quantity) AS s FROM agg GROUP BY customer"


def timeit(fn, repeat: int = 3) -> float:
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def make_frame(pd, rows: int, n_customer: int, n_product: int, n_month: int):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "customer": rng.integers(0, n_customer, rows),
        "product": rng.integers(0, n_product, rows),
        "month": rng.integers(0, n_month, rows),
        "quantity": rng.random(rows),
    })


def clickhouse_ms(pd, summary, tag: str) -> float:
    """Время запроса к сводке в ClickHouse за вычетом накладных расходов."""
    try:
        import chdb.session
    except ImportError:
        return float("nan")
    path = os.path.join(os.path.dirname(__file__), f"_card_{tag}.parquet")
    session = None
    try:
        summary.to_parquet(path)
        session = chdb.session.Session()
        session.query(f"CREATE DATABASE d{tag}")
        session.query(
            f"CREATE TABLE d{tag}.agg ENGINE = MergeTree ORDER BY customer AS "
            f"SELECT * FROM file('{path}', Parquet) "
            "SETTINGS schema_inference_make_columns_nullable=0")
        sql = SUMMARY_SQL.replace("agg", f"d{tag}.agg")
        # Формат Arrow, как в B11: результат материализуется, иначе замер
        # показывает только разбор запроса.
        overhead = clickhouse_overhead(session)
        return (timeit(lambda: session.query(sql, "Arrow")) - overhead) * 1e3
    except Exception as exc:                      # pragma: no cover — среда
        print(f"    chdb: {str(exc)[:70]}")
        return float("nan")
    finally:
        if session is not None:
            session.close()
        if os.path.exists(path):
            os.remove(path)


def run_point(pd, duckdb, rows: int, nc: int, np_: int, nm: int, tag: str) -> None:
    frame = make_frame(pd, rows, nc, np_, nm)
    shape = (nc, np_, nm)
    summary = frame.groupby(["customer", "product", "month"], as_index=False).agg(
        quantity=("quantity", "sum"), n=("quantity", "size"))
    layout = choose_layout(len(summary), shape, 8)

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        db = Database()
        db.load_frame(frame, ["customer", "product", "month"], "quantity", "sales")
        load = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        ours = timeit(lambda: db.sql(QUERY)) * 1e3
    except Exception as exc:
        tracemalloc.stop()
        load = peak = ours = float("nan")
        print(f"    AMDB: {type(exc).__name__}: {str(exc)[:70]}")

    con = duckdb.connect()
    con.register("_s", summary)
    con.execute("CREATE TABLE agg AS SELECT * FROM _s")
    duck = timeit(lambda: con.execute(SUMMARY_SQL).fetchdf()) * 1e3
    con.close()

    ch = clickhouse_ms(pd, summary, tag)
    ratio = duck / ours if ours == ours else float("nan")
    print("%-12s %13s %12s %10.2f %11.0f %13.2f %11.2f %11.2f %8.2f×"
          % (format(nc, ","), format(nc * np_ * nm, ","), layout, load, peak,
             ours, duck, ch, ratio))
    del frame, summary
    gc.collect()


def extreme(pd, rows: int) -> None:
    """Предельный случай: ключевые измерения высокой мощности."""
    nc, npr = 1_000_000, 100_000
    print("\nПредельный случай: %s клиентов × %s товаров" % (f"{nc:,}", f"{npr:,}"))
    print("  ячеек %.1e, плотно вместе со счётным кубом %.1f ТиБ"
          % (nc * npr, 2 * estimate_bytes((nc, npr), 8) / 2**40))
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"customer": rng.integers(0, nc, rows),
                          "product": rng.integers(0, npr, rows),
                          "quantity": rng.random(rows)})
    t0 = time.perf_counter()
    db = Database()
    db.load_frame(frame, ["customer", "product"], "quantity", "sales")
    print("  представление: %s, загрузка %.2f с"
          % (db.catalog.cube("sales").layout, time.perf_counter() - t0))
    for sql in ("SELECT SUM(quantity) AS s FROM sales", QUERY):
        t0 = time.perf_counter()
        n_rows = len(db.sql(sql).rows)
        print("  %-58s %7.1f мс, строк %s"
              % (sql, (time.perf_counter() - t0) * 1e3, f"{n_rows:,}"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--cards", default="100,1000,10000,100000,1000000",
                   help="мощности растущего измерения через запятую")
    p.add_argument("--products", type=int, default=100)
    p.add_argument("--months", type=int, default=12)
    p.add_argument("--extreme", action="store_true",
                   help="дополнительно прогнать предельный случай 10^6 × 10^5")
    args = p.parse_args()

    try:
        import duckdb
        import pandas as pd
    except ImportError:
        print("нужны pandas и duckdb: pip install -e \".[bench]\"")
        return 1

    print("фактов %s, товаров %d, месяцев %d"
          % (f"{args.rows:,}", args.products, args.months))
    head = ("%-12s %13s %12s %10s %11s %13s %11s %11s %8s"
            % ("клиентов", "ячеек", "представл.", "загр., с", "память МиБ",
               "AMDB, мс", "DuckDB, мс", "CH, мс", "отнош."))
    print()
    print(head)
    print("-" * len(head))
    for nc in (int(x) for x in args.cards.split(",")):
        run_point(pd, duckdb, args.rows, nc, args.products, args.months, str(nc))

    print("\nВремя AMDB определяется числом ячеек, время колоночной СУБД — длиной")
    print("сводки. При неизменном числе фактов сводка почти не растёт, поэтому")
    print("преимущество теряется вместе с ростом мощности измерения.")

    if args.extreme:
        extreme(pd, min(args.rows, 5_000_000))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
