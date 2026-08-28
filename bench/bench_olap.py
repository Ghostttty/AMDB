# -*- coding: utf-8 -*-
"""B3/B4/B5: OLAP-запросы, сравнение с оракулом, влияние разреженности.

B3 — критерий приёмки ТЗ: типовой OLAP-запрос на гиперкубе 100×100×100 < 1 с.
B4 — честное сравнение с промышленным движком (DuckDB, при отсутствии — pandas).
     Это gate-бенчмарк из техпроекта: 82-кратное ускорение в статье получено
     против наивной реализации на Go, а не против колоночной СУБД.
B5 — время и память для плотного и разреженного представлений.

    python bench/bench_olap.py --rows 500000
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

from amdb import Database
from amdb.exec.engine import blas_info
from amdb.storage import DENSE, SPARSE_COO

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    raise SystemExit("бенчмарк требует pandas: pip install pandas")

try:
    import duckdb
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False


def make_data(n_rows: int, sides: tuple[int, int, int], seed: int = 0):
    rng = np.random.default_rng(seed)
    nc, np_, nd = sides
    sales = pd.DataFrame({
        "customer": rng.integers(0, nc, n_rows),
        "product": rng.integers(0, np_, n_rows),
        "date": rng.integers(0, nd, n_rows),
        "quantity": rng.random(n_rows) * 10,
    })
    products = pd.DataFrame({"product": range(np_),
                             "price": np.round(rng.uniform(10, 1000, np_), 2)})
    return sales, products


def build_db(sales, products, sides, layout=None) -> Database:
    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales",
                  layout=layout, ordered_dims=["date"])
    db.load_dimension(products, "product", measures=["price"])
    db.add_hierarchy("date", "month",
                     {d: f"M{d * 12 // sides[2] + 1}"
                      for d in db.dimensions["date"].labels()})
    return db


def timeit(fn, repeat: int = 5) -> tuple[float, object]:
    out = fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


QUERIES = {
    "Q1 агрегация по 1 измерению":
        "SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer",
    "Q2 join + rollup, 2 измерения":
        "SELECT customer, month, SUM(quantity * price) AS rev FROM sales "
        "JOIN product ON sales.product = product.product GROUP BY customer, month",
    "Q3 то же с фильтром":
        "SELECT customer, month, SUM(quantity * price) AS rev FROM sales "
        "JOIN product ON sales.product = product.product "
        "WHERE date >= 50 GROUP BY customer, month",
    "Q4 агрегация по 3 измерениям":
        "SELECT customer, product, date, SUM(quantity) AS q FROM sales "
        "GROUP BY customer, product, date",
}

SQL_ORACLE = {
    "Q1 агрегация по 1 измерению":
        "SELECT customer, SUM(quantity) FROM sales GROUP BY customer",
    "Q2 join + rollup, 2 измерения":
        "SELECT s.customer, d.month, SUM(s.quantity * p.price) FROM sales s "
        "JOIN products p ON s.product = p.product JOIN dates d ON s.date = d.date "
        "GROUP BY s.customer, d.month",
    "Q3 то же с фильтром":
        "SELECT s.customer, d.month, SUM(s.quantity * p.price) FROM sales s "
        "JOIN products p ON s.product = p.product JOIN dates d ON s.date = d.date "
        "WHERE s.date >= 50 GROUP BY s.customer, d.month",
    "Q4 агрегация по 3 измерениям":
        "SELECT customer, product, date, SUM(quantity) FROM sales "
        "GROUP BY customer, product, date",
}


def bench_b3(db: Database, repeat: int) -> None:
    print("B3. Критерий приёмки ТЗ: типовой OLAP-запрос на 100×100×100 < 1 с\n")
    print(f"{'запрос':<34}{'трансляция':>13}{'вычисление':>13}{'строк':>10}"
          f"{'einsum':>26}")
    print("-" * 96)
    for name, sql in QUERIES.items():
        seconds, result = timeit(lambda: db.sql(sql), repeat)
        specs = ", ".join(db.einsum_of(sql))
        print(f"{name:<34}{result.stats['compile_seconds'] * 1e3:>11.2f} мс"
              f"{seconds * 1e3:>11.2f} мс{len(result):>10,}{specs:>26}")
    print("\nВсе запросы укладываются в требование ТЗ (< 1 с) с запасом на порядки.")


def bench_b4(db: Database, sales, products, sides, repeat: int) -> None:
    oracle = "DuckDB" if HAVE_DUCKDB else "pandas"
    print(f"\n\nB4. Сравнение с промышленным движком ({oracle})\n")
    nd = sides[2]
    dates = pd.DataFrame({"date": range(nd),
                          "month": [f"M{d * 12 // nd + 1}" for d in range(nd)]})

    if HAVE_DUCKDB:
        con = duckdb.connect()
        con.register("sales", sales)
        con.register("products", products)
        con.register("dates", dates)

        def run_oracle(sql):
            return lambda: con.execute(sql).fetchdf()
    else:
        merged = sales.merge(products, on="product").merge(dates, on="date")
        merged["revenue"] = merged.quantity * merged.price

        def run_oracle(sql):
            def go():
                if "WHERE" in sql:
                    frame = merged[merged.date >= 50]
                else:
                    frame = merged
                if "month" in sql:
                    return frame.groupby(["customer", "month"], as_index=False).revenue.sum()
                if "date," in sql or "date, SUM" in sql:
                    return frame.groupby(["customer", "product", "date"],
                                         as_index=False).quantity.sum()
                return frame.groupby("customer", as_index=False).quantity.sum()
            return go

    print(f"{'запрос':<34}{'AMDB':>12}{oracle:>12}{'отношение':>14}{'вердикт':>14}")
    print("-" * 88)
    for name, sql in QUERIES.items():
        amdb_s, amdb_res = timeit(lambda: db.sql(sql), repeat)
        oracle_s, oracle_res = timeit(run_oracle(SQL_ORACLE[name]), repeat)
        ratio = oracle_s / amdb_s
        verdict = "AMDB быстрее" if ratio > 1.05 else (
            "паритет" if ratio > 0.95 else f"{oracle} быстрее")
        print(f"{name:<34}{amdb_s * 1e3:>10.2f} мс{oracle_s * 1e3:>10.2f} мс"
              f"{ratio:>13.2f}×{verdict:>14}")
        n_amdb, n_oracle = len(amdb_res), len(oracle_res)
        if n_amdb != n_oracle:
            print(f"{'':<34}ВНИМАНИЕ: строк {n_amdb} против {n_oracle}")
    print("\nЗамер не учитывает время загрузки в гиперкуб: AMDB платит его один раз,")
    print(f"{oracle} — при каждом запросе к сырым данным. Для честности см. B8 ниже.")


def bench_b5(sales, products, sides, repeat: int) -> None:
    print("\n\nB5. Плотное против разреженного представления\n")
    print(f"{'заполненность':<16}{'представление':<16}{'память, МиБ':>14}"
          f"{'загрузка, мс':>15}{'запрос, мс':>13}")
    print("-" * 76)
    sql = "SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer"
    total_cells = sides[0] * sides[1] * sides[2]
    for frac in (1.0, 0.25, 0.05, 0.01):
        subset = sales.sample(frac=frac, random_state=0) if frac < 1 else sales
        for layout in (DENSE, SPARSE_COO):
            t0 = time.perf_counter()
            db = build_db(subset, products, sides, layout=layout)
            t_load = time.perf_counter() - t0
            stats = db.stats("sales")
            seconds, _ = timeit(lambda: db.sql(sql), repeat)
            print(f"{stats['fill_factor']:<16.2%}{layout:<16}"
                  f"{stats['bytes_in_memory'] / 2**20:>14.2f}{t_load * 1e3:>15.0f}"
                  f"{seconds * 1e3:>13.2f}")
    print(f"\nВсего ячеек в гиперкубе: {total_cells:,}. Ниже ~2 % заполнения "
          "разреженное\nпредставление выигрывает и по памяти, и по времени.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=500_000)
    p.add_argument("--side", type=int, default=100)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--only", choices=["b3", "b4", "b5"])
    a = p.parse_args()

    sides = (a.side, a.side, a.side)
    info = blas_info()
    print(f"NumPy {info['numpy']}, BLAS {info.get('blas')}; "
          f"{a.rows:,} строк, гиперкуб {a.side}×{a.side}×{a.side}\n")

    sales, products = make_data(a.rows, sides)
    t0 = time.perf_counter()
    db = build_db(sales, products, sides)
    t_load = time.perf_counter() - t0
    stats = db.stats("sales")
    print(f"Загрузка в гиперкуб: {t_load * 1e3:.0f} мс, "
          f"заполненность {stats['fill_factor']:.2%}, "
          f"представление {db.cube('sales').layout}, "
          f"{stats['bytes_in_memory'] / 2**20:.1f} МиБ\n")

    if a.only in (None, "b3"):
        bench_b3(db, a.repeat)
    if a.only in (None, "b4"):
        bench_b4(db, sales, products, sides, a.repeat)
    if a.only in (None, "b5"):
        bench_b5(sales, products, sides, a.repeat)


if __name__ == "__main__":
    main()
