# -*- coding: utf-8 -*-
"""Сквозная демонстрация AMDB на синтетической звёздной схеме.

    python examples/generate.py --out data
    python examples/demo.py --data data
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb import Database
from amdb.storage.loader import read_csv

QUERIES = [
    ("Выручка по клиентам и месяцам (пример §6.1 техпроекта)", """
        SELECT customer, month, SUM(quantity * price) AS revenue
        FROM sales JOIN product ON sales.product = product.product
        GROUP BY customer, month
    """),
    ("То же с фильтром по региону (§6.2)", """
        SELECT customer, month, SUM(quantity * price) AS revenue
        FROM sales JOIN product ON sales.product = product.product
        WHERE region = 'Смоленск'
        GROUP BY customer, month
    """),
    ("Топ категорий по выручке", """
        SELECT category_dim, SUM(quantity * price) AS revenue
        FROM sales JOIN product ON sales.product = product.product
        GROUP BY category_dim
        ORDER BY revenue DESC
    """),
    ("Средний чек и число продаж по кварталам", """
        SELECT quarter, COUNT(*) AS deals, AVG(quantity) AS avg_qty
        FROM sales GROUP BY quarter
    """),
    ("Накопительная выручка по месяцам (оконная функция, §6.5)", """
        SELECT month, SUM(quantity * price) OVER (ORDER BY month) AS running
        FROM sales JOIN product ON sales.product = product.product
        GROUP BY month
    """),
    ("Максимум по ячейке куба (редукция вне einsum, §6.4)", """
        SELECT quarter, MAX(quantity) AS peak FROM sales GROUP BY quarter
    """),
]


def build(data: Path) -> Database:
    db = Database()
    t0 = time.perf_counter()
    cube = db.load_csv(data / "sales.csv", ["customer", "product", "date"],
                       "quantity", "sales", ordered_dims=["date"])
    t_load = time.perf_counter() - t0

    products = read_csv(data / "products.csv")
    db.load_dimension(products, "product", attributes=["category"], measures=["price"])
    customers = read_csv(data / "customers.csv")
    db.load_dimension(customers, "customer", attributes=["region"])

    dates = read_csv(data / "dates.csv")
    order = db.catalog.dimension("date").encode(dates["date"])
    months = [None] * len(order)
    quarters = [None] * len(order)
    for pos, m, q in zip(order, dates["month"], dates["quarter"]):
        months[pos], quarters[pos] = m, q
    db.add_hierarchy("date", "month", months)
    db.add_hierarchy("month", "quarter",
                     _month_to_quarter(db, months, quarters))
    # Категория как измерение (иерархия product -> category_dim) делает
    # GROUP BY по категории обычной (0,1)-свёрткой.
    cat_order = db.catalog.dimension("product").encode(products["product"])
    cats = [None] * len(cat_order)
    for pos, c in zip(cat_order, products["category"]):
        cats[pos] = c
    db.add_hierarchy("product", "category_dim", cats)

    stats = db.stats("sales")
    print(f"Загрузка: {t_load * 1e3:.0f} мс, гиперкуб {cube.shape} "
          f"({stats['total_cells']:,} ячеек, заполнено {stats['fill_factor']:.2%}, "
          f"представление {cube.layout})")
    return db


def _month_to_quarter(db: Database, months, quarters) -> list:
    mapping: dict = {}
    for m, q in zip(months, quarters):
        mapping.setdefault(m, q)
    return [mapping[m] for m in db.catalog.dimension("month").labels()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--rows", type=int, default=8, help="строк результата на экран")
    p.add_argument("--explain", action="store_true", help="печатать планы")
    a = p.parse_args()
    if not (a.data / "sales.csv").exists():
        raise SystemExit(f"нет данных в {a.data}; сначала запустите examples/generate.py")

    db = build(a.data)
    print()
    print(db.summary())

    for title, sql in QUERIES:
        print("\n" + "=" * 78)
        print(title)
        print("-" * 78)
        if a.explain:
            print(db.explain(sql))
            print("-" * 78)
        else:
            print("einsum:", ", ".join(db.einsum_of(sql)))
        res = db.sql(sql)
        print(res.to_text(max_rows=a.rows))
        print(f"трансляция {res.stats['compile_seconds'] * 1e3:.2f} мс, "
              f"вычисление {res.stats['compute_seconds'] * 1e3:.2f} мс")

    print("\n" + "=" * 78)
    print("Разграничение доступа по измерениям")
    print("-" * 78)
    smolensk = [c for c, r in zip(db.dimensions["customer"].labels(),
                                  db.dimensions["customer"].attributes["region"])
                if r == "Смоленск"]
    db.grant("smolensk_analyst", "customer", allowed=smolensk)
    full = db.sql("SELECT COUNT(*) AS n FROM sales")
    limited = db.sql("SELECT COUNT(*) AS n FROM sales", role="smolensk_analyst")
    print(f"без роли: {full[0][0]:,.0f} записей; "
          f"роль smolensk_analyst видит {limited[0][0]:,.0f} "
          f"({limited[0][0] / full[0][0]:.1%})")


if __name__ == "__main__":
    main()
