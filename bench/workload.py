# -*- coding: utf-8 -*-
"""Общая нагрузка для сравнения AMDB и DuckDB.

Здесь собраны схема, генератор данных и набор запросов в двух записях — на
языке AMDB и на SQL для DuckDB. Модуль используют и стенд ``bench_duckdb.py``,
и тест ``tests/test_duckdb_parity.py``, чтобы замеряемое и проверяемое были
одним и тем же.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

REGIONS = ["Смоленск", "Москва", "Санкт-Петербург", "Новосибирск"]
CATEGORIES = ["электроника", "одежда", "продукты", "книги", "мебель"]


@dataclass
class Workload:
    """Звёздная схема: факт продаж и три справочника."""

    sales: "pd.DataFrame"
    products: "pd.DataFrame"
    customers: "pd.DataFrame"
    dates: "pd.DataFrame"
    n_customers: int
    n_products: int
    n_days: int

    @property
    def cells(self) -> int:
        return self.n_customers * self.n_products * self.n_days

    @property
    def rows(self) -> int:
        return len(self.sales)


def make_workload(rows: int = 500_000, customers: int = 100, products: int = 100,
                  days: int = 100, seed: int = 42, skew: float = 0.0) -> Workload:
    """Генерирует звёздную схему.

    ``skew`` > 0 задаёт степенное распределение продаж по клиентам и товарам —
    это ближе к реальности, чем равномерное, и заметно меняет разреженность
    гиперкуба, то есть условия, в которых сравниваются системы.
    """
    if pd is None:  # pragma: no cover
        raise ImportError("нужен pandas: pip install pandas")
    rng = np.random.default_rng(seed)

    def pick(n: int) -> np.ndarray:
        if skew <= 0:
            return rng.integers(0, n, rows)
        w = 1.0 / np.power(np.arange(1, n + 1), skew)
        return rng.choice(n, size=rows, p=w / w.sum())

    sales = pd.DataFrame({
        "customer": pick(customers),
        "product": pick(products),
        "date": rng.integers(0, days, rows),
        "quantity": np.round(rng.uniform(1, 20, rows), 3),
    })
    products_df = pd.DataFrame({
        "product": np.arange(products),
        "price": np.round(rng.uniform(10, 1000, products), 2),
        "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(products)],
    })
    customers_df = pd.DataFrame({
        "customer": np.arange(customers),
        "region": [REGIONS[i % len(REGIONS)] for i in range(customers)],
    })
    dates_df = pd.DataFrame({
        "date": np.arange(days),
        "month": [f"M{d * 12 // days + 1:02d}" for d in range(days)],
        "quarter": [f"Q{d * 4 // days + 1}" for d in range(days)],
    })
    return Workload(sales, products_df, customers_df, dates_df,
                    customers, products, days)


@dataclass
class Query:
    """Один запрос в двух записях плюс описание ожидаемого поведения."""

    name: str
    amdb: str
    duck: str
    keys: tuple[str, ...]           # столбцы-ключи результата
    value: str                      # сравниваемый столбец
    comment: str = ""
    tolerance: float = 1e-9         # относительная, для сверки результатов
    expect: str = ""                # ожидание: где AMDB должна выигрывать
    skip_reason: str = field(default="")


#: Набор подобран так, чтобы покрыть и сильные, и слабые стороны подхода.
QUERIES: list[Query] = [
    Query(
        "Q1. Скалярная сумма",
        "SELECT SUM(quantity) AS v FROM sales",
        "SELECT SUM(quantity) AS v FROM sales",
        (), "v",
        "Полная свёртка куба в число",
        expect="AMDB: одна свёртка без материализации результата",
    ),
    Query(
        "Q2. Группировка по одному измерению",
        "SELECT customer, SUM(quantity) AS v FROM sales GROUP BY customer",
        "SELECT customer, SUM(quantity) AS v FROM sales GROUP BY customer",
        ("customer",), "v",
        "Сворачиваются две оси из трёх",
        expect="AMDB: сильное схлопывание измерений",
    ),
    Query(
        "Q3. Соединение со справочником",
        "SELECT customer, SUM(quantity * price) AS v FROM sales "
        "JOIN product ON sales.product = product.product GROUP BY customer",
        "SELECT s.customer, SUM(s.quantity * p.price) AS v FROM sales s "
        "JOIN products p ON s.product = p.product GROUP BY s.customer",
        ("customer",), "v",
        "Мера из другого куба входит сомножителем в ту же свёртку",
        expect="AMDB: соединение не отдельный шаг",
    ),
    Query(
        "Q4. Соединение и свёртка по иерархии",
        "SELECT customer, month, SUM(quantity * price) AS v FROM sales "
        "JOIN product ON sales.product = product.product GROUP BY customer, month",
        "SELECT s.customer, d.month, SUM(s.quantity * p.price) AS v FROM sales s "
        "JOIN products p ON s.product = p.product JOIN dates d ON s.date = d.date "
        "GROUP BY s.customer, d.month",
        ("customer", "month"), "v",
        "Два соединения плюс группировка по двум измерениям",
    ),
    Query(
        "Q5. То же с избирательным фильтром",
        "SELECT customer, month, SUM(quantity * price) AS v FROM sales "
        "JOIN product ON sales.product = product.product "
        "WHERE region = 'Смоленск' GROUP BY customer, month",
        "SELECT s.customer, d.month, SUM(s.quantity * p.price) AS v FROM sales s "
        "JOIN products p ON s.product = p.product JOIN dates d ON s.date = d.date "
        "JOIN customers c ON s.customer = c.customer "
        "WHERE c.region = 'Смоленск' GROUP BY s.customer, d.month",
        ("customer", "month"), "v",
        "Фильтр по атрибуту измерения — маска-сомножитель",
        expect="AMDB: фильтр ускоряет свёртку, а не добавляет проход",
    ),
    Query(
        "Q6. Диапазонный фильтр",
        "SELECT month, SUM(quantity) AS v FROM sales WHERE date >= 50 GROUP BY month",
        "SELECT d.month, SUM(s.quantity) AS v FROM sales s "
        "JOIN dates d ON s.date = d.date WHERE s.date >= 50 GROUP BY d.month",
        ("month",), "v",
        "Диапазон по упорядоченному измерению",
    ),
    Query(
        "Q7. Средний чек и число сделок",
        "SELECT month, COUNT(*) AS v FROM sales GROUP BY month",
        "SELECT d.month, COUNT(*) AS v FROM sales s "
        "JOIN dates d ON s.date = d.date GROUP BY d.month",
        ("month",), "v",
        "COUNT по спутниковому счётному кубу",
    ),
    Query(
        "Q8. Среднее",
        "SELECT quarter, AVG(quantity) AS v FROM sales GROUP BY quarter",
        "SELECT d.quarter, AVG(s.quantity) AS v FROM sales s "
        "JOIN dates d ON s.date = d.date GROUP BY d.quarter",
        ("quarter",), "v",
        "Отношение двух свёрток",
        tolerance=1e-9,
    ),
    Query(
        "Q9. Умеренно широкий результат",
        "SELECT customer, product, SUM(quantity) AS v FROM sales "
        "GROUP BY customer, product",
        "SELECT customer, product, SUM(quantity) AS v FROM sales "
        "GROUP BY customer, product",
        ("customer", "product"), "v",
        "Свёртка одной оси; результат в десятки тысяч строк",
    ),
    Query(
        "Q10. Широкий результат: без свёртки",
        "SELECT customer, product, date, SUM(quantity) AS v FROM sales "
        "GROUP BY customer, product, date",
        "SELECT customer, product, date, SUM(quantity) AS v FROM sales "
        "GROUP BY customer, product, date",
        ("customer", "product", "date"), "v",
        "Ни одна ось не сворачивается; результат — сотни тысяч строк",
        expect="преимущество падает: свёртывать нечего, время уходит на выдачу",
    ),
    Query(
        "Q11. COUNT DISTINCT по измерению",
        "SELECT customer, COUNT(DISTINCT product) AS v FROM sales GROUP BY customer",
        "SELECT customer, COUNT(DISTINCT product) AS v FROM sales GROUP BY customer",
        ("customer",), "v",
        "Свёртка над булевым полукольцом, затем подсчёт",
    ),
]


def build_amdb(work: Workload, **load_kwargs):
    """Собирает базу AMDB: куб факта, справочники, иерархии."""
    from amdb import Database

    db = Database()
    db.load_frame(work.sales, ["customer", "product", "date"], "quantity", "sales",
                  ordered_dims=["date"], **load_kwargs)
    db.load_dimension(work.products, "product", attributes=["category"],
                      measures=["price"])
    db.load_dimension(work.customers, "customer", attributes=["region"])
    # Иерархии задаются в порядке ординалов дочернего измерения.
    order = db.catalog.dimension("date").encode(work.dates["date"].to_numpy())
    months = [None] * len(order)
    quarters = [None] * len(order)
    for pos, m, q in zip(order, work.dates["month"], work.dates["quarter"]):
        months[pos], quarters[pos] = m, q
    db.add_hierarchy("date", "month", months)
    month_to_quarter = {m: q for m, q in zip(months, quarters)}
    db.add_hierarchy("month", "quarter",
                     [month_to_quarter[m] for m in db.catalog.dimension("month").labels()])
    return db


def build_duckdb(work: Workload, threads: int | None = None):
    """Регистрирует те же данные в DuckDB."""
    import duckdb

    con = duckdb.connect()
    if threads:
        con.execute(f"PRAGMA threads={threads}")
    con.register("sales", work.sales)
    con.register("products", work.products)
    con.register("customers", work.customers)
    con.register("dates", work.dates)
    return con


def build_duckdb_materialized(work: Workload, threads: int | None = None):
    """DuckDB с **материализованным агрегатом** той же гранулярности, что гиперкуб.

    Это симметричный аналог гиперкуба: факт свёрнут до уровня
    (customer, product, date) и снабжён счётчиком строк — ровно так же, как
    AMDB хранит спутниковый счётный куб. Без такой таблицы сравнение было бы
    асимметричным: одна система пользуется предпосчитанной сводкой, другая
    каждый раз обходит сырые строки.

    Возвращает (соединение, время построения в секундах).
    """
    import time

    import duckdb

    con = duckdb.connect()
    if threads:
        con.execute(f"PRAGMA threads={threads}")
    con.register("raw_sales", work.sales)
    con.register("products", work.products)
    con.register("customers", work.customers)
    con.register("dates", work.dates)

    t0 = time.perf_counter()
    con.execute("""
        CREATE TABLE sales AS
        SELECT customer, product, date,
               SUM(quantity) AS quantity,
               COUNT(*)      AS n
        FROM raw_sales
        GROUP BY customer, product, date
    """)
    con.execute("SELECT COUNT(*) FROM sales").fetchall()
    return con, time.perf_counter() - t0


#: Запросы к материализованному агрегату: SUM переносится на свёрнутую меру,
#: COUNT — на счётчик строк, как и в гиперкубе.
MATERIALIZED_SQL: dict[str, str] = {
    "Q1. Скалярная сумма": "SELECT SUM(quantity) AS v FROM sales",
    "Q2. Группировка по одному измерению":
        "SELECT customer, SUM(quantity) AS v FROM sales GROUP BY customer",
    "Q3. Соединение со справочником":
        "SELECT s.customer, SUM(s.quantity * p.price) AS v FROM sales s "
        "JOIN products p ON s.product = p.product GROUP BY s.customer",
    "Q4. Соединение и свёртка по иерархии":
        "SELECT s.customer, d.month, SUM(s.quantity * p.price) AS v FROM sales s "
        "JOIN products p ON s.product = p.product JOIN dates d ON s.date = d.date "
        "GROUP BY s.customer, d.month",
    "Q5. То же с избирательным фильтром":
        "SELECT s.customer, d.month, SUM(s.quantity * p.price) AS v FROM sales s "
        "JOIN products p ON s.product = p.product JOIN dates d ON s.date = d.date "
        "JOIN customers c ON s.customer = c.customer "
        "WHERE c.region = 'Смоленск' GROUP BY s.customer, d.month",
    "Q6. Диапазонный фильтр":
        "SELECT d.month, SUM(s.quantity) AS v FROM sales s "
        "JOIN dates d ON s.date = d.date WHERE s.date >= 50 GROUP BY d.month",
    "Q7. Средний чек и число сделок":
        "SELECT d.month, SUM(s.n) AS v FROM sales s "
        "JOIN dates d ON s.date = d.date GROUP BY d.month",
    "Q8. Среднее":
        "SELECT d.quarter, SUM(s.quantity) / SUM(s.n) AS v FROM sales s "
        "JOIN dates d ON s.date = d.date GROUP BY d.quarter",
    "Q9. Умеренно широкий результат":
        "SELECT customer, product, SUM(quantity) AS v FROM sales "
        "GROUP BY customer, product",
    "Q10. Широкий результат: без свёртки":
        "SELECT customer, product, date, SUM(quantity) AS v FROM sales "
        "GROUP BY customer, product, date",
    "Q11. COUNT DISTINCT по измерению":
        "SELECT customer, COUNT(DISTINCT product) AS v FROM sales GROUP BY customer",
}


def normalize(rows, keys_count: int, digits: int = 6) -> list[tuple]:
    """Приводит результат к сопоставимому виду: ключи как строки, значение округлено."""
    out = []
    for row in rows:
        key = tuple(str(v) for v in row[:keys_count])
        value = row[keys_count] if len(row) > keys_count else row[-1]
        out.append(key + (round(float(value), digits),))
    return sorted(out)
