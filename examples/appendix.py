# -*- coding: utf-8 -*-
"""Минимальный пример трансляции из приложения к статье.

Показывает весь путь от SQL-подобного запроса до элементарной операции ядра:
соединение со справочником, отбор по атрибуту, свёртка по иерархии и агрегация
собираются в одно (0, 2)-свёрнутое произведение четырёх операндов.

    python examples/appendix.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:                                  # запуск из копии репозитория, без установки
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb import Database

QUERY = (
    "SELECT customer, month, SUM(sales.quantity * product.price) AS revenue "
    "FROM sales JOIN product "
    "WHERE product.category = 'A' "
    "GROUP BY customer, month"
)


def main() -> int:
    rng = np.random.default_rng(0)
    n = 200_000
    sales = pd.DataFrame({
        "customer": rng.integers(0, 100, n),
        "product": rng.integers(0, 100, n),
        "date": rng.integers(0, 120, n),
        "quantity": rng.random(n) * 10,
    })
    products = pd.DataFrame({
        "product": range(100),
        "category": ["A" if i % 2 else "B" for i in range(100)],
        "price": rng.random(100) * 100,
    })

    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    db.load_dimension(products, "product",
                      attributes=["category"], measures=["price"])
    db.add_hierarchy("date", "month", {i: f"M{i // 10}" for i in range(120)})

    print(db.explain(QUERY))
    result = db.sql(QUERY)
    print(f"\nстрок в результате: {len(result.rows)}")
    print("первые три строки:")
    for row in result.rows[:3]:
        print(f"  клиент {row[0]}, месяц {row[1]}, выручка {row[2]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
