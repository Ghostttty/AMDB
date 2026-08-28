# -*- coding: utf-8 -*-
"""Генератор синтетической звёздной схемы для демонстраций и бенчмарков.

    python examples/generate.py --out data --customers 100 --products 100 --days 100 --rows 500000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REGIONS = ["Смоленск", "Москва", "Санкт-Петербург", "Новосибирск"]
CATEGORIES = ["электроника", "одежда", "продукты", "книги", "мебель"]


def generate(out: Path, n_customers: int, n_products: int, n_days: int,
             n_rows: int, seed: int = 42, skew: float = 0.0) -> dict[str, Path]:
    """Создаёт sales.csv, customers.csv, products.csv, dates.csv.

    ``skew`` > 0 делает распределение продаж неравномерным (степенной закон),
    что ближе к реальным данным и заметно влияет на разреженность гиперкуба.
    """
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)

    def pick(n_values: int) -> np.ndarray:
        if skew <= 0:
            return rng.integers(0, n_values, n_rows)
        w = 1.0 / np.power(np.arange(1, n_values + 1), skew)
        return rng.choice(n_values, size=n_rows, p=w / w.sum())

    customer = pick(n_customers)
    product = pick(n_products)
    date = rng.integers(0, n_days, n_rows)
    quantity = rng.integers(1, 20, n_rows)

    sales = out / "sales.csv"
    with open(sales, "w", encoding="utf-8", newline="") as f:
        f.write("customer,product,date,quantity\n")
        np.savetxt(f, np.column_stack([customer, product, date, quantity]),
                   fmt="%d", delimiter=",")

    customers = out / "customers.csv"
    with open(customers, "w", encoding="utf-8", newline="") as f:
        f.write("customer,region\n")
        for i in range(n_customers):
            f.write(f"{i},{REGIONS[i % len(REGIONS)]}\n")

    products = out / "products.csv"
    prices = np.round(rng.uniform(10, 1000, n_products), 2)
    with open(products, "w", encoding="utf-8", newline="") as f:
        f.write("product,category,price\n")
        for i in range(n_products):
            f.write(f"{i},{CATEGORIES[i % len(CATEGORIES)]},{prices[i]}\n")

    dates = out / "dates.csv"
    with open(dates, "w", encoding="utf-8", newline="") as f:
        f.write("date,month,quarter\n")
        for d in range(n_days):
            f.write(f"{d},M{d // 30 + 1},Q{d // 90 + 1}\n")

    return {"sales": sales, "customers": customers, "products": products, "dates": dates}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data", type=Path)
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--products", type=int, default=100)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--rows", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skew", type=float, default=0.0,
                   help="показатель степенного закона (0 = равномерно)")
    a = p.parse_args()
    files = generate(a.out, a.customers, a.products, a.days, a.rows, a.seed, a.skew)
    total = a.customers * a.products * a.days
    print(f"сгенерировано {a.rows:,} строк фактов; гиперкуб "
          f"{a.customers}×{a.products}×{a.days} = {total:,} ячеек")
    for name, path in files.items():
        print(f"  {name}: {path} ({path.stat().st_size / 2**20:.1f} МиБ)")


if __name__ == "__main__":
    main()
