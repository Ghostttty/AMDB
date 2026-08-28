# -*- coding: utf-8 -*-
import numpy as np
import pytest

from amdb import Database

pd = pytest.importorskip("pandas")

NC, NP, ND = 12, 6, 12
REGIONS = ["Смоленск", "Москва", "Санкт-Петербург"]


@pytest.fixture(scope="session")
def frames():
    rng = np.random.default_rng(11)
    n = 20_000
    sales = pd.DataFrame({
        "customer": rng.integers(0, NC, n),
        "product": rng.integers(0, NP, n),
        "date": rng.integers(0, ND, n),
        "quantity": rng.random(n).round(3),
    })
    products = pd.DataFrame({
        "product": range(NP),
        "category": ["A" if i % 2 == 0 else "B" for i in range(NP)],
        "price": (rng.random(NP) * 10).round(2),
    })
    customers = pd.DataFrame({
        "customer": range(NC),
        "region": [REGIONS[i % len(REGIONS)] for i in range(NC)],
    })
    return sales, products, customers


@pytest.fixture(scope="session")
def enriched(frames):
    """Тот же факт в плоском виде — эталон для дифференциальных тестов."""
    sales, products, customers = frames
    df = sales.copy()
    df["price"] = df["product"].map(dict(zip(products["product"], products["price"])))
    df["revenue"] = df.quantity * df.price
    df["region"] = df["customer"].map(dict(zip(customers["customer"], customers["region"])))
    df["category"] = df["product"].map(dict(zip(products["product"], products["category"])))
    df["month"] = ["Q%d" % (d // 3 + 1) for d in df["date"]]
    return df


@pytest.fixture(scope="session")
def db(frames):
    sales, products, customers = frames
    d = Database()
    d.load_frame(sales, ["customer", "product", "date"], "quantity", "sales",
                 ordered_dims=["date"])
    d.load_dimension(products, "product", attributes=["category"], measures=["price"])
    d.load_dimension(customers, "customer", attributes=["region"])
    d.add_hierarchy("date", "month", {i: "Q%d" % (i // 3 + 1) for i in range(ND)})
    return d


@pytest.fixture(scope="session")
def cells(enriched):
    """Гранулярность ячейки гиперкуба — эталон для MIN/MAX."""
    return enriched.groupby(["customer", "product", "date"], as_index=False).quantity.sum()
