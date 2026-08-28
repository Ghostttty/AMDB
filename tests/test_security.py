# -*- coding: utf-8 -*-
"""Разграничение доступа на уровне измерений."""
import numpy as np
import pytest

from amdb import Database
from amdb.security import AccessDenied, cell_suppression


@pytest.fixture
def secured(frames):
    sales, products, customers = frames
    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    db.load_dimension(products, "product", attributes=["category"], measures=["price"])
    db.load_dimension(customers, "customer", attributes=["region"])
    db.grant("smolensk", "customer", allowed=[0, 3, 6, 9])
    db.grant("no_product_rollup", "product", can_project=False)
    db.grant("strict", "customer", allowed=[0], permissive=False)
    return db


def test_role_sees_only_allowed_members(secured, enriched):
    result = secured.sql("SELECT customer, SUM(quantity) AS q FROM sales "
                         "GROUP BY customer", role="smolensk")
    assert sorted(result.column("customer")) == [0, 3, 6, 9]
    expected = enriched[enriched.customer.isin([0, 3, 6, 9])] \
        .groupby("customer").quantity.sum()
    got = dict(zip(result.column("customer"), result.column("q")))
    for k, v in expected.items():
        assert abs(got[k] - v) / v < 1e-12


def test_totals_exclude_forbidden_rows(secured, enriched):
    """Запрещённые данные не участвуют в арифметике, а не отфильтровываются потом."""
    total = secured.sql("SELECT SUM(quantity) AS t FROM sales", role="smolensk")[0][0]
    expected = enriched[enriched.customer.isin([0, 3, 6, 9])].quantity.sum()
    assert abs(total - expected) / expected < 1e-12


def test_can_project_false_forbids_aggregating_over_axis(secured):
    with pytest.raises(AccessDenied, match="не может агрегировать"):
        secured.sql("SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer",
                    role="no_product_rollup")


def test_can_project_false_allows_query_with_axis_in_group_by(secured):
    result = secured.sql("SELECT customer, product, SUM(quantity) AS q FROM sales "
                         "GROUP BY customer, product", role="no_product_rollup")
    assert len(result) > 0


def test_non_permissive_role_denies_ungranted_dimension(secured):
    with pytest.raises(AccessDenied, match="не имеет доступа"):
        secured.sql("SELECT product, SUM(quantity) AS q FROM sales GROUP BY product",
                    role="strict")


def test_rls_applies_to_min_max_by_selection_not_masking(secured, frames):
    """Для MIN/MAX маска-умножение дала бы ложный ноль — применяется выборка."""
    sales, *_ = frames
    result = secured.sql("SELECT product, MIN(quantity) AS m FROM sales GROUP BY product",
                         role="smolensk")
    values = np.array(result.column("m"))
    assert (values > 0).all(), "MIN не должен превращаться в ноль от маскирования"


def test_unknown_role_is_rejected(secured):
    with pytest.raises(KeyError, match="роль"):
        secured.sql("SELECT SUM(quantity) FROM sales", role="нет_такой")


def test_cell_suppression():
    values = np.array([10.0, 20.0, 30.0])
    counts = np.array([1, 5, 100])
    out = cell_suppression(values, counts, k=5)
    assert np.isnan(out[0]) and out[1] == 20.0 and out[2] == 30.0


def test_role_survives_persistence(secured, tmp_path):
    secured.save(tmp_path / "sec")
    restored = Database.open(tmp_path / "sec")
    result = restored.sql("SELECT customer, SUM(quantity) AS q FROM sales "
                          "GROUP BY customer", role="smolensk")
    assert sorted(result.column("customer")) == [0, 3, 6, 9]
