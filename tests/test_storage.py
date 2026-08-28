# -*- coding: utf-8 -*-
"""Хранилище: измерения, иерархии, загрузка, персистентность, политика."""
import numpy as np
import pytest

from amdb import Catalog, Database
from amdb.storage import (
    DENSE,
    SPARSE_COO,
    Dimension,
    Hierarchy,
    choose_layout,
    load_fact,
)

pd = pytest.importorskip("pandas")


def test_dimension_is_append_only():
    """Ординалы существующих значений не меняются при дозагрузке (риск R9)."""
    d = Dimension("customer", [10, 20, 30])
    before = {v: d.ordinal(v) for v in [10, 20, 30]}
    d.extend([5, 40, 20])
    assert all(d.ordinal(v) == o for v, o in before.items())
    assert d.ordinal(5) == 3 and d.ordinal(40) == 4
    assert len(d) == 5


def test_encode_uses_binary_search_and_matches_slow_path():
    """Быстрый путь кодирования обязан совпадать с пословным.

    Кодирование — 90 % времени загрузки крупного факта, поэтому оно
    векторизовано двоичным поиском. Ошибка здесь тихо перепутала бы данные
    местами, поэтому путь сверяется с медленным.
    """
    d = Dimension("customer", [10, 20, 30, 40])
    values = np.array([30, 10, 40, 20, 30])
    fast = d.encode(values)
    slow = np.fromiter((d.ordinal(v) for v in values), dtype=np.int64)
    assert np.array_equal(fast, slow)
    assert [d.label(o) for o in fast] == values.tolist()


def test_encode_works_for_text_dimensions():
    d = Dimension("region", ["Москва", "Смоленск", "Тверь"])
    got = d.encode(np.array(["Тверь", "Москва", "Тверь"]))
    assert [d.label(o) for o in got] == ["Тверь", "Москва", "Тверь"]


def test_encode_reports_unknown_value():
    d = Dimension("customer", [1, 2, 3])
    with pytest.raises(KeyError, match="customer"):
        d.encode(np.array([1, 99]))


def test_encode_stays_correct_after_appending():
    """Дозагрузка сбрасывает кэш поиска — иначе новые значения не нашлись бы."""
    d = Dimension("product", [5, 15])
    assert d.encode(np.array([15, 5])).tolist() == [1, 0]
    d.extend([1, 25])                      # 1 встаёт перед 5 по значению, но не по ординалу
    got = d.encode(np.array([1, 5, 15, 25]))
    assert [d.label(o) for o in got] == [1, 5, 15, 25]
    assert got.tolist() == [2, 0, 1, 3], "ординалы должны сохраниться, а не пересортироваться"


def test_dimension_masks():
    d = Dimension("month", [1, 2, 3, 4])
    assert np.allclose(d.mask_of([2, 4]), [0, 1, 0, 1])
    assert np.allclose(d.mask_range(2, 3), [0, 1, 1, 0])
    assert np.allclose(d.mask_range(low=3, inclusive=(False, True)), [0, 0, 0, 1])
    assert d.mask_all().sum() == 4


def test_range_on_categorical_dimension_is_rejected():
    d = Dimension("category", ["книги", "мебель"])
    with pytest.raises(ValueError, match="категориальное"):
        d.mask_range("а", "я")


def test_numeric_dimension_is_comparable_without_flag():
    assert Dimension("date", [1, 2, 3]).comparable
    assert not Dimension("region", ["Смоленск"]).comparable


def test_unknown_value_error_names_dimension():
    d = Dimension("customer", [1, 2])
    with pytest.raises(KeyError, match="customer"):
        d.ordinal(99)


def test_attributes_extend_with_dimension():
    d = Dimension("customer", [1, 2])
    d.set_attribute("region", ["Смоленск", "Москва"])
    d.extend([3])
    assert len(d.attributes["region"]) == 3
    assert d.attributes["region"][2] is None


def test_hierarchy_matrix():
    child = Dimension("date", list(range(6)))
    parent = Dimension("month", [])
    h = Hierarchy("date->month", child, parent, ["M1", "M1", "M1", "M2", "M2", "M2"])
    m = h.matrix()
    assert m.shape == (6, 2)
    assert np.allclose(m.sum(axis=0), [3, 3])


def test_hierarchy_rejects_wrong_length():
    child = Dimension("date", [0, 1, 2])
    with pytest.raises(ValueError):
        Hierarchy("bad", child, Dimension("m", []), ["M1"])


def test_hierarchy_path_composes():
    cat = Catalog()
    db = Database(cat)
    df = pd.DataFrame({"date": list(range(12)), "q": np.ones(12)})
    load_fact(cat, df, ["date"], "q", "f")
    db.add_hierarchy("date", "month", {i: f"M{i // 3}" for i in range(12)})
    db.add_hierarchy("month", "quarter", {f"M{i}": f"Q{i // 2}" for i in range(4)})
    m = cat.rollup_matrix("date", "quarter")
    assert m.shape == (12, 2)
    assert np.allclose(m.sum(axis=0), [6, 6])
    assert cat.rollup_matrix("date", "customer") is None


def test_choose_layout():
    assert choose_layout(nnz=900_000, shape=(100, 100, 100)) == DENSE
    assert choose_layout(nnz=1000, shape=(1000, 1000, 100)) == SPARSE_COO


def test_load_fact_aggregates_duplicates():
    cat = Catalog()
    df = pd.DataFrame({"a": [0, 0, 1], "b": [0, 0, 1], "v": [1.0, 2.0, 5.0]})
    cube = load_fact(cat, df, ["a", "b"], "v", "f")
    assert np.allclose(cube.matrix.data, [[3.0, 0.0], [0.0, 5.0]])
    assert np.allclose(cat.cube("f__count").matrix.data, [[2, 0], [0, 1]])


def test_load_fact_chooses_sparse_for_sparse_data():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame({"a": rng.integers(0, 500, n), "b": rng.integers(0, 500, n),
                       "c": rng.integers(0, 500, n), "v": rng.random(n)})
    cat = Catalog()
    cube = load_fact(cat, df, ["a", "b", "c"], "v", "f")
    assert cube.layout == SPARSE_COO
    assert cube.is_sparse


def test_catalog_rejects_axis_length_mismatch():
    from amdb.core import MultidimensionalMatrix
    from amdb.storage import Cube

    cat = Catalog()
    cat.ensure_dimension("a", [1, 2, 3])
    with pytest.raises(ValueError, match="мощность измерения"):
        cat.add_cube(Cube("x", "x", MultidimensionalMatrix(np.ones(5), ("a",))))


def test_catalog_roundtrip(tmp_path, frames):
    sales, products, customers = frames
    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    db.load_dimension(products, "product", attributes=["category"], measures=["price"])
    db.add_hierarchy("date", "month", {i: f"M{i // 3}" for i in range(12)})
    db.grant("analyst", "customer", allowed=[0, 1, 2])
    db.save(tmp_path / "base")

    restored = Database.open(tmp_path / "base")
    assert set(restored.cubes) == set(db.cubes)
    assert np.allclose(restored.cube("sales").matrix.data, db.cube("sales").matrix.data)
    assert restored.dimensions["product"].attributes["category"].tolist() == \
        db.dimensions["product"].attributes["category"].tolist()
    assert restored.catalog.rollup_matrix("date", "month").shape == (12, 4)
    assert restored.catalog.role("analyst").grants["customer"].allowed.sum() == 3

    a = db.sql("SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer")
    b = restored.sql("SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer")
    assert np.allclose(a.column("q"), b.column("q"))


def test_sparse_cube_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    n = 3000
    df = pd.DataFrame({"a": rng.integers(0, 400, n), "b": rng.integers(0, 400, n),
                       "c": rng.integers(0, 400, n), "v": rng.random(n)})
    db = Database()
    db.load_frame(df, ["a", "b", "c"], "v", "f")
    assert db.cube("f").is_sparse
    db.save(tmp_path / "sp")
    restored = Database.open(tmp_path / "sp")
    assert restored.cube("f").is_sparse
    assert np.allclose(np.sort(restored.cube("f").matrix.values),
                       np.sort(db.cube("f").matrix.values))


def test_cube_version_increments_on_reload(frames):
    sales, *_ = frames
    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    v1 = db.cube("sales").version
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    assert db.cube("sales").version == v1 + 1
