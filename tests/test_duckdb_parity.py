# -*- coding: utf-8 -*-
"""Сверка AMDB с DuckDB как с оракулом.

Здесь DuckDB играет ту же роль, что pandas в ``test_queries.py``, но это
проверка сильнее: колоночная СУБД реализует полноценную семантику SQL, и
совпадение с ней говорит о корректности трансляции больше, чем совпадение
с построчной библиотекой.

Запросы берутся из ``bench/workload.py`` — того же набора, на котором ведётся
замер. Это намеренно: замеряется ровно то, что проверено на правильность.

Тест пропускается, если DuckDB не установлен: ``pip install duckdb``.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

duckdb = pytest.importorskip("duckdb", reason="нужен DuckDB: pip install duckdb")
pytest.importorskip("pandas")

from bench.workload import QUERIES, build_amdb, build_duckdb, make_workload, normalize


@pytest.fixture(scope="module")
def pair():
    """Одна и та же база в обеих системах. Небольшая — тест не про скорость."""
    work = make_workload(rows=20_000, customers=15, products=12, days=20, seed=7)
    return build_amdb(work), build_duckdb(work), work


@pytest.mark.parametrize("query", QUERIES, ids=lambda q: q.name.split(".")[0])
def test_matches_duckdb(pair, query):
    """Результат AMDB совпадает с DuckDB на том же наборе данных."""
    db, con, _ = pair

    got = normalize([tuple(r) for r in db.sql(query.amdb).rows], len(query.keys))
    ref = normalize(con.execute(query.duck).fetchall(), len(query.keys))

    assert len(got) == len(ref), (
        f"{query.name}: строк {len(got)}, у DuckDB {len(ref)}"
    )
    for g, r in zip(got, ref):
        assert g[:-1] == r[:-1], f"{query.name}: разошлись ключи {g[:-1]} и {r[:-1]}"
        assert abs(g[-1] - r[-1]) <= query.tolerance * max(abs(r[-1]), 1.0), (
            f"{query.name}: при ключе {g[:-1]} получено {g[-1]}, "
            f"у DuckDB {r[-1]}"
        )


def test_query_set_covers_both_strengths_and_weaknesses():
    """Набор обязан включать и заведомо невыгодный для AMDB случай.

    Без такого запроса стенд превратился бы в рекламу: замер, где выбраны
    только удобные случаи, ничего не доказывает.
    """
    wide = [q for q in QUERIES if len(q.keys) >= 3]
    assert wide, "в наборе нет запроса с широкой выдачей"
    assert any(q.expect for q in QUERIES), "не указаны ожидания по запросам"


def test_sparse_layout_also_matches_duckdb():
    """Разреженный путь свёртки сверяется с тем же оракулом."""
    from amdb.storage import SPARSE_COO

    work = make_workload(rows=8_000, customers=40, products=40, days=40, seed=3)
    db = build_amdb(work, layout=SPARSE_COO)
    con = build_duckdb(work)
    assert db.cube("sales").is_sparse

    sql_amdb = ("SELECT customer, SUM(quantity) AS v FROM sales "
                "WHERE date >= 20 GROUP BY customer")
    sql_duck = ("SELECT customer, SUM(quantity) AS v FROM sales "
                "WHERE date >= 20 GROUP BY customer")
    got = normalize([tuple(r) for r in db.sql(sql_amdb).rows], 1)
    ref = normalize(con.execute(sql_duck).fetchall(), 1)
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g[0] == r[0]
        assert abs(g[1] - r[1]) <= 1e-9 * max(abs(r[1]), 1.0)


def test_skewed_data_also_matches(pair):
    """Перекошенное распределение меняет разреженность куба, но не ответы."""
    work = make_workload(rows=20_000, customers=15, products=12, days=20,
                         seed=11, skew=1.3)
    db = build_amdb(work)
    con = build_duckdb(work)
    for query in QUERIES[:6]:
        got = normalize([tuple(r) for r in db.sql(query.amdb).rows], len(query.keys))
        ref = normalize(con.execute(query.duck).fetchall(), len(query.keys))
        assert len(got) == len(ref), query.name
        for g, r in zip(got, ref):
            assert abs(g[-1] - r[-1]) <= query.tolerance * max(abs(r[-1]), 1.0), (
                f"{query.name}: {g} против {r}"
            )
