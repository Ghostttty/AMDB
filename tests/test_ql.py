# -*- coding: utf-8 -*-
"""Транслятор: лексер, парсер, связывание, планирование, оптимизация."""
import numpy as np
import pytest

from amdb.core import TooManyIndicesError
from amdb.ql import BindError, QuerySyntaxError, bind, normalize, parse, plan_query
from amdb.ql.ast import Aggregate, BinOp, Column, Literal
from amdb.ql.optimizer import decompose, needs_decomposition, optimize_step
from amdb.ql.planner import ARRAY, EinsumStep, Operand


# --- разбор ----------------------------------------------------------------
def test_parse_full_query():
    q = parse("""
        SELECT customer, month, SUM(quantity * price) AS revenue
        FROM sales JOIN products ON sales.product = products.product
        WHERE region = 'Смоленск' AND date BETWEEN 1 AND 10
        GROUP BY customer, month
        HAVING revenue > 5
        ORDER BY revenue DESC
        LIMIT 10
    """)
    assert q.sources == ("sales", "products")
    assert q.group_by == ("customer", "month")
    assert q.limit == 10
    assert q.order_by[0].descending
    assert len(q.aggregates()) == 1


def test_parse_cyrillic_identifiers():
    q = parse("SELECT клиент, SUM(количество) AS итог FROM продажи GROUP BY клиент")
    assert q.group_by == ("клиент",)
    assert q.items[1].alias == "итог"


def test_parse_comments_and_semicolon():
    q = parse("SELECT SUM(q) FROM f -- считаем сумму\n;")
    assert q.source == "f"


def test_parse_window():
    q = parse("SELECT c, SUM(v) OVER (PARTITION BY c ORDER BY m "
              "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) FROM f GROUP BY c, m")
    w = q.aggregates()[0].window
    assert w.partition_by == ("c",) and w.order_by == "m" and w.frame_preceding == 3


def test_parse_subquery():
    q = parse("SELECT SUM(v) FROM f WHERE c IN (SELECT c FROM d WHERE r = 'x')")
    assert q.where.subquery.source == "d"


@pytest.mark.parametrize("sql", [
    "SELECT FROM",
    "SELECT SUM(v FROM f",
    "SELECT v FROM f WHERE c === 1",
    "SELECT v FROM f WHERE c = 'незакрытая",
    "SELECT v FROM f #",
])
def test_syntax_errors_point_at_position(sql):
    with pytest.raises(QuerySyntaxError):
        parse(sql)


# --- нормализация выражений ------------------------------------------------
def test_normalize_expands_products_and_sums():
    terms = normalize(parse("SELECT SUM(a * b - 2 * c) FROM f").aggregates()[0].arg)
    got = sorted((t.coef, t.measures) for t in terms)
    assert got == [(-2.0, ("c",)), (1.0, ("a", "b"))]


def test_normalize_divides_by_constant():
    terms = normalize(parse("SELECT SUM(a / 4) FROM f").aggregates()[0].arg)
    assert terms[0].coef == 0.25 and terms[0].measures == ("a",)


def test_normalize_rejects_division_by_measure():
    with pytest.raises(BindError, match="деление на меру"):
        normalize(parse("SELECT SUM(a / b) FROM f").aggregates()[0].arg)


# --- связывание ------------------------------------------------------------
def test_bind_requires_select_columns_in_group_by(db):
    with pytest.raises(BindError, match="отсутствует в GROUP BY"):
        db.compile("SELECT customer, product, SUM(quantity) FROM sales GROUP BY customer")


def test_bind_reports_unknown_column(db):
    with pytest.raises(BindError, match="неизвестный столбец"):
        db.compile("SELECT SUM(quantity) FROM sales WHERE отсутствует = 1")


def test_bind_reports_unknown_source(db):
    with pytest.raises(BindError, match="источник"):
        db.compile("SELECT SUM(quantity) FROM неведомый GROUP BY customer")


def test_bind_rejects_group_by_attribute(db):
    with pytest.raises(BindError, match="создайте иерархию"):
        db.compile("SELECT region, SUM(quantity) FROM sales GROUP BY region")


def test_bind_rejects_unreachable_group_axis(db):
    with pytest.raises(BindError, match="GROUP BY"):
        db.compile("SELECT SUM(price) FROM product GROUP BY customer")


def test_comparing_text_attribute_with_number_gives_clear_error(db):
    with pytest.raises(BindError, match="неприменимо к атрибуту"):
        db.compile("SELECT SUM(quantity) FROM sales WHERE region > 5")


def test_between_on_text_attribute_gives_clear_error(db):
    with pytest.raises(BindError, match="неприменимо к атрибуту"):
        db.compile("SELECT SUM(quantity) FROM sales WHERE category BETWEEN 1 AND 2")


def test_count_distinct_binds_to_its_own_plan(db):
    """COUNT DISTINCT выразим над булевым полукольцом и имеет отдельный шаг плана."""
    plan = db.compile("SELECT customer, COUNT(DISTINCT product) AS n FROM sales "
                      "GROUP BY customer", use_cache=False)
    agg = plan.aggregates[0]
    assert agg.func == "COUNT_DISTINCT"
    assert agg.distinct is not None and agg.distinct.distinct_axis == "product"
    assert agg.terms == [], "у COUNT DISTINCT нет einsum-слагаемых над (+,·)"


def test_distinct_rejected_outside_count(db):
    with pytest.raises(BindError, match="только в COUNT"):
        db.compile("SELECT SUM(DISTINCT quantity) FROM sales")


def test_bind_rejects_minmax_with_arithmetic(db):
    with pytest.raises(BindError, match="редукция, а не свёртка"):
        db.compile("SELECT customer, MAX(quantity * price) FROM sales "
                   "JOIN product ON sales.product = product.product GROUP BY customer")


def test_join_on_different_dimensions_is_rejected(db):
    with pytest.raises(BindError, match="соединяются по имени"):
        db.compile("SELECT customer, SUM(quantity) FROM sales "
                   "JOIN product ON sales.customer = product.product GROUP BY customer")


# --- планирование ----------------------------------------------------------
def test_plan_produces_expected_einsum(db):
    specs = db.einsum_of("SELECT customer, month, SUM(quantity * price) AS rev "
                         "FROM sales JOIN product ON sales.product = product.product "
                         "GROUP BY customer, month")
    assert specs == ["abc,b,cd->ad"]


def test_filter_becomes_extra_operand(db):
    """Фильтр — операнд свёртки; «висячие» оси куба сворачиваются заранее.

    Спецификация 'a,a->a', а не 'abc,a->a', потому что оси product и date не
    встречаются ни у маски, ни в результате, и оптимизатор сворачивает их до
    произведения (см. push_projections).
    """
    plan = db.compile("SELECT customer, SUM(quantity) FROM sales "
                      "WHERE region = 'Москва' GROUP BY customer", use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    assert step.spec == "a,a->a"
    assert len(step.operands) == 2, "маска должна остаться отдельным операндом"
    cube = next(o for o in step.operands if o.name == "sales")
    assert cube.presum == ("product", "date")


def test_lambda_mu_roles_are_reported(db):
    plan = db.compile("SELECT product, SUM(quantity * price) FROM sales "
                      "JOIN product ON sales.product = product.product GROUP BY product",
                      use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    assert step.lam_mu == (1, 0)   # product общий и остаётся -> λ=1


def test_plan_cache_is_reused(db):
    sql = "SELECT customer, SUM(quantity) FROM sales GROUP BY customer"
    db.compile(sql)
    before = db.plan_cache.hits
    db.compile(sql)
    assert db.plan_cache.hits == before + 1


def test_explain_mentions_convolution(db):
    text = db.explain("SELECT customer, SUM(quantity) FROM sales GROUP BY customer")
    assert "einsum(" in text and "свёртка" in text


# --- декомпозиция при > 52 индексах ----------------------------------------
def _wide_step(n_operands=60):
    rng = np.random.default_rng(0)
    mats = [rng.random((2, 3)) for _ in range(n_operands)]
    ops = [Operand(ARRAY, f"c{i}", (f"x{i}", "k"), mats[i]) for i in range(n_operands)]
    step = EinsumStep.__new__(EinsumStep)
    step.operands, step.output, step.spec, step.path, step.chain = ops, ("x0",), "", None, None
    return step, mats


def test_direct_plan_over_52_indices_is_rejected():
    with pytest.raises(TooManyIndicesError):
        from amdb.core import build_spec
        build_spec([(f"x{i}", "k") for i in range(60)], ("x0",))


def test_decomposition_is_numerically_correct():
    step, mats = _wide_step()
    assert needs_decomposition(step)
    chain = decompose(step, None)
    assert len(chain) == len(mats) - 1
    cur = None
    for sub in chain:
        arrays = [(cur if o.name.startswith("__tmp") else o.array) for o in sub.operands]
        cur = np.einsum(sub.spec, *arrays)
    acc = np.ones(3)
    for m in mats[1:]:
        acc = acc * m.sum(axis=0)
    assert np.allclose(cur, mats[0] @ acc)


def test_optimizer_estimates_cost(db):
    plan = db.compile("SELECT customer, month, SUM(quantity * price) FROM sales "
                      "JOIN product ON sales.product = product.product "
                      "GROUP BY customer, month", use_cache=False)
    cost = optimize_step(plan.aggregates[0].terms[0][1], db.catalog)
    assert cost.flops > 0
    assert cost.largest_intermediate > 0


def test_optimizer_refuses_oversized_intermediate(db):
    plan = db.compile("SELECT customer, product, SUM(quantity) FROM sales "
                      "GROUP BY customer, product", use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    with pytest.raises(MemoryError, match="бюджет"):
        optimize_step(step, db.catalog, max_intermediate_bytes=1)
