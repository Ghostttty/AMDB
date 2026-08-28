# -*- coding: utf-8 -*-
"""Дифференциальные тесты: AMDB против pandas на одних и тех же данных.

pandas здесь играет роль оракула (в CI той же цели служит DuckDB, см. bench/).
Сравнение ведётся по относительной погрешности: обе стороны считают в float64,
поэтому расхождение должно быть на уровне машинной точности.
"""
import numpy as np
import pytest

RTOL = 1e-12


def assert_matches(result, expected, keys, value_col, exp_col=None):
    got = result.to_pandas().sort_values(keys).reset_index(drop=True)
    exp = expected.sort_values(keys).reset_index(drop=True)
    exp_col = exp_col or exp.columns[-1]
    assert len(got) == len(exp), f"строк {len(got)}, ожидалось {len(exp)}"
    for k in keys:
        assert (got[k].astype(str).to_numpy() == exp[k].astype(str).to_numpy()).all()
    a = got[value_col].to_numpy(float)
    b = exp[exp_col].to_numpy(float)
    rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
    assert rel.max() <= RTOL, f"относительная погрешность {rel.max():.2e}"


def test_sum_one_dimension(db, enriched):
    assert_matches(
        db.sql("SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer"),
        enriched.groupby("customer", as_index=False).quantity.sum(), ["customer"], "q")


def test_join_with_dimension_measure(db, enriched):
    """SUM(quantity * price) — меры из разных кубов в одной свёртке."""
    assert_matches(
        db.sql("SELECT customer, SUM(quantity * price) AS rev FROM sales "
               "JOIN product ON sales.product = product.product GROUP BY customer"),
        enriched.groupby("customer", as_index=False).revenue.sum(), ["customer"], "rev")


def test_hierarchy_rollup(db, enriched):
    assert_matches(
        db.sql("SELECT month, SUM(quantity) AS q FROM sales GROUP BY month"),
        enriched.groupby("month", as_index=False).quantity.sum(), ["month"], "q")


def test_two_group_axes_with_join_and_rollup(db, enriched):
    """Пример §6.1 техпроекта целиком."""
    assert_matches(
        db.sql("SELECT customer, month, SUM(quantity * price) AS rev FROM sales "
               "JOIN product ON sales.product = product.product "
               "GROUP BY customer, month"),
        enriched.groupby(["customer", "month"], as_index=False).revenue.sum(),
        ["customer", "month"], "rev")


def test_filter_by_attribute(db, enriched):
    assert_matches(
        db.sql("SELECT customer, SUM(quantity) AS q FROM sales "
               "WHERE region = 'Смоленск' GROUP BY customer"),
        enriched[enriched.region == "Смоленск"]
        .groupby("customer", as_index=False).quantity.sum(), ["customer"], "q")


def test_subquery_filter(db, enriched):
    """Пример §6.2: WHERE customer IN (SELECT ... WHERE region = ...)."""
    assert_matches(
        db.sql("SELECT month, SUM(quantity) AS q FROM sales WHERE customer IN "
               "(SELECT customer FROM customers WHERE region = 'Смоленск') "
               "GROUP BY month"),
        enriched[enriched.region == "Смоленск"]
        .groupby("month", as_index=False).quantity.sum(), ["month"], "q")


def test_between_filter(db, enriched):
    assert_matches(
        db.sql("SELECT customer, SUM(quantity) AS q FROM sales "
               "WHERE date BETWEEN 2 AND 5 GROUP BY customer"),
        enriched[(enriched.date >= 2) & (enriched.date <= 5)]
        .groupby("customer", as_index=False).quantity.sum(), ["customer"], "q")


def test_in_list_and_comparison(db, enriched):
    assert_matches(
        db.sql("SELECT customer, SUM(quantity) AS q FROM sales "
               "WHERE product IN (1, 2) AND date >= 6 GROUP BY customer"),
        enriched[enriched["product"].isin([1, 2]) & (enriched.date >= 6)]
        .groupby("customer", as_index=False).quantity.sum(), ["customer"], "q")


def test_or_across_different_axes(db, enriched):
    """Дизъюнкция по разным осям требует объединённой маски-тензора."""
    assert_matches(
        db.sql("SELECT customer, SUM(quantity) AS q FROM sales "
               "WHERE product = 1 OR date = 3 GROUP BY customer"),
        enriched[(enriched["product"] == 1) | (enriched.date == 3)]
        .groupby("customer", as_index=False).quantity.sum(), ["customer"], "q")


def test_not_in(db, enriched):
    assert_matches(
        db.sql("SELECT customer, SUM(quantity) AS q FROM sales "
               "WHERE product NOT IN (0, 1) GROUP BY customer"),
        enriched[~enriched["product"].isin([0, 1])]
        .groupby("customer", as_index=False).quantity.sum(), ["customer"], "q")


def test_count_is_exact_row_count(db, enriched):
    """Спутниковый счётный куб даёт точный COUNT(*), а не число ячеек."""
    exp = enriched.groupby("customer", as_index=False).quantity.count()
    assert_matches(db.sql("SELECT customer, COUNT(*) AS n FROM sales GROUP BY customer"),
                   exp, ["customer"], "n")


def test_avg(db, enriched):
    assert_matches(db.sql("SELECT customer, AVG(quantity) AS a FROM sales GROUP BY customer"),
                   enriched.groupby("customer", as_index=False).quantity.mean(),
                   ["customer"], "a")


def test_max_is_over_cube_cells(db, cells):
    """MIN/MAX работают на гранулярности ячейки гиперкуба, а не исходной строки."""
    assert_matches(db.sql("SELECT product, MAX(quantity) AS m FROM sales GROUP BY product"),
                   cells.groupby("product", as_index=False).quantity.max(),
                   ["product"], "m")


def test_min_with_filter(db, cells):
    assert_matches(
        db.sql("SELECT product, MIN(quantity) AS m FROM sales WHERE date >= 6 "
               "GROUP BY product"),
        cells[cells.date >= 6].groupby("product", as_index=False).quantity.min(),
        ["product"], "m")


def test_ratio_of_aggregates(db, enriched):
    exp = enriched.groupby("customer", as_index=False).agg(
        r=("revenue", "sum"), q=("quantity", "sum"))
    exp["ratio"] = exp.r / exp.q
    assert_matches(
        db.sql("SELECT customer, SUM(quantity * price) / SUM(quantity) AS ratio "
               "FROM sales JOIN product ON sales.product = product.product "
               "GROUP BY customer"),
        exp[["customer", "ratio"]], ["customer"], "ratio")


def test_linear_expression_under_aggregate(db, enriched):
    """SUM(a*b - c) раскладывается в две свёртки, результаты складываются."""
    exp = enriched.assign(net=enriched.revenue - enriched.quantity)
    assert_matches(
        db.sql("SELECT customer, SUM(quantity * price - quantity) AS net FROM sales "
               "JOIN product ON sales.product = product.product GROUP BY customer"),
        exp.groupby("customer", as_index=False).net.sum(), ["customer"], "net")


def test_having_order_limit(db, enriched):
    full = enriched.groupby("customer", as_index=False).quantity.sum()
    exp = full[full.quantity > 800].sort_values("quantity", ascending=False).head(3)
    result = db.sql("SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer "
                    "HAVING q > 800 ORDER BY q DESC LIMIT 3")
    assert len(result) == len(exp)
    assert result.column("customer") == exp["customer"].tolist()


def test_scalar_aggregate_without_group_by(db, enriched):
    result = db.sql("SELECT SUM(quantity) AS total FROM sales")
    assert len(result) == 1
    assert abs(result[0][0] - enriched.quantity.sum()) / enriched.quantity.sum() < RTOL


def test_window_running_sum(db, enriched):
    result = db.sql("SELECT customer, month, SUM(quantity) OVER "
                    "(PARTITION BY customer ORDER BY month) AS running "
                    "FROM sales GROUP BY customer, month")
    expected = (enriched.groupby(["customer", "month"]).quantity.sum()
                .unstack().cumsum(axis=1))
    got = result.to_pandas().pivot(index="customer", columns="month", values="running")
    assert np.abs(got.to_numpy(float) - expected.to_numpy(float)).max() < 1e-9


def test_empty_groups_are_dropped(db):
    """Группа без исходных записей не должна попадать в результат (семантика SQL)."""
    result = db.sql("SELECT customer, SUM(quantity) AS q FROM sales "
                    "WHERE customer IN (0, 1) GROUP BY customer")
    assert sorted(result.column("customer")) == [0, 1]


def test_multiple_aggregates_in_one_query(db, enriched):
    result = db.sql("SELECT customer, SUM(quantity) AS q, COUNT(*) AS n, "
                    "AVG(quantity) AS a FROM sales GROUP BY customer")
    exp = enriched.groupby("customer").quantity.agg(["sum", "count", "mean"])
    got = result.to_pandas().set_index("customer")
    assert np.allclose(got["q"], exp["sum"].reindex(got.index))
    assert np.allclose(got["n"], exp["count"].reindex(got.index))
    assert np.allclose(got["a"], exp["mean"].reindex(got.index))


def test_count_distinct_over_dimension(db, enriched):
    """COUNT DISTINCT выразим над булевым полукольцом: ∨-свёртка, затем подсчёт."""
    assert_matches(
        db.sql("SELECT customer, COUNT(DISTINCT product) AS n FROM sales "
               "GROUP BY customer"),
        enriched.groupby("customer", as_index=False)["product"].nunique(),
        ["customer"], "n")


def test_count_distinct_with_filter(db, enriched):
    assert_matches(
        db.sql("SELECT customer, COUNT(DISTINCT date) AS d FROM sales "
               "WHERE product IN (1, 2) GROUP BY customer"),
        enriched[enriched["product"].isin([1, 2])]
        .groupby("customer", as_index=False)["date"].nunique(),
        ["customer"], "d")


def test_count_distinct_rejects_expression(db):
    """DISTINCT по мере потребовал бы вынести её область значений в отдельную ось."""
    from amdb.ql import BindError

    with pytest.raises(BindError, match="только для измерения"):
        db.compile("SELECT COUNT(DISTINCT quantity * 2) FROM sales")


def test_count_distinct_rejects_grouping_axis(db):
    from amdb.ql import BindError

    with pytest.raises(BindError, match="одновременно в GROUP BY"):
        db.compile("SELECT customer, COUNT(DISTINCT customer) FROM sales "
                   "GROUP BY customer")


def test_sparse_cube_query_matches_dense():
    """Разреженный путь свёртки даёт тот же результат, что и плотный."""
    pd = pytest.importorskip("pandas")
    from amdb import Database
    from amdb.storage import DENSE, SPARSE_COO

    rng = np.random.default_rng(4)
    n = 20_000
    df = pd.DataFrame({"a": rng.integers(0, 150, n), "b": rng.integers(0, 150, n),
                       "c": rng.integers(0, 150, n), "v": rng.random(n)})
    sparse_db = Database()
    sparse_db.load_frame(df, ["a", "b", "c"], "v", "f", layout=SPARSE_COO)
    dense_db = Database()
    dense_db.load_frame(df, ["a", "b", "c"], "v", "f", layout=DENSE)
    assert sparse_db.cube("f").is_sparse and not dense_db.cube("f").is_sparse

    sql = "SELECT a, b, SUM(v) AS s FROM f WHERE c >= 50 GROUP BY a, b"
    s = sparse_db.sql(sql).to_pandas().sort_values(["a", "b"]).reset_index(drop=True)
    d = dense_db.sql(sql).to_pandas().sort_values(["a", "b"]).reset_index(drop=True)
    assert len(s) == len(d)
    assert np.abs(s["s"].to_numpy() - d["s"].to_numpy()).max() < 1e-10


def test_group_by_dimension_of_a_joined_fact_cube():
    """Регрессия: группировка по оси второго факта отвергалась.

    Соединение двух фактов с общим измерением, остающимся в группировке, —
    это (1, 0)-свёрнутое произведение: свёртки нет, ранг результата больше
    ранга каждого операнда. Шаг агрегата собирался верно, но служебный шаг
    наличия групп строился по одному лишь основному кубу, у которого оси
    второго факта нет, и запрос падал с BindError.
    """
    pd = pytest.importorskip("pandas")
    from amdb import Database

    rng = np.random.default_rng(0)
    left = pd.DataFrame({"customer": rng.integers(0, 8, 500),
                         "product": rng.integers(0, 6, 500),
                         "q": rng.random(500)})
    right = pd.DataFrame({"product": rng.integers(0, 6, 500),
                          "date": rng.integers(0, 5, 500),
                          "w": rng.random(500)})
    db = Database()
    db.load_frame(left, ["customer", "product"], "q", "f1")
    db.load_frame(right, ["product", "date"], "w", "f2")

    query = ("SELECT customer, product, date, SUM(f1.q * f2.w) AS v "
             "FROM f1 JOIN f2 GROUP BY customer, product, date")
    plan = db.explain(query)
    assert "(1,0)-свёртка" in plan, "соединение без агрегации по общей оси есть (1,0)-произведение"

    result = db.sql(query)
    a = db.cube("f1").matrix.data
    b = db.cube("f2").matrix.data
    reference = np.einsum("ab,bc->abc", a, b)
    got = np.zeros_like(reference)
    for customer, product, date, value in result.rows:
        got[customer, product, date] = value
    assert np.allclose(got, reference)
    assert reference.ndim > max(a.ndim, b.ndim), "ранг результата должен вырасти"


def test_count_star_over_a_join_of_two_facts_counts_pairs():
    """COUNT(*) по соединению двух фактов — число пар, как и в SQL."""
    pd = pytest.importorskip("pandas")
    from amdb import Database

    left = pd.DataFrame({"a": [0, 0, 1], "b": [0, 1, 1], "q": [1.0, 1.0, 1.0]})
    right = pd.DataFrame({"b": [0, 1, 1], "c": [0, 0, 1], "w": [1.0, 1.0, 1.0]})
    db = Database()
    db.load_frame(left, ["a", "b"], "q", "l")
    db.load_frame(right, ["b", "c"], "w", "r")
    rows = db.sql("SELECT a, b, c, COUNT(*) AS n FROM l JOIN r GROUP BY a, b, c").rows
    got = {(a, b, c): n for a, b, c, n in rows}
    # (0,0)×(0,0): одна строка слева, одна справа -> одна пара.
    assert got[(0, 0, 0)] == 1
    # (0,1)×(1,0) и (0,1)×(1,1): по одной паре каждая.
    assert got[(0, 1, 0)] == 1 and got[(0, 1, 1)] == 1
    assert got[(1, 1, 0)] == 1 and got[(1, 1, 1)] == 1


def test_avg_over_a_join_divides_by_the_number_of_pairs():
    """Знаменатель AVG по соединению — число пар, а не строк одного факта.

    Раньше знаменатель строился по кубу первой меры терма, поэтому AVG по
    соединению двух фактов либо падал (если группировка задевала ось второго
    факта), либо делил на не то число.
    """
    pd = pytest.importorskip("pandas")
    from amdb import Database

    left = pd.DataFrame({"a": [0, 0], "b": [0, 0], "q": [2.0, 4.0]})
    right = pd.DataFrame({"b": [0, 0], "c": [0, 0], "w": [1.0, 3.0]})
    db = Database()
    db.load_frame(left, ["a", "b"], "q", "l")
    db.load_frame(right, ["b", "c"], "w", "r")

    # Куб слева: сумма 6 при двух строках; справа: сумма 4 при двух строках.
    # Произведение сумм 24, число пар 2 * 2 = 4, среднее 6.
    rows = db.sql("SELECT a, b, c, AVG(l.q * r.w) AS v FROM l JOIN r "
                  "GROUP BY a, b, c").rows
    assert len(rows) == 1
    assert rows[0][3] == pytest.approx(24.0 / 4.0)

    total = db.sql("SELECT a, b, c, SUM(l.q * r.w) AS v FROM l JOIN r "
                   "GROUP BY a, b, c").rows[0][3]
    pairs = db.sql("SELECT a, b, c, COUNT(*) AS n FROM l JOIN r "
                   "GROUP BY a, b, c").rows[0][3]
    assert total == pytest.approx(24.0) and pairs == pytest.approx(4.0)


def test_avg_of_a_squared_measure_does_not_double_count_the_denominator():
    """AVG(q * q) делит на число строк, а не на его квадрат."""
    pd = pytest.importorskip("pandas")
    from amdb import Database

    frame = pd.DataFrame({"a": [0, 0], "q": [2.0, 4.0]})
    db = Database()
    db.load_frame(frame, ["a"], "q", "f")
    # Ячейка содержит сумму 6 при двух строках: AVG(q*q) = 36 / 2 = 18.
    rows = db.sql("SELECT a, AVG(f.q * f.q) AS v FROM f GROUP BY a").rows
    assert rows[0][1] == pytest.approx(18.0)


@pytest.mark.parametrize("layout", ["dense", "sparse_coo"])
def test_scalar_aggregate_without_group_by_on_both_layouts(layout):
    """Регрессия: SUM без GROUP BY падал на разреженном кубе.

    Оба представления обязаны давать один и тот же ответ: выбор представления
    есть решение о хранении, а не о семантике.
    """
    pd = pytest.importorskip("pandas")
    from amdb import Database

    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame({"customer": rng.integers(0, 80, n),
                          "product": rng.integers(0, 80, n),
                          "date": rng.integers(0, 80, n),
                          "quantity": rng.random(n)})
    db = Database()
    db.load_frame(frame, ["customer", "product", "date"], "quantity", "sales",
                  layout=layout)
    assert db.cube("sales").is_sparse == (layout == "sparse_coo")

    assert db.sql("SELECT SUM(quantity) AS q FROM sales").rows[0][0] == \
        pytest.approx(float(frame.quantity.sum()))
    assert db.sql("SELECT COUNT(*) AS n FROM sales").rows[0][0] == pytest.approx(n)
    assert db.sql("SELECT AVG(quantity) AS a FROM sales").rows[0][0] == \
        pytest.approx(float(frame.quantity.mean()))
