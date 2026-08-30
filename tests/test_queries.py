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


def test_query_execution_uses_the_batched_gemm_path():
    """Регрессия: исполнитель обходил ядро и терял разложение на сечения.

    `Executor` вызывал einsum напрямую, поэтому оптимизация §4.6 статьи
    применялась только при обращении к ядру и ни разу — при исполнении запроса.
    """
    from importlib import import_module

    pd = pytest.importorskip("pandas")
    from amdb import Database

    convolve_module = import_module("amdb.core.convolve")
    rng = np.random.default_rng(0)
    n, side = 20_000, 60
    sales = pd.DataFrame({"customer": rng.integers(0, side, n),
                          "product": rng.integers(0, side, n),
                          "date": rng.integers(0, side, n),
                          "q": rng.random(n)})
    price = pd.DataFrame([(p, d) for p in range(side) for d in range(side)],
                         columns=["product", "date"])
    price["w"] = rng.random(len(price))
    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "q", "sales")
    db.load_frame(price, ["product", "date"], "w", "price")

    query = ("SELECT customer, product, SUM(sales.q * price.w) AS v "
             "FROM sales JOIN price GROUP BY customer, product")
    assert "(1,1)-свёртка" in db.explain(query)

    calls = []
    original = convolve_module._batched_matmul

    def spy(a, b, lam, mu):
        calls.append((lam, mu))
        return original(a, b, lam, mu)

    convolve_module._batched_matmul = spy
    try:
        result = db.sql(query)
    finally:
        convolve_module._batched_matmul = original

    assert calls, "исполнитель не воспользовался пакетным gemm"
    assert all(lam >= 1 and mu >= 1 for lam, mu in calls)

    reference = np.einsum("abc,bc->ab", db.cube("sales").matrix.data,
                          db.cube("price").matrix.data)
    got = np.zeros_like(reference)
    for customer, product, value in result.rows:
        got[customer, product] = value
    assert np.allclose(got, reference)


def test_two_fact_queries_of_the_gpu_stand_match_duckdb():
    """Запросы части 3 стенда B14 — новые, и сверять их больше негде.

    Это произведение куба на куб: λ >= 1 при μ >= 1 либо μ = 0 с ростом ранга.
    Именно на нём проверяется гипотеза об ускорителе, поэтому ошибка в самих
    запросах обесценила бы весь замер.
    """
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pandas")
    from amdb import Database
    from bench.bench_gpu_case import TWO_FACT_QUERIES, build_two_fact

    side, rows = 12, 4_000
    sales, price = build_two_fact(side, rows, seed=3)
    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    db.load_frame(price, ["product", "date"], "price", "price")

    con = duckdb.connect()
    con.register("_s", sales)
    con.register("_p", price)
    con.execute("CREATE TABLE sales AS SELECT customer, product, date, "
                "SUM(quantity) AS quantity FROM _s GROUP BY 1, 2, 3")
    con.execute("CREATE TABLE price AS SELECT * FROM _p")

    for label, amdb_sql, duck_sql, keys in TWO_FACT_QUERIES:
        got = {tuple(r[:keys]): r[keys] for r in db.sql(amdb_sql).rows}
        expected = {tuple(r[:keys]): r[keys]
                    for r in con.execute(duck_sql).fetchall()}
        assert set(got) == set(expected), label
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, rel=1e-9), (label, key)


def test_two_fact_generator_covers_the_whole_price_grid():
    """Цены заданы на всей сетке (товар, дата) — иначе соединение теряло бы факты."""
    pytest.importorskip("pandas")
    from bench.bench_gpu_case import build_two_fact

    side = 8
    sales, price = build_two_fact(side, 500, seed=1)
    assert len(price) == side * side
    assert set(zip(price["product"], price["date"])) == {
        (p, d) for p in range(side) for d in range(side)}
    assert sales["customer"].max() < side


def _case_fixture():
    pd = pytest.importorskip("pandas")
    from amdb import Database

    rng = np.random.default_rng(0)
    n = 20_000
    sales = pd.DataFrame({"customer": rng.integers(0, 20, n),
                          "product": rng.integers(0, 20, n),
                          "date": rng.integers(0, 30, n),
                          "quantity": rng.random(n) * 10})
    products = pd.DataFrame({"product": range(20),
                             "price": rng.random(20) * 100,
                             "category": ["A" if i % 2 else "B" for i in range(20)]})
    db = Database()
    db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    db.load_dimension(products, "product", attributes=["category"], measures=["price"])
    return db, sales


@pytest.mark.parametrize("amdb_sql,pandas_expr", [
    ("SELECT customer, SUM(CASE WHEN product IN (1,2) THEN quantity ELSE 0 END) AS v "
     "FROM sales GROUP BY customer",
     lambda f: f.assign(t=np.where(f["product"].isin([1, 2]), f["quantity"], 0.0))),
    ("SELECT customer, SUM(CASE WHEN date BETWEEN 5 AND 10 THEN quantity END) AS v "
     "FROM sales GROUP BY customer",
     lambda f: f.assign(t=np.where(f["date"].between(5, 10), f["quantity"], 0.0))),
    ("SELECT customer, SUM(CASE WHEN product IN (1) THEN quantity "
     "WHEN product IN (2,3) THEN quantity * 2 ELSE 0 END) AS v FROM sales GROUP BY customer",
     lambda f: f.assign(t=np.where(f["product"] == 1, f["quantity"],
                                   np.where(f["product"].isin([2, 3]),
                                            f["quantity"] * 2, 0.0)))),
    ("SELECT customer, SUM(CASE WHEN product IN (1,2) THEN 1 ELSE 0 END) AS v "
     "FROM sales GROUP BY customer",
     lambda f: f.assign(t=np.where(f["product"].isin([1, 2]), 1.0, 0.0))),
])
def test_case_when_expands_into_indicator_operands(amdb_sql, pandas_expr):
    """Разбор случаев считается по строкам факта, как и в SQL.

    Ветвление раскрывается умножением на индикатор, поэтому счёт идёт по
    исходным строкам, а не по непустым ячейкам: за это отвечает спутниковый
    счётный гиперкуб (см. предложение 1 статьи).
    """
    db, sales = _case_fixture()
    got = dict(zip(db.sql(amdb_sql).column("customer"), db.sql(amdb_sql).column("v")))
    expected = pandas_expr(sales).groupby("customer")["t"].sum().to_dict()
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-9)


def test_case_when_puts_indicators_into_the_same_contraction():
    """Ветвь не добавляет прохода по данным — только сомножитель."""
    db, _ = _case_fixture()
    plan = db.compile("SELECT customer, SUM(CASE WHEN product IN (1,2) "
                      "THEN quantity ELSE 0 END) AS v FROM sales GROUP BY customer",
                      use_cache=False)
    names = [o.name for _, step in plan.aggregates[0].terms for o in step.operands]
    assert any(n.startswith("case") for n in names), names


def test_theta_join_matches_duckdb():
    """Соединение по неравенству между разными измерениями."""
    duckdb = pytest.importorskip("duckdb")
    pd = pytest.importorskip("pandas")
    from amdb import Database

    rng = np.random.default_rng(1)
    n = 8_000
    sales = pd.DataFrame({"customer": rng.integers(0, 10, n),
                          "date": rng.integers(0, 12, n),
                          "quantity": rng.random(n) * 10})
    budget = pd.DataFrame([(r, p) for r in range(4) for p in range(12)],
                          columns=["region", "period"])
    budget["plan"] = rng.random(len(budget)) * 50
    db = Database()
    db.load_frame(sales, ["customer", "date"], "quantity", "sales")
    db.load_frame(budget, ["region", "period"], "plan", "budget")

    con = duckdb.connect()
    con.register("_s", sales)
    con.register("_b", budget)
    con.execute("CREATE TABLE sales AS SELECT customer, date, SUM(quantity) quantity "
                "FROM _s GROUP BY 1, 2")
    con.execute("CREATE TABLE budget AS SELECT * FROM _b")

    query = ("SELECT customer, SUM(sales.quantity * budget.plan) AS v FROM sales "
             "JOIN budget ON sales.date <= budget.period GROUP BY customer")
    reference = ("SELECT s.customer, SUM(s.quantity * b.plan) AS v FROM sales s "
                 "JOIN budget b ON s.date <= b.period GROUP BY 1")
    got = {r[0]: r[1] for r in db.sql(query).rows}
    expected = {r[0]: r[1] for r in con.execute(reference).fetchall()}
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-9)

    spec = db.einsum_of(query)[0]
    assert spec.count(",") == 2, f"матрица сравнения должна быть операндом: {spec}"


def test_theta_join_rejects_dimensions_too_fine_for_a_comparison_matrix():
    """Матрица сравнения имеет размер произведения мощностей — это надо сказать."""
    pd = pytest.importorskip("pandas")
    from amdb import Database
    from amdb.ql.binder import BindError

    rng = np.random.default_rng(2)
    left = pd.DataFrame({"a": rng.integers(0, 20_000, 40_000),
                         "x": rng.random(40_000)})
    right = pd.DataFrame({"b": range(20_000), "y": np.random.random(20_000)})
    db = Database()
    db.load_frame(left, ["a"], "x", "l")
    db.load_frame(right, ["b"], "y", "r")
    with pytest.raises(BindError, match="слишком мелк"):
        db.compile("SELECT SUM(l.x * r.y) AS v FROM l JOIN r ON l.a <= r.b",
                   use_cache=False)


def test_select_distinct_is_grouping_without_aggregates():
    pd = pytest.importorskip("pandas")
    from amdb import Database

    rng = np.random.default_rng(3)
    n = 5_000
    frame = pd.DataFrame({"customer": rng.integers(0, 20, n),
                          "product": rng.integers(0, 7, n),
                          "q": rng.random(n)})
    db = Database()
    db.load_frame(frame, ["customer", "product"], "q", "sales")

    assert len(db.sql("SELECT DISTINCT customer FROM sales").rows) == \
        frame["customer"].nunique()
    pairs = db.sql("SELECT DISTINCT customer, product FROM sales").rows
    assert len(pairs) == len(frame[["customer", "product"]].drop_duplicates())
    assert len(db.sql("SELECT DISTINCT product FROM sales LIMIT 3").rows) == 3


# --- отсутствующие значения (NULL) -----------------------------------------
def _null_frame(pd, np_):
    """Факт с NULL и в мере, и в измерении."""
    rng = np_.random.default_rng(11)
    n = 5_000
    frame = pd.DataFrame({"customer": rng.integers(0, 5, n),
                          "product": rng.integers(0, 4, n).astype(float),
                          "quantity": rng.random(n).round(3)})
    frame.loc[frame.index[:900], "product"] = np_.nan
    frame.loc[frame.index[1000:1700], "quantity"] = np_.nan
    return frame


@pytest.mark.parametrize("sql", [
    "SELECT customer, SUM(quantity) AS s FROM sales GROUP BY customer",
    "SELECT customer, COUNT(quantity) AS c FROM sales GROUP BY customer",
    "SELECT customer, AVG(quantity) AS a FROM sales GROUP BY customer",
    "SELECT product, SUM(quantity) AS s FROM sales GROUP BY product",
    "SELECT SUM(quantity) AS s FROM sales WHERE product = 2",
    "SELECT SUM(quantity) AS s FROM sales WHERE product IS NULL",
    "SELECT SUM(quantity) AS s FROM sales WHERE product IS NOT NULL",
    "SELECT SUM(quantity) AS s FROM sales WHERE product BETWEEN 1 AND 2",
    "SELECT SUM(quantity) AS s FROM sales WHERE product IN (0, 1)",
    "SELECT SUM(quantity) AS s FROM sales WHERE NOT (product = 2)",
    "SELECT customer, SUM(CASE WHEN product = 2 THEN quantity ELSE 0 END) AS a "
    "FROM sales GROUP BY customer",
])
def test_null_semantics_match_sql(sql):
    """Отсутствующее значение ведёт себя ровно как NULL в SQL.

    Мера NULL схлопывается в нейтральный по ⊕ нуль и не попадает в счётный куб,
    поэтому SUM, COUNT и AVG совпадают с SQL. Измерение NULL получает
    собственный ординал: группировка собирает их в одну группу, ни одно
    сравнение на этом ординале не выполняется, а отрицание сравнения его тоже
    не пропускает — двузначная арифметика индикаторов воспроизводит трёхзначную
    логику WHERE.
    """
    pd = pytest.importorskip("pandas")
    duckdb = pytest.importorskip("duckdb")
    from amdb import Database

    frame = _null_frame(pd, np)
    db = Database()
    db.load_frame(frame, ["customer", "product"], "quantity", "sales",
                  ordered_dims=["product"])
    con = duckdb.connect()
    con.register("_f", frame)
    con.execute("CREATE TABLE sales AS SELECT * FROM _f")

    def key(value):
        if value is None or (isinstance(value, float) and value != value):
            return "NULL"
        return "NULL" if repr(value) == "NULL" else round(float(value), 6)

    got = {tuple(key(x) for x in r[:-1]): key(r[-1]) for r in db.sql(sql).rows}
    exp = {tuple(key(x) for x in r[:-1]): key(r[-1])
           for r in con.execute(sql).fetchall()}
    assert got == exp


def test_null_appears_on_incremental_load():
    """Словарь append-only: NULL может появиться и во второй порции данных."""
    pd = pytest.importorskip("pandas")
    from amdb import Database

    first = pd.DataFrame({"product": [0.0, 1.0, 2.0], "q": [1.0, 2.0, 3.0]})
    both = pd.DataFrame({"product": [0.0, 1.0, 2.0, np.nan], "q": [1.0, 2.0, 3.0, 4.0]})
    db = Database()
    db.load_frame(first, ["product"], "q", "s", ordered_dims=["product"])
    assert db.catalog.dimension("product").null_ordinal is None
    db.load_frame(both, ["product"], "q", "s", ordered_dims=["product"])

    dim = db.catalog.dimension("product")
    assert dim.null_ordinal == 3, "NULL занимает следующий свободный ординал"
    assert [repr(v) for v in dim.values] == ["0.0", "1.0", "2.0", "NULL"]
    rows = db.sql("SELECT product, SUM(q) AS s FROM s GROUP BY product").rows
    assert dict((repr(r[0]), r[1]) for r in rows)["NULL"] == 4.0


def test_is_null_on_dimension_without_nulls_is_rejected():
    pd = pytest.importorskip("pandas")
    from amdb import Database

    from amdb.ql import BindError

    frame = pd.DataFrame({"product": [0, 1, 2], "q": [1.0, 2.0, 3.0]})
    db = Database()
    db.load_frame(frame, ["product"], "q", "s")
    with pytest.raises(BindError, match="отсутствующих значений"):
        db.compile("SELECT SUM(q) AS s FROM s WHERE product IS NULL",
                   use_cache=False)
