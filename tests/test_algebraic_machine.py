# -*- coding: utf-8 -*-
"""Свойства алгебраической машины: план как терм, оптимизация как переписывание.

Концепция алгебраической машины баз данных (В.И. Мунерман и др.) предполагает
двухосновную алгебраическую систему: одна основа — данные, другая — операции
над ними; программа есть терм второй основы, а оптимизация — переписывание
этого терма, сохраняющее интерпретацию.

Здесь проверяется именно машинное свойство, а не отдельная операция:

* план запроса есть терм из (λ, μ)-произведений (`term`, `sokolov`);
* любой допустимый порядок свёрток даёт один и тот же результат — то есть
  переписывание терма семантику сохраняет;
* добавление операнда прав доступа коммутирует с оптимизацией;
* интерпретация терма операциями ядра совпадает с исполнением плана движком.

Оговорка: работы Мунермана в проекте отсутствуют, поэтому проверяется
соответствие формулировке из ТЗ, а не первоисточнику.
"""
import itertools

import numpy as np
import pytest

from amdb.ql.planner import ARRAY, INDICATOR, binary_decomposition, term_expression

QUERY = ("SELECT customer, month, SUM(quantity * price) AS rev FROM sales "
         "JOIN product ON sales.product = product.product "
         "WHERE region = 'Смоленск' GROUP BY customer, month")


def _operand_arrays(db, step) -> list[np.ndarray]:
    arrays = []
    for op in step.operands:
        if op.kind == ARRAY:
            arrays.append(np.asarray(op.array, dtype=np.float64))
        else:
            data = db.cube(op.name).dense().data.astype(np.float64)
            arrays.append((data != 0).astype(float) if op.kind == INDICATOR else data)
    return arrays


def _all_pairwise_paths(n: int) -> list[list[tuple[int, int]]]:
    """Все порядки попарной свёртки n операндов."""
    if n <= 2:
        return [[(0, 1)]]
    out = []
    for pair in itertools.combinations(range(n), 2):
        for rest in _all_pairwise_paths(n - 1):
            out.append([pair] + rest)
    return out


# --- Машинное свойство: переписывание терма сохраняет семантику -------------
def test_every_contraction_order_gives_the_same_result(db):
    """Все 18 порядков парных свёрток дают один результат.

    Это и есть машинное свойство: терм можно переписывать (менять порядок
    применения операций), не меняя интерпретацию. Именно на нём стоит
    оптимизатор — иначе выбор порядка свёрток менял бы ответ.
    """
    plan = db.compile(QUERY, use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    arrays = _operand_arrays(db, step)
    assert len(arrays) == 4

    reference = np.einsum(step.spec, *arrays, optimize="optimal")
    paths = _all_pairwise_paths(len(arrays))
    assert len(paths) == 18

    worst = 0.0
    for path in paths:
        got = np.einsum(step.spec, *arrays, optimize=["einsum_path"] + path)
        worst = max(worst, np.abs(got - reference).max() / np.abs(reference).max())
    assert worst < 1e-12, f"переписывание терма меняет результат: {worst:.2e}"


@pytest.mark.parametrize("mode", ["optimal", "greedy", True, False])
def test_optimization_mode_does_not_change_semantics(db, mode):
    plan = db.compile(QUERY, use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    arrays = _operand_arrays(db, step)
    reference = np.einsum(step.spec, *arrays, optimize="optimal")
    got = np.einsum(step.spec, *arrays, optimize=mode)
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-12


def test_interpretation_of_term_equals_engine_execution(db):
    """Интерпретация терма операциями ядра совпадает с исполнением движком."""
    from amdb.core import MultidimensionalMatrix, convolve_named

    plan = db.compile(QUERY, use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    by_axes = {tuple(op.axes): arr
               for op, arr in zip(step.operands, _operand_arrays(db, step))}

    current: MultidimensionalMatrix | None = None
    for bp in binary_decomposition(step):
        def side(axes, ones=False):
            if ones:
                shape = tuple(current.axis_length(a) for a in axes)
                return MultidimensionalMatrix(np.ones(shape), axes)
            if current is not None and current.axes == axes:
                return current
            return MultidimensionalMatrix(by_axes[axes], axes)

        left = side(bp.left)
        right = side(bp.right, bp.right_is_ones)
        current = convolve_named(left, right, keep=set(bp.lam_axes))

    engine = db._executor._run_step(step, plan.group_axes)
    got = current.transpose(step.output).data
    assert np.abs(got - engine).max() / np.abs(engine).max() < 1e-12


# --- Двухосновность: данные и операции — разные сорта -----------------------
def test_plan_is_a_first_class_term(db):
    """План — самостоятельный объект, а не побочный эффект исполнения."""
    plan = db.compile(QUERY, use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    assert step.spec == "abc,b,cd,a->ad"
    assert step.path is not None, "порядок свёрток — часть терма, а не рантайма"
    text = term_expression(step)
    assert text.count("∗[") == 3, f"терм должен содержать три произведения: {text}"
    assert "sales" in text and "price" in text and "rollup" in text and "mask" in text


def test_term_rendering_reflects_lambda_mu(db):
    text = db.term(QUERY)
    assert "∗[0,1]" in text, "свёртки по product и date — (0,1)"
    assert "∗[1,0]" in text, "маска по customer — (1,0)"


def test_term_of_linear_combination_has_two_products(db):
    """SUM(a·b − c) — линейная комбинация двух термов (по билинейности)."""
    text = db.term("SELECT customer, SUM(quantity * price - quantity) AS n FROM sales "
                   "JOIN product ON sales.product = product.product GROUP BY customer")
    assert " + " in text and "-1·" in text


def test_term_of_avg_is_ratio_of_two_terms(db):
    text = db.term("SELECT product, AVG(quantity) AS a FROM sales GROUP BY product")
    assert " / " in text
    assert text.count("∗[0,2]") == 2


def test_term_marks_minmax_as_outside_the_ring(db):
    text = db.term("SELECT product, MAX(quantity) AS m FROM sales GROUP BY product")
    assert "max_" in text and "вне (+,·)" in text


# --- Права доступа как операция той же алгебры ------------------------------
def test_access_control_is_an_algebraic_rewriting(db, frames):
    """Ограничение доступа добавляет операнд в терм, а не фильтрует результат."""
    from amdb import Database

    sales, products, customers = frames
    guarded = Database()
    guarded.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    guarded.load_dimension(products, "product", measures=["price"])
    guarded.load_dimension(customers, "customer", attributes=["region"])
    guarded.grant("limited", "customer", allowed=[0, 1, 2])

    plain = guarded.compile("SELECT customer, SUM(quantity) AS q FROM sales "
                            "GROUP BY customer", use_cache=False)
    secured = guarded.compile("SELECT customer, SUM(quantity) AS q FROM sales "
                              "GROUP BY customer", role="limited", use_cache=False)
    n_plain = len(plain.aggregates[0].terms[0][1].operands)
    n_secured = len(secured.aggregates[0].terms[0][1].operands)
    assert n_secured == n_plain + 1, "право доступа должно стать операндом терма"
    assert "rls:customer" in term_expression(secured.aggregates[0].terms[0][1])


def test_access_control_commutes_with_optimization(db, frames):
    """Оптимизация терма с маской прав даёт тот же результат, что и без неё.

    Проверяется, что вставка операнда прав и переписывание порядка свёрток
    коммутируют: иначе оптимизатор мог бы «оптимизировать» права доступа.
    """
    from amdb import Database

    sales, products, customers = frames
    guarded = Database()
    guarded.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    guarded.load_dimension(customers, "customer", attributes=["region"])
    guarded.grant("limited", "customer", allowed=[0, 1, 2])

    secured = guarded.sql("SELECT customer, SUM(quantity) AS q FROM sales "
                          "GROUP BY customer", role="limited")
    manual = guarded.sql("SELECT customer, SUM(quantity) AS q FROM sales "
                         "WHERE customer IN (0, 1, 2) GROUP BY customer")
    assert secured.column("customer") == manual.column("customer")
    assert np.allclose(secured.column("q"), manual.column("q"))


# --- Отображение реляционной модели в матричную ----------------------------
def _cube_of(rows, dims, measure):
    from amdb import Database

    db = Database()
    db.load_frame(rows, dims, measure, "f")
    return db, db.cube("f").matrix.data


def test_mapping_is_injective_when_dimensions_form_a_key():
    """Если измерения — ключ отношения, отображение обратимо (изоморфизм на образ)."""
    pd = pytest.importorskip("pandas")

    rows = pd.DataFrame({"a": [0, 0, 1, 1], "b": [0, 1, 0, 1],
                         "v": [3.0, 5.0, 7.0, 11.0]})
    _, cube = _cube_of(rows, ["a", "b"], "v")
    recovered = {(i, j): cube[i, j] for i in range(2) for j in range(2) if cube[i, j]}
    original = {(r.a, r.b): r.v for r in rows.itertuples()}
    assert recovered == original


def test_mapping_is_not_injective_when_dimensions_are_not_a_key():
    """Если измерения не ключ, разные отношения дают один куб — изоморфизма нет.

    Это математическая причина расхождения гранулярности MIN/MAX: слой
    (прообраз ячейки) схлопывается суммой, и информация о его составе теряется.
    """
    pd = pytest.importorskip("pandas")

    left = pd.DataFrame({"a": [0, 0, 1], "b": [0, 0, 1], "v": [3.0, 5.0, 4.0]})
    right = pd.DataFrame({"a": [0, 0, 1], "b": [0, 0, 1], "v": [1.0, 7.0, 4.0]})
    _, cube_left = _cube_of(left, ["a", "b"], "v")
    _, cube_right = _cube_of(right, ["a", "b"], "v")
    assert np.allclose(cube_left, cube_right), "разные отношения дали разные кубы"

    # Следствие: максимум по ячейкам не равен максимуму по строкам
    assert cube_left.max() == 8.0
    assert left.v.max() == 5.0


def test_mapping_is_a_homomorphism_for_sum():
    """Куб объединения отношений равен сумме кубов: φ(R ⊎ R′) = φ(R) ⊕ φ(R′)."""
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(5)
    left = pd.DataFrame({"a": rng.integers(0, 4, 20), "b": rng.integers(0, 3, 20),
                         "v": rng.random(20)})
    right = pd.DataFrame({"a": rng.integers(0, 4, 15), "b": rng.integers(0, 3, 15),
                          "v": rng.random(15)})
    union = pd.concat([left, right], ignore_index=True)
    # Общие словари измерений: строим по объединению, чтобы формы совпали
    from amdb import Database

    def cube(frame):
        db = Database()
        db.load_frame(union.assign(v=0.0), ["a", "b"], "v", "f")   # фиксируем словари
        db.load_frame(frame, ["a", "b"], "v", "f")
        return db.cube("f").matrix.data

    assert np.allclose(cube(union), cube(left) + cube(right))


def test_count_cube_is_a_finer_invariant():
    """Пара (куб сумм, счётный куб) различает то, что куб сумм не различает."""
    pd = pytest.importorskip("pandas")

    from amdb import Database

    one = pd.DataFrame({"a": [0, 0], "b": [0, 0], "v": [4.0, 4.0]})
    two = pd.DataFrame({"a": [0], "b": [0], "v": [8.0]})
    db1, db2 = Database(), Database()
    db1.load_frame(one, ["a", "b"], "v", "f")
    db2.load_frame(two, ["a", "b"], "v", "f")
    assert np.allclose(db1.cube("f").matrix.data, db2.cube("f").matrix.data)
    assert not np.allclose(db1.cube("f__count").matrix.data,
                           db2.cube("f__count").matrix.data)


# --- Предварительная свёртка «висячих» осей ---------------------------------
def test_projections_are_pushed_before_the_product(db):
    """Оси операнда, никому больше не нужные, сворачиваются до произведения.

    einsum подбирает лишь порядок стягивания между операндами и собственные
    «висячие» оси операнда не сворачивает. Пропуск этой оптимизации стоил
    на порядок больше работы (см. optimizer.push_projections).
    """
    plan = db.compile("SELECT month, COUNT(*) AS n FROM sales GROUP BY month",
                      use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    cube = next(o for o in step.operands if o.name.startswith("sales"))
    assert cube.presum == ("customer", "product"), "лишние оси должны сворачиваться заранее"
    assert cube.axes == ("date",)
    assert step.spec == "a,ab->b"


def test_pushing_projections_preserves_the_result(db, enriched):
    """Оптимизация обязана сохранять ответ — это следствие теоремы о разложении."""
    from amdb.ql.optimizer import push_projections

    sql = "SELECT month, SUM(quantity) AS q FROM sales GROUP BY month"
    optimized = db.sql(sql).to_pandas().sort_values("month").reset_index(drop=True)

    # Тот же план без предварительной свёртки: строим и считаем вручную.
    plan = db.compile(sql, use_cache=False)
    step = plan.aggregates[0].terms[0][1]
    assert any(o.presum for o in step.operands), "оптимизация должна была сработать"

    expected = (enriched.groupby("month", as_index=False).quantity.sum()
                .sort_values("month").reset_index(drop=True))
    assert np.allclose(optimized["q"], expected["quantity"])


def test_projection_appears_in_the_algebraic_decomposition(db):
    """Предварительная свёртка — не «трюк реализации», а шаг терма."""
    from amdb.ql.planner import binary_decomposition

    plan = db.compile("SELECT month, COUNT(*) AS n FROM sales GROUP BY month",
                      use_cache=False)
    chain = binary_decomposition(plan.aggregates[0].terms[0][1])
    first = chain[0]
    assert first.right_is_ones, "первый шаг — свёртка с матрицей единиц"
    assert first.mu_axes == ("customer", "product")
    assert first.result == ("date",)


# --- Сигнатура объявлена явно и не расходится с тестами ---------------------
def test_signature_declares_both_sorts():
    from amdb.core import SYSTEM

    names = [s.name for s in SYSTEM.sorts]
    assert any("S" in n for n in names) and any("T" in n for n in names)
    assert len(SYSTEM.operations) >= 8
    assert SYSTEM.outside, "операции вне сигнатуры должны быть перечислены явно"


def test_every_declared_axiom_has_an_existing_test():
    """Ссылки на проверки не должны протухать: каждая аксиома проверяема.

    Без этого теста объявление аксиом превратилось бы в декларацию о
    намерениях — ровно то, чего в алгебраической системе быть не должно.
    """
    import pathlib
    import re

    from amdb.core import SYSTEM

    tests_dir = pathlib.Path(__file__).parent
    cache: dict[str, str] = {}
    missing = []
    for axiom in SYSTEM.axioms:
        for ref in axiom.test.split(","):
            ref = ref.strip()
            if "::" not in ref:
                continue
            filename, func = ref.split("::")
            if filename not in cache:
                path = tests_dir / filename
                cache[filename] = path.read_text(encoding="utf-8") if path.exists() else ""
            if not re.search(rf"def {re.escape(func)}\b", cache[filename]):
                missing.append(f"{axiom.name} -> {ref}")
    assert not missing, "аксиомы ссылаются на несуществующие тесты: " + "; ".join(missing)


def test_signature_description_is_renderable():
    from amdb.core import SYSTEM

    text = SYSTEM.describe()
    assert "E = (S, T; Ω)" in text
    assert "conv[λ,μ]" in text
    assert "Аксиомы" in text


# --- Замкнутость сорта данных ----------------------------------------------
def test_data_sort_is_closed_under_signature(rng=None):
    """Результат каждой операции сигнатуры — снова многомерная матрица."""
    from amdb.core import (
        MultidimensionalMatrix,
        convolve_named,
        internal_convolution,
        reduce_axes,
    )

    r = np.random.default_rng(1)
    A = MultidimensionalMatrix(r.random((3, 4, 5)), ("a", "b", "c"))
    B = MultidimensionalMatrix(r.random((4, 5, 6)), ("b", "c", "d"))
    S = MultidimensionalMatrix(r.random((4, 4)), ("b", "b2"))
    results = [
        convolve_named(A, B, keep={"b"}),
        A + A,
        A * 2.0,
        A.transpose(("c", "a", "b")),
        A.project(["b"]),
        A.slice(a=1),
        reduce_axes(A, ["b"], "max"),
        internal_convolution(MultidimensionalMatrix(r.random((3, 4, 4)),
                                                    ("a", "b", "b2")), "b", "b2"),
    ]
    assert all(isinstance(x, MultidimensionalMatrix) for x in results)
    assert all(len(x.axes) == x.data.ndim for x in results)
    assert S.axes == ("b", "b2")
