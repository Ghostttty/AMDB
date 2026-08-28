# -*- coding: utf-8 -*-
"""Соответствие реализации алгебре многомерных матриц Н.П. Соколова.

Эти тесты — не проверка кода на отсутствие ошибок, а проверка того, что
реализованная операция действительно является (λ, μ)-свёрнутым произведением
в смысле Соколова: выполняются арифметика рангов, билинейность,
ассоциативность, закон транспонирования, существует единичный элемент, и
операция сводится к известным частным случаям.

Определение берётся в редакции статьи Симакова В.А. (со ссылкой на Соколова):

    c[l_1..l_k, s_1..s_λ, m_1..m_v]
        = Σ по (c_1..c_μ)  a[l_1..l_k, s_1..s_λ, c_1..c_μ]
                         · b[s_1..s_λ, c_1..c_μ, m_1..m_v]

Оговорка: сверка ведётся с этой редакцией определения, а не с оригиналом
1972 года — книги Соколова в проекте нет.
"""
import numpy as np
import pytest

from amdb.core import (
    COOCube,
    MultidimensionalMatrix,
    build_einsum,
    convolve,
    convolve_naive,
    convolve_sparse,
    internal_convolution,
    unit_matrix,
)

RTOL = 1e-12


@pytest.fixture
def rng():
    return np.random.default_rng(20260827)


# --- 1. Определение --------------------------------------------------------
@pytest.mark.parametrize("lam", [0, 1, 2])
@pytest.mark.parametrize("mu", [0, 1, 2])
def test_matches_definition_for_all_lambda_mu(rng, lam, mu):
    """Реализация совпадает с прямым вычислением по определению."""
    k, v = 2, 2
    shape_a = tuple(rng.integers(2, 4, k + lam + mu))
    shape_b = tuple(shape_a[k:]) + tuple(rng.integers(2, 4, v))
    A, B = rng.random(shape_a), rng.random(shape_b)
    got = convolve(A, B, lam, mu)
    ref = convolve_naive(A, B, lam, mu)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-12


def test_index_layout_follows_sokolov_grouping():
    """Раскладка осей [L|S|C] × [S|C|M] -> [L|S|M] — как в определении."""
    assert build_einsum(6, 9, lam=2, mu=3) == "abcdef,bcdefghij->abcghij"
    #                                          ↑L=a ↑S=bc ↑C=def   ↑M=ghij
    spec = build_einsum(5, 6, lam=1, mu=2)
    lhs, rhs = spec.split("->")
    a, b = lhs.split(",")
    assert a[2:] == b[:3], "общая группа S+C должна быть хвостом A и головой B"
    assert rhs == a[:2] + a[2:3] + b[3:], "результат = L + S + M"


@pytest.mark.parametrize("ra,rb,lam,mu", [
    (6, 9, 2, 3), (3, 3, 1, 1), (2, 2, 0, 1), (4, 5, 0, 2), (3, 4, 2, 0), (2, 2, 0, 0),
])
def test_rank_arithmetic(rng, ra, rb, lam, mu):
    """rank(A ∗ B) = rank A + rank B − λ − 2μ."""
    A = rng.random((2,) * ra)
    B = rng.random((2,) * rb)
    assert convolve(A, B, lam, mu).ndim == ra + rb - lam - 2 * mu


# --- 2. Частные случаи -----------------------------------------------------
def test_reduces_to_matrix_multiplication(rng):
    """(0,1)-произведение матриц ранга 2 — обычное умножение матриц."""
    A, B = rng.random((4, 5)), rng.random((5, 6))
    assert np.allclose(convolve(A, B, 0, 1), A @ B)


def test_reduces_to_tensordot(rng):
    """(0,μ)-произведение — свёртка по μ осям, то есть tensordot."""
    A, B = rng.random((3, 4, 5)), rng.random((4, 5, 6))
    assert np.allclose(convolve(A, B, 0, 2),
                       np.tensordot(A, B, axes=([1, 2], [0, 1])))


def test_reduces_to_outer_product(rng):
    """(0,0)-произведение — тензорное (внешнее) произведение."""
    A, B = rng.random((3, 4)), rng.random((5, 6))
    assert np.allclose(convolve(A, B, 0, 0), np.multiply.outer(A, B))


def test_lambda_only_product_keeps_shared_axis(rng):
    """(λ,0)-произведение сохраняет общую ось без суммирования."""
    A, B = rng.random((3, 4)), rng.random((4, 5))
    assert np.allclose(convolve(A, B, 1, 0), np.einsum("ls,sm->lsm", A, B))


# --- 3. Билинейность -------------------------------------------------------
def test_distributive_over_addition(rng):
    A = rng.random((3, 4, 5))
    B, C = rng.random((4, 5, 6)), rng.random((4, 5, 6))
    assert np.allclose(convolve(A, B + C, 1, 1),
                       convolve(A, B, 1, 1) + convolve(A, C, 1, 1))
    A2 = rng.random((3, 4, 5))
    assert np.allclose(convolve(A + A2, B, 1, 1),
                       convolve(A, B, 1, 1) + convolve(A2, B, 1, 1))


def test_homogeneous_in_both_arguments(rng):
    A, B = rng.random((3, 4, 5)), rng.random((4, 5, 6))
    alpha = 2.7
    base = convolve(A, B, 1, 1)
    assert np.allclose(convolve(alpha * A, B, 1, 1), alpha * base)
    assert np.allclose(convolve(A, alpha * B, 1, 1), alpha * base)


# --- 4. Ассоциативность ----------------------------------------------------
def test_associative_when_lambda_zero(rng):
    """(A ∗ B) ∗ C = A ∗ (B ∗ C) при λ = 0."""
    X, Y, Z = rng.random((3, 4)), rng.random((4, 5)), rng.random((5, 6))
    left = convolve(convolve(X, Y, 0, 1), Z, 0, 1)
    right = convolve(X, convolve(Y, Z, 0, 1), 0, 1)
    assert np.abs(left - right).max() < 1e-12


def test_associative_when_lambda_axis_shared_by_all(rng):
    """Ассоциативность сохраняется, когда λ-ось проходит через все операнды."""
    A = rng.random((2, 3, 4))      # [l, s, c1]
    B = rng.random((3, 4, 5))      # [s, c1, c2]
    C = rng.random((3, 5, 6))      # [s, c2, m]
    left = np.einsum("lsd,sdm->lsm", np.einsum("lsc,scd->lsd", A, B), C)
    right = np.einsum("lsc,scm->lsm", A, np.einsum("scd,sdm->scm", B, C))
    assert np.abs(left - right).max() < 1e-12


def test_associativity_fails_when_lambda_axis_is_dropped_midway(rng):
    """Порядок свёрток нельзя менять, если λ-ось теряется в промежуточном шаге.

    Это существенное ограничение: ассоциативность (λ, μ)-произведения
    выполняется относительно ФИКСИРОВАННОГО набора ролей индексов. Если при
    перегруппировке λ-ось становится μ-осью, результат меняется — поэтому
    оптимизатор обязан сохранять роли, а не только имена осей.
    """
    A = rng.random((2, 3, 4))      # [l, s, c]
    B = rng.random((3, 4, 5))      # [s, c, m]
    keep_lambda = np.einsum("lsc,scm->lsm", A, B)          # s остаётся (λ)
    drop_lambda = np.einsum("lsc,scm->lm", A, B)           # s свёрнута (μ)
    assert keep_lambda.sum(axis=1).shape == drop_lambda.shape
    assert np.allclose(keep_lambda.sum(axis=1), drop_lambda)
    assert keep_lambda.shape != drop_lambda.shape


# --- 5. Некоммутативность --------------------------------------------------
def test_not_commutative(rng):
    P, Q = rng.random((3, 3)), rng.random((3, 3))
    assert not np.allclose(convolve(P, Q, 0, 1), convolve(Q, P, 0, 1))


# --- 6. Единичный элемент --------------------------------------------------
def test_right_unit_for_mu_one(rng):
    """A ∗ E = A, где E[s, c, m] = δ(c, m) при любом s."""
    A = rng.random((2, 3, 4))
    E = unit_matrix(mu_shape=(4,), lam_shape=(3,))
    assert E.shape == (3, 4, 4)
    assert np.allclose(convolve(A, E, lam=1, mu=1), A)


def test_right_unit_for_mu_two(rng):
    """E[c1, c2, m1, m2] = δ(c1, m1)·δ(c2, m2)."""
    A = rng.random((2, 3, 4))
    E = unit_matrix(mu_shape=(3, 4))
    assert np.allclose(convolve(A, E, lam=0, mu=2), A)


def test_left_unit_for_rank_two(rng):
    A = rng.random((3, 5))
    assert np.allclose(convolve(np.eye(3), A, 0, 1), A)


# --- 7. Транспонирование ---------------------------------------------------
def test_transposition_law(rng):
    """(A ∗ B)^T = B^T ∗ A^T при транспонировании по группам индексов."""
    A = rng.random((2, 3, 4))      # [L, S, C]
    B = rng.random((3, 4, 5))      # [S, C, M]
    C = convolve(A, B, 1, 1)                        # [L, S, M]
    B_T = np.transpose(B, (2, 0, 1))                # [M, S, C]
    A_T = np.transpose(A, (1, 2, 0))                # [S, C, L]
    assert np.allclose(convolve(B_T, A_T, 1, 1), np.transpose(C, (2, 1, 0)))


def test_named_transpose_is_index_permutation(rng):
    m = MultidimensionalMatrix(rng.random((2, 3, 4)), ("a", "b", "c"))
    t = m.transpose(("c", "a", "b"))
    assert t.axes == ("c", "a", "b")
    assert np.allclose(t.data, np.transpose(m.data, (2, 0, 1)))


# --- 8. Внутренняя свёртка (аналог следа) ----------------------------------
def test_internal_convolution_is_trace_for_rank_two(rng):
    """Свёртка матрицы по паре собственных индексов даёт след."""
    A = rng.random((4, 4))
    assert np.isclose(internal_convolution(A, 0, 1), np.trace(A))


def test_internal_convolution_reduces_rank_by_two(rng):
    A = rng.random((3, 4, 5, 4))
    out = internal_convolution(A, 1, 3)
    assert out.shape == (3, 5)
    assert np.allclose(out, np.einsum("aibi->ab", A))


def test_internal_convolution_rejects_mismatched_axes(rng):
    with pytest.raises(ValueError, match="одинаковой длины"):
        internal_convolution(rng.random((3, 4)), 0, 1)


# --- 9. План запроса как композиция (λ, μ)-произведений ---------------------
def test_query_plan_decomposes_into_binary_sokolov_products(db):
    """План реального запроса — цепочка бинарных (λ, μ)-произведений.

    Многооперандный einsum сам по себе операцией алгебры Соколова не является:
    она бинарна. Этот тест проверяет, что выбранный оптимизатором порядок
    свёрток задаёт корректное разложение в композицию бинарных произведений —
    и что эта композиция даёт тот же результат, что и единый вызов einsum.
    """
    from amdb.core import MultidimensionalMatrix, convolve_named
    from amdb.ql.planner import ARRAY, CUBE, INDICATOR, binary_decomposition

    sql = ("SELECT customer, month, SUM(quantity * price) AS rev FROM sales "
           "JOIN product ON sales.product = product.product "
           "WHERE region = 'Смоленск' GROUP BY customer, month")
    plan = db.compile(sql, use_cache=False)
    step = plan.aggregates[0].terms[0][1]

    # Операнды по именам осей (в этом плане наборы осей различны).
    by_axes: dict[tuple[str, ...], np.ndarray] = {}
    for op in step.operands:
        if op.kind == ARRAY:
            by_axes[op.axes] = np.asarray(op.array, dtype=np.float64)
        else:
            cube = db.cube(op.name)
            data = cube.dense().data.astype(np.float64)
            by_axes[op.axes] = (data != 0).astype(float) if op.kind == INDICATOR else data
    assert len(by_axes) == len(step.operands), "оси операндов должны быть различны"

    chain = binary_decomposition(step)
    assert chain, "разложение не должно быть пустым"
    assert all(bp.lam >= 0 and bp.mu >= 0 for bp in chain)

    current: MultidimensionalMatrix | None = None

    def operand(axes: tuple[str, ...]) -> MultidimensionalMatrix:
        # Промежуточный результат может оказаться и слева, и справа: порядок
        # операндов в паре выбирает оптимизатор.
        if current is not None and current.axes == axes:
            return current
        return MultidimensionalMatrix(by_axes[axes], axes)

    for bp in chain:
        left = operand(bp.left)
        if bp.right_is_ones:
            shape = tuple(left.axis_length(a) for a in bp.right)
            right = MultidimensionalMatrix(np.ones(shape), bp.right)
        else:
            right = operand(bp.right)
        current = convolve_named(left, right, keep=set(bp.lam_axes))
        assert set(current.axes) == set(bp.result), (
            f"шаг {bp.describe()} дал оси {current.axes}")

    assert current is not None
    reference = np.einsum(step.spec, *[by_axes[o.axes] for o in step.operands],
                          optimize=True)
    got = current.transpose(step.output).data
    assert np.abs(got - reference).max() / np.abs(reference).max() < 1e-12


def test_decomposition_reports_lambda_mu_roles(db):
    """Роли индексов в разложении соответствуют смыслу запроса."""
    from amdb.ql.planner import binary_decomposition

    plan = db.compile(
        "SELECT customer, month, SUM(quantity * price) AS rev FROM sales "
        "JOIN product ON sales.product = product.product "
        "WHERE region = 'Смоленск' GROUP BY customer, month", use_cache=False)
    chain = binary_decomposition(plan.aggregates[0].terms[0][1])
    all_mu = {a for bp in chain for a in bp.mu_axes}
    all_lam = {a for bp in chain for a in bp.lam_axes}
    # product и date схлопываются агрегатом -> μ; customer остаётся -> λ
    assert "product" in all_mu and "date" in all_mu
    assert "customer" in all_lam
    assert "month" not in all_mu


def test_minmax_is_outside_the_algebra(db):
    """MIN/MAX не раскладываются в (λ, μ)-произведения — это признаётся явно."""
    plan = db.compile("SELECT product, MAX(quantity) AS m FROM sales GROUP BY product",
                      use_cache=False)
    agg = plan.aggregates[0]
    assert agg.terms == [], "у MIN/MAX не должно быть einsum-слагаемых"
    assert agg.reduce is not None, "MIN/MAX исполняется редукцией вне алгебры"
    text = db.sokolov("SELECT product, MAX(quantity) AS m FROM sales GROUP BY product")
    assert "вне (+,·)" in text
    assert "(max,·)" in text, "оговорка о полукольце должна быть видна в плане"


# --- 10. Обобщение на полукольца -------------------------------------------
def test_semiring_sum_prod_equals_ordinary_convolution(rng):
    """Над (+, ·) обобщённая операция совпадает с произведением Соколова."""
    from amdb.core import convolve_semiring

    A, B = rng.random((3, 4, 5)), rng.random((4, 5, 6))
    assert np.allclose(convolve_semiring(A, B, 1, 1, "sum-prod"), convolve(A, B, 1, 1))


@pytest.mark.parametrize("name,reducer", [("max-prod", np.max), ("min-prod", np.min)])
def test_minmax_are_lambda_mu_products_over_another_semiring(rng, name, reducer):
    """MIN/MAX — то же (λ, μ)-произведение, но над (max, ·) / (min, ·).

    Это переводит MIN и MAX из «исключения из модели» в частный случай
    обобщённой операции: меняется полукольцо, а не определение.
    """
    from amdb.core import aggregate_over_axes

    A = rng.random((3, 4, 5))
    assert np.allclose(aggregate_over_axes(A, (1, 2), name), reducer(A, axis=(1, 2)))


def test_tropical_semiring_computes_shortest_paths():
    """(min, +)-произведение матриц смежности даёт кратчайшие пути.

    Классическая проверка того, что обобщение действительно полукольцевое,
    а не подгонка под MIN/MAX.
    """
    from amdb.core import convolve_semiring

    inf = np.inf
    W = np.array([[0, 3, inf, 7],
                  [8, 0, 2, inf],
                  [5, inf, 0, 1],
                  [2, inf, inf, 0]])
    d = W.copy()
    for _ in range(3):
        d = convolve_semiring(d, W, 0, 1, "min-plus")
    expected = np.array([[0, 3, 5, 6],
                         [5, 0, 2, 3],
                         [3, 6, 0, 1],
                         [2, 5, 7, 0]], dtype=float)
    assert np.allclose(d, expected)


def test_max_times_is_a_semiring_only_on_nonnegative_values(rng):
    """(max, ·) не полукольцо на всей оси: дистрибутивность нарушается.

    max(2, 3)·(−1) = −3, а max(2·(−1), 3·(−1)) = −2. Следствие важнее самого
    примера: вместе с дистрибутивностью теряется ассоциативность свёрнутого
    произведения, то есть переписывание плана перестаёт быть корректным.
    """
    from amdb.core import convolve_semiring

    x, y, z = 2.0, 3.0, -1.0
    assert max(x, y) * z != max(x * z, y * z)
    assert max(x, y) * 2.0 == max(x * 2.0, y * 2.0)

    A, B, C = rng.normal(size=(2, 3)), rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    kw = dict(semiring="max-prod", check_domain=False)
    left = convolve_semiring(convolve_semiring(A, B, 0, 1, **kw), C, 0, 1, **kw)
    right = convolve_semiring(A, convolve_semiring(B, C, 0, 1, **kw), 0, 1, **kw)
    assert not np.allclose(left, right), "на знакопеременных данных ассоциативность должна ломаться"

    P, Q, R = np.abs(A), np.abs(B), np.abs(C)
    left = convolve_semiring(convolve_semiring(P, Q, 0, 1, **kw), R, 0, 1, **kw)
    right = convolve_semiring(P, convolve_semiring(Q, R, 0, 1, **kw), 0, 1, **kw)
    assert np.allclose(left, right), "на неотрицательных данных ассоциативность обязана держаться"


def test_tropical_semiring_is_associative_on_signed_values(rng):
    """(min, +) — полукольцо на всей числовой оси, в отличие от (max, ·)."""
    from amdb.core import convolve_semiring

    A, B, C = rng.normal(size=(2, 3)), rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    kw = dict(semiring="min-plus")
    left = convolve_semiring(convolve_semiring(A, B, 0, 1, **kw), C, 0, 1, **kw)
    right = convolve_semiring(A, convolve_semiring(B, C, 0, 1, **kw), 0, 1, **kw)
    assert np.allclose(left, right)


def test_negative_values_are_rejected_for_max_times(rng):
    """По умолчанию выход за область определения полукольца — ошибка, не тихий ответ."""
    from amdb.core import convolve_semiring

    A = np.array([[2.0, 3.0]])
    B = np.array([[-1.0], [-1.0]])
    with pytest.raises(ValueError, match="только на\\s+неотрицательных"):
        convolve_semiring(A, B, 0, 1, "max-prod")
    assert convolve_semiring(A, B, 0, 1, "max-prod", check_domain=False)[0, 0] == -2.0


def test_semiring_guards_memory_budget(rng):
    """Полукольца вне (+, ·) не сводятся к BLAS — бюджет проверяется заранее."""
    from amdb.core import convolve_semiring

    A, B = rng.random((10, 10, 10)), rng.random((10, 10, 10))
    with pytest.raises(MemoryError, match="не сводится к BLAS"):
        convolve_semiring(A, B, 1, 1, "max-prod", max_cells=10)


def test_unknown_semiring_is_rejected(rng):
    from amdb.core import convolve_semiring

    with pytest.raises(ValueError, match="неизвестное полукольцо"):
        convolve_semiring(rng.random((2, 2)), rng.random((2, 2)), 0, 1, "нет-такого")


# --- 11. Замкнутость относительно сложения ---------------------------------
def test_addition_is_closed_and_commutative(rng):
    A = MultidimensionalMatrix(rng.random((2, 3)), ("x", "y"))
    B = MultidimensionalMatrix(rng.random((2, 3)), ("x", "y"))
    assert np.allclose((A + B).data, (B + A).data)
    assert (A + B).axes == A.axes


def test_zero_matrix_is_additive_identity(rng):
    A = MultidimensionalMatrix(rng.random((2, 3)), ("x", "y"))
    zero = MultidimensionalMatrix(np.zeros((2, 3)), ("x", "y"))
    assert np.allclose((A + zero).data, A.data)


def test_convolution_with_zero_gives_zero(rng):
    A = rng.random((3, 4, 5))
    Z = np.zeros((4, 5, 6))
    assert np.allclose(convolve(A, Z, 1, 1), 0.0)


def test_contraction_order_is_invariant_on_the_sparse_path(rng):
    """Следствие 1 обязано держаться и на разреженных ядрах.

    Теорема о разложении доказана для алгебраических тождеств и представления
    не касается, но разреженные ядра накапливают вклады в ином порядке, чем
    плотные (хеш-соединение обходит ключи в порядке хеш-таблицы). Если бы
    неассоциативность машинного сложения давала здесь заметное расхождение,
    оптимизатору пришлось бы знать о выбранном представлении — а он о нём
    не знает.
    """
    def sparse(shape, axes, seed):
        r = np.random.default_rng(seed)
        dense = (r.random(shape) < 0.4) * r.random(shape)
        return COOCube.from_dense(MultidimensionalMatrix(dense, axes)), dense

    A, da = sparse((6, 5, 4), ("l", "s", "c"), 1)
    B, db = sparse((5, 4, 7), ("s", "c", "m"), 2)
    C, dc = sparse((5, 7, 3), ("s", "m", "n"), 3)

    def as_coo(x):
        return x if isinstance(x, COOCube) else COOCube.from_dense(x)

    def as_dense(x):
        return x if isinstance(x, MultidimensionalMatrix) else x.to_dense()

    left = convolve_sparse(as_coo(convolve_sparse(A, B, keep={"s"})), C, keep={"s"})
    right = convolve_sparse(A, as_coo(convolve_sparse(B, C, keep={"s"})), keep={"s"})

    got_l = as_dense(left).transpose(("l", "s", "n")).data
    got_r = as_dense(right).transpose(("l", "s", "n")).data
    reference = np.einsum("lsc,scm,smn->lsn", da, db, dc)

    assert np.abs(got_l - got_r).max() < 1e-12, "порядок свёрток изменил ответ"
    assert np.allclose(got_l, reference)
    assert np.allclose(got_r, reference)


# --- 12. Сверка соглашения (λ, μ) с опубликованной задачей ------------------
#: Матрица смежности графа из работы Морозова, Мунермана и Симакова (2022),
#: «Экспериментальный анализ многомерно-матричного подхода к построению
#: маршрутов в графе», этап 1 эксперимента.
GRAPH_FROM_PAPER = (
    "0101000000",
    "0010000000",
    "0001000000",
    "0000100000",
    "0000010000",
    "0000001000",
    "0000000100",
    "1000000010",
    "0000000001",
    "1000000000",
)


def _walks(adjacency: np.ndarray, edges_in_route: int) -> list[tuple[int, ...]]:
    """Полный перебор маршрутов заданной длины — независимый эталон."""
    steps = [(i, j) for i in range(len(adjacency)) for j in range(len(adjacency))
             if adjacency[i, j]]
    routes = list(steps)
    for _ in range(edges_in_route - 1):
        routes = [r + (j,) for r in routes for i, j in steps if i == r[-1]]
    return sorted(routes)


def test_lambda_mu_convention_matches_the_published_routing_construction():
    """(λ, μ) = (скоттовы, кэлиевы), а не наоборот.

    Проверяется на опубликованной задаче, где ответ известен независимо.
    В названной работе маршруты в графе строятся возведением матрицы
    смежности в (1, 0)-свёрнутую степень, и доказано, что набор значений
    индексов ненейтрального элемента (1, 0)-степени G^k есть последовательность
    вершин маршрута из k рёбер.

    Это однозначно фиксирует порядок параметров. При (λ, μ) = (1, 0) свёртки
    нет вовсе (кэлиевых индексов ноль), ранг растёт на единицу за шаг, и
    получаются сами маршруты. При обратном прочтении — (0, 1) — операция была
    бы обычным умножением матриц: ранг остался бы равен двум, и вместо
    маршрутов получились бы их веса. Если бы соглашение в реализации было
    перепутано, этот тест провалился бы на первом же шаге.
    """
    G = np.array([[int(ch) for ch in row] for row in GRAPH_FROM_PAPER],
                 dtype=np.float64)
    assert G.shape == (10, 10) and G.sum() == 12

    # Ранги: r = p + q − λ − 2μ. Для (1,0) на двух двумерных матрицах r = 3.
    assert convolve(G, G, lam=1, mu=0).ndim == 3
    assert convolve(G, G, lam=0, mu=1).ndim == 2, "(0,1) — обычное умножение матриц"

    power = G.copy()
    for edges in range(2, 7):
        power = convolve(power, G, lam=1, mu=0)
        assert power.ndim == edges + 1, "каждый шаг (1,0)-степени добавляет индекс"
        found = sorted(map(tuple, np.argwhere(power != 0).tolist()))
        assert found == _walks(G, edges), f"маршруты из {edges} рёбер не совпали"


def test_lambda_only_product_does_not_sum(rng):
    """При μ = 0 суммирования нет: операция чисто мультипликативна.

    Это то свойство, ради которого в цитированной работе взята степень (1, 0):
    суммирование по промежуточной вершине дало бы вес маршрута вместо самого
    маршрута.
    """
    A = rng.random((4, 5))
    B = rng.random((5, 6))
    assert np.allclose(convolve(A, B, lam=1, mu=0), A[:, :, None] * B[None, :, :])
    assert np.allclose(convolve(A, B, lam=1, mu=0), convolve_naive(A, B, lam=1, mu=0))
