# -*- coding: utf-8 -*-
"""Математическое ядро: (λ, μ)-свёртка против эталона по определению."""
import numpy as np
import pytest

from amdb.core import (
    COOCube,
    MultidimensionalMatrix,
    TooManyIndicesError,
    build_einsum,
    build_spec,
    convolve,
    convolve_naive,
    convolve_named,
    convolve_sparse,
    indicator,
    lambda_mu,
    reduce_axes,
    rollup_matrix,
    running_sum,
    weighted_window,
)


def test_einsum_spec_matches_article():
    """Пример из статьи Симакова: ранг 6 × ранг 9, (2,3)-свёртка."""
    assert build_einsum(6, 9, lam=2, mu=3) == "abcdef,bcdefghij->abcghij"


@pytest.mark.parametrize("rank_a,rank_b,lam,mu,expected", [
    (2, 2, 0, 1, "ab,bc->ac"),          # обычное матричное умножение
    (3, 3, 1, 1, "abc,bcd->abd"),       # (1,1)-свёртка из таблицы 1 статьи
    (2, 2, 0, 2, "ab,ab->"),            # полная свёртка в скаляр
    (1, 1, 1, 0, "a,a->a"),             # поэлементное произведение
])
def test_einsum_spec_cases(rank_a, rank_b, lam, mu, expected):
    assert build_einsum(rank_a, rank_b, lam, mu) == expected


def test_too_many_indices():
    with pytest.raises(TooManyIndicesError):
        build_einsum(30, 30, 0, 0)


def test_invalid_ranks():
    with pytest.raises(ValueError):
        build_einsum(2, 2, lam=2, mu=2)


@pytest.mark.parametrize("lam,mu", [(0, 1), (1, 1), (2, 1), (0, 2), (2, 2)])
def test_convolve_matches_definition(lam, mu):
    """einsum-реализация совпадает с прямым вычислением по определению."""
    rng = np.random.default_rng(lam * 10 + mu)
    k, v = 2, 2
    shape_a = tuple(rng.integers(2, 4, k + lam + mu))
    shape_b = tuple(shape_a[k:]) + tuple(rng.integers(2, 4, v))
    A = rng.random(shape_a)
    B = rng.random(shape_b)
    got = convolve(A, B, lam, mu)
    ref = convolve_naive(A, B, lam, mu)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-12


def test_convolve_rejects_mismatched_axes():
    with pytest.raises(ValueError):
        convolve(np.ones((2, 3)), np.ones((4, 5)), lam=0, mu=1)


def test_named_convolution_infers_roles():
    rng = np.random.default_rng(0)
    A = MultidimensionalMatrix(rng.random((4, 5, 6)), ("customer", "product", "date"))
    B = MultidimensionalMatrix(rng.random((5, 6, 3)), ("product", "date", "store"))
    keep = {"product"}
    assert lambda_mu(A, B, keep) == (1, 1)
    out = convolve_named(A, B, keep=keep)
    assert out.axes == ("customer", "product", "store")
    ref = np.einsum("cpd,pds->cps", A.data, B.data)
    assert np.abs(out.data - ref).max() < 1e-12


def test_named_convolution_checks_axis_lengths():
    A = MultidimensionalMatrix(np.ones((2, 3)), ("a", "b"))
    B = MultidimensionalMatrix(np.ones((4, 5)), ("b", "c"))
    with pytest.raises(ValueError, match="ось 'b'"):
        convolve_named(A, B)


def test_mdm_validation():
    with pytest.raises(ValueError):
        MultidimensionalMatrix(np.ones((2, 3)), ("a",))
    with pytest.raises(ValueError):
        MultidimensionalMatrix(np.ones((2, 3)), ("a", "a"))


def test_slice_project_transpose():
    rng = np.random.default_rng(1)
    m = MultidimensionalMatrix(rng.random((3, 4, 5)), ("a", "b", "c"))
    assert m.slice(a=1).axes == ("b", "c")
    assert m.slice(a=slice(0, 2)).shape == (2, 4, 5)
    assert m.slice(b=np.array([0, 2])).shape == (3, 2, 5)
    assert m.project(["b"]).axes == ("a", "c")
    assert np.allclose(m.project(["b"]).data, m.data.sum(axis=1))
    assert m.transpose(("c", "a", "b")).shape == (5, 3, 4)
    assert m.keep_only(("c", "a")).axes == ("c", "a")


def test_elementwise_broadcasts_by_axis_name():
    a = MultidimensionalMatrix(np.ones((2, 3)), ("x", "y"))
    b = MultidimensionalMatrix(np.full(3, 2.0), ("y",))
    assert np.allclose((a * b).data, 2.0)
    assert (a * b).axes == ("x", "y")


def test_running_sum_is_cumulative():
    rng = np.random.default_rng(2)
    m = MultidimensionalMatrix(rng.random((3, 5)), ("customer", "month"))
    assert np.allclose(running_sum(m, "month").data, np.cumsum(m.data, axis=1))
    windowed = running_sum(m, "month", window=2).data
    expected = m.data.copy()
    expected[:, 1:] += m.data[:, :-1]
    assert np.allclose(windowed, expected)


def test_weighted_window_rejects_wrong_shape():
    m = MultidimensionalMatrix(np.ones((2, 3)), ("a", "b"))
    with pytest.raises(ValueError):
        weighted_window(m, "b", np.ones((2, 2)))


def test_rollup_matrix_is_transition_matrix():
    m = rollup_matrix(np.array([0, 0, 1, 1, 2]), 3)
    assert m.shape == (5, 3)
    assert np.allclose(m.sum(axis=1), 1.0)
    data = MultidimensionalMatrix(np.arange(10.0).reshape(2, 5), ("c", "d"))
    rolled = np.einsum("cd,dp->cp", data.data, m)
    assert np.allclose(rolled[0], [0 + 1, 2 + 3, 4])


def test_reduce_and_indicator():
    m = MultidimensionalMatrix(np.array([[1.0, 0.0], [3.0, 4.0]]), ("a", "b"))
    assert np.allclose(reduce_axes(m, ["b"], "max").data, [1.0, 4.0])
    assert np.allclose(indicator(m).data, [[1, 0], [1, 1]])


# --- разреженное ядро ------------------------------------------------------
def _random_sparse(shape, axes, density, seed):
    rng = np.random.default_rng(seed)
    dense = (rng.random(shape) < density) * rng.random(shape)
    return COOCube.from_dense(MultidimensionalMatrix(dense, axes)), dense


def test_coo_roundtrip_and_stats():
    coo, dense = _random_sparse((4, 5, 6), ("a", "b", "c"), 0.3, 7)
    assert np.allclose(coo.to_dense().data, dense)
    assert coo.nnz == np.count_nonzero(dense)
    assert 0 < coo.fill_factor < 1


def test_coo_coalesce_sums_duplicates():
    coo = COOCube(np.array([[0, 0], [0, 0], [1, 1]]), np.array([1.0, 2.0, 5.0]),
                  ("a", "b"), (2, 2))
    merged = coo.coalesce()
    assert merged.nnz == 2
    assert np.allclose(merged.to_dense().data, [[3.0, 0.0], [0.0, 5.0]])


def test_coo_project_and_mask():
    coo, dense = _random_sparse((4, 5), ("a", "b"), 0.5, 3)
    assert np.allclose(coo.project(["b"]).to_dense().data, dense.sum(axis=1))
    mask = np.array([1, 0, 1, 0, 1])
    assert np.allclose(coo.mask("b", mask).to_dense().data, dense * mask)


def test_sparse_dense_convolution_matches_dense():
    coo, dense = _random_sparse((4, 5, 6), ("c", "p", "d"), 0.2, 5)
    rng = np.random.default_rng(9)
    B = MultidimensionalMatrix(rng.random((5, 6, 3)), ("p", "d", "s"))
    got = convolve_sparse(coo, B, keep={"p"})
    got = got.to_dense() if isinstance(got, COOCube) else got
    ref = convolve_named(MultidimensionalMatrix(dense, ("c", "p", "d")), B, keep={"p"})
    assert np.abs(got.transpose(ref.axes).data - ref.data).max() < 1e-12


def test_sparse_sparse_convolution_matches_dense():
    a, da = _random_sparse((4, 5, 6), ("c", "p", "d"), 0.25, 1)
    b, dbb = _random_sparse((5, 6, 3), ("p", "d", "s"), 0.25, 2)
    got = convolve_sparse(a, b, keep={"p"})
    got = got.to_dense() if isinstance(got, COOCube) else got
    ref = convolve_named(MultidimensionalMatrix(da, ("c", "p", "d")),
                         MultidimensionalMatrix(dbb, ("p", "d", "s")), keep={"p"})
    assert np.abs(got.transpose(ref.axes).data - ref.data).max() < 1e-12


def test_sparse_convolution_with_empty_operand():
    empty = COOCube(np.zeros((0, 2), np.int64), np.zeros(0), ("a", "b"), (3, 4))
    other = MultidimensionalMatrix(np.ones((4, 2)), ("b", "c"))
    out = convolve_sparse(empty, other)
    dense = out.to_dense() if isinstance(out, COOCube) else out
    assert np.allclose(dense.data, 0.0)


def test_build_spec_named():
    assert build_spec([("c", "p", "d"), ("p",), ("d", "m")], ("c", "m")) == "abc,b,cd->ad"


def test_batched_matmul_path_matches_einsum_for_all_ranks():
    """Быстрый путь при λ >= 1 обязан совпадать с einsum на любых рангах.

    Путь обходит einsum ради BLAS (einsum не отправляет туда пакетную свёртку),
    поэтому расхождение здесь означало бы тихо неверные ответы на всех запросах
    с соединением — то есть на большинстве.
    """
    from amdb.core.convolve import _batched_matmul, build_einsum

    rng = np.random.default_rng(0)
    checked = 0
    for lam in (1, 2, 3):
        for mu in (1, 2):
            for k in (0, 1, 2):
                for v in (0, 1, 2):
                    rank_a, rank_b = k + lam + mu, lam + mu + v
                    if rank_a > 6 or rank_b > 6:
                        continue
                    sizes = [int(rng.integers(2, 6)) for _ in range(k + lam + mu + v)]
                    L, S = sizes[:k], sizes[k:k + lam]
                    C, M = sizes[k + lam:k + lam + mu], sizes[k + lam + mu:]
                    A = rng.random(tuple(L + S + C))
                    B = rng.random(tuple(S + C + M))
                    ref = np.einsum(build_einsum(rank_a, rank_b, lam, mu), A, B)
                    got = _batched_matmul(A, B, lam, mu)
                    assert got is not None, f"путь отказал при λ={lam}, μ={mu}"
                    assert got.shape == ref.shape
                    assert np.allclose(got, ref)
                    checked += 1
    assert checked >= 40, "перебор форм оказался слишком узким"


def test_batched_matmul_path_is_actually_taken():
    """Порог должен пропускать крупные λ-свёртки в matmul, мелкие — в einsum."""
    from importlib import import_module
    from unittest.mock import patch

    # import_module, а не from-import: пакет amdb.core экспортирует функцию
    # convolve, и обычный импорт вернул бы её вместо модуля.
    conv_mod = import_module("amdb.core.convolve")

    big_a, big_b = np.ones((20, 20, 20)), np.ones((20, 20, 20))
    with patch.object(conv_mod, "_batched_matmul",
                      wraps=conv_mod._batched_matmul) as spy:
        conv_mod.convolve(big_a, big_b, lam=1, mu=1)
        assert spy.call_count == 1, "крупная λ-свёртка должна идти пакетным gemm"
        conv_mod.convolve(np.ones((2, 3, 4)), np.ones((3, 4, 2)), lam=1, mu=1)
        assert spy.call_count == 1, "мелкая свёртка не окупает перестановок осей"
        conv_mod.convolve(big_a, big_b, lam=0, mu=2)
        assert spy.call_count == 1, "при λ = 0 einsum уходит в BLAS сам"


def test_batched_matmul_agrees_with_naive_definition():
    """Сверка с прямой реализацией по определению Соколова, а не с einsum."""
    rng = np.random.default_rng(7)
    A = rng.random((3, 4, 5))
    B = rng.random((4, 5, 3))
    assert np.allclose(convolve(A, B, lam=1, mu=1), convolve_naive(A, B, lam=1, mu=1))


def test_sparse_cube_supports_rank_zero_projection():
    """Проекция разреженного куба на скаляр не должна падать.

    Проталкивание проекций (§3.2 статьи) сворачивает частные оси операнда
    заранее. У скалярного агрегата без GROUP BY частными оказываются **все**
    оси, и проекция даёт куб ранга 0. Массив координат при этом имеет форму
    (nnz, 0), из которой число строк вывести нельзя, — на этом реализация и
    ломалась, но только на разреженном пути и только без GROUP BY.
    """
    rng = np.random.default_rng(0)
    dense = MultidimensionalMatrix((rng.random((6, 5, 4)) < 0.3) * rng.random((6, 5, 4)),
                                   ("a", "b", "c"))
    sparse = COOCube.from_dense(dense)

    scalar = sparse.project({"a", "b", "c"})
    assert scalar.rank == 0 and scalar.axes == ()
    assert scalar.coords.shape == (scalar.nnz, 0)
    assert float(scalar.to_dense().data) == pytest.approx(float(dense.data.sum()))

    partial = sparse.project({"b"})
    assert partial.axes == ("a", "c")
    assert np.allclose(partial.to_dense().data, dense.data.sum(axis=1))
