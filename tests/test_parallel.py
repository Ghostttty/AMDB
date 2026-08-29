# -*- coding: utf-8 -*-
"""Параллельное исполнение по скоттовым сечениям.

Тесты порождают дочерние процессы, поэтому держатся отдельным файлом: при
необходимости их можно исключить одним ``--ignore``. Проверяется прежде всего
совпадение с однопоточным путём — ускорение без правильного ответа не нужно.
"""
import numpy as np
import pytest

from amdb.exec.engine import NumpyEngine
from amdb.exec.parallel import ParallelEngine, scott_axes


def test_scott_axes_selects_shared_and_surviving_indices():
    """Годятся для разбиения лишь общие индексы, остающиеся в результате.

    По ним разбиение режет оба сомножителя и репликации не требует; по
    свободной оси пришлось бы копировать второй операнд целиком.
    """
    assert scott_axes("slc,scm->slm") == ["s"]
    assert scott_axes("abc,bcd->abd") == ["b"]
    assert scott_axes("abc,bcd->ad") == []        # общие оси свёрнуты
    assert scott_axes("ab,cd->abcd") == []        # общих осей нет
    assert scott_axes("abc,abd->abcd") == ["a", "b"]


@pytest.mark.parametrize("spec,shape_a,shape_b", [
    ("slc,scm->slm", (6, 5, 4), (6, 4, 7)),
    ("slc,sc->sl", (6, 5, 4), (6, 4)),
    ("abc,bcd->abd", (5, 6, 4), (6, 4, 3)),
    ("abc,abd->abcd", (4, 5, 3), (4, 5, 2)),
])
def test_parallel_matches_single_threaded(spec, shape_a, shape_b):
    rng = np.random.default_rng(0)
    A, B = rng.random(shape_a), rng.random(shape_b)
    reference = NumpyEngine().einsum(spec, A, B)
    engine = ParallelEngine(workers=2, min_cells=0)
    try:
        assert np.allclose(engine.einsum(spec, A, B), reference)
    finally:
        engine.close()


def test_parallel_falls_back_when_it_cannot_split():
    """Отказ должен быть тихим и правильным, а не ошибкой.

    Разбить нечего, если операнд один, если общих сохраняемых осей нет или если
    работа слишком мала, чтобы окупить запуск процессов.
    """
    rng = np.random.default_rng(1)
    engine = ParallelEngine(workers=2, min_cells=0)
    try:
        one = rng.random((4, 5))
        assert np.allclose(engine.einsum("ab->a", one), np.einsum("ab->a", one))

        a, b = rng.random((4, 5)), rng.random((5, 6))
        assert np.allclose(engine.einsum("ab,bc->ac", a, b), a @ b)
        assert not engine._shared, "операнды не должны попадать в общую память"
    finally:
        engine.close()


def test_threshold_keeps_small_work_single_threaded():
    rng = np.random.default_rng(2)
    A, B = rng.random((4, 5, 3)), rng.random((4, 3, 5))
    engine = ParallelEngine(workers=2, min_cells=10**9)
    try:
        assert np.allclose(engine.einsum("slc,scm->slm", A, B),
                           np.einsum("slc,scm->slm", A, B))
        assert engine._pool is None, "пул не должен создаваться ниже порога"
    finally:
        engine.close()


def test_resident_operands_are_published_once():
    """Повторный вызов с тем же массивом не должен копировать его заново."""
    rng = np.random.default_rng(3)
    A, B = rng.random((6, 20, 15)), rng.random((6, 15, 20))
    engine = ParallelEngine(workers=2, min_cells=0, resident=True)
    try:
        engine.einsum("slc,scm->slm", A, B)
        names = {block.name for _, block in engine._shared.values()}
        assert len(names) == 2
        engine.einsum("slc,scm->slm", A, B)
        assert {block.name for _, block in engine._shared.values()} == names
    finally:
        engine.close()


def test_close_releases_shared_memory():
    rng = np.random.default_rng(4)
    A, B = rng.random((6, 20, 15)), rng.random((6, 15, 20))
    engine = ParallelEngine(workers=2, min_cells=0)
    engine.einsum("slc,scm->slm", A, B)
    assert engine._shared
    engine.close()
    assert not engine._shared and engine._pool is None


def test_query_runs_on_the_parallel_engine():
    """Сквозная проверка: тот же ответ, что и на обычном движке."""
    pd = pytest.importorskip("pandas")
    from amdb import Database

    rng = np.random.default_rng(5)
    n, side = 5_000, 30
    frame = pd.DataFrame({"customer": rng.integers(0, side, n),
                          "product": rng.integers(0, side, n),
                          "date": rng.integers(0, side, n),
                          "q": rng.random(n)})
    db = Database()
    db.load_frame(frame, ["customer", "product", "date"], "q", "sales")
    query = "SELECT customer, SUM(q) AS v FROM sales GROUP BY customer"
    expected = db.sql(query).column("v")

    engine = ParallelEngine(workers=2, min_cells=0)
    try:
        got = db.use_engine(engine).sql(query).column("v")
        assert np.allclose(got, expected)
    finally:
        engine.close()
        db.use_engine(NumpyEngine())
