# -*- coding: utf-8 -*-
"""Разбиение работы между узлами.

Проверяется правило выбора оси разбиения. Оно опирается на утверждение из
работы Мунермана и Мунермана (2022): при λ > 0 результат (λ, μ)-свёрнутого
произведения составлен из сечений ориентации скоттовых индексов, каждое из
которых есть произведение соответствующих сечений сомножителей. Отсюда
разбиение по скоттовой оси режет **оба** операнда и не требует репликации,
тогда как разбиение по свободной оси реплицирует второй операнд целиком.
"""
import numpy as np
import pytest

from amdb.exec import axis_kinds, chunk_ranges, concat_chunks, replication_factor, split_axis
from amdb.exec.chunker import FREE, SCOTT
from amdb.ql.planner import ARRAY, EinsumStep, Operand


def binary_step():
    """A[l,s,c] ∗ B[s,c,m] -> [l,s,m]: s — скоттова, c — кэлиева, l и m — свободные."""
    ops = [
        Operand(ARRAY, "A", ("l", "s", "c"), np.ones((2, 3, 4))),
        Operand(ARRAY, "B", ("s", "c", "m"), np.ones((3, 4, 5))),
    ]
    return EinsumStep(ops, ("l", "s", "m"))


def test_axis_kinds_distinguishes_scott_from_free():
    kinds = axis_kinds(binary_step())
    assert kinds == {"l": FREE, "s": SCOTT, "m": FREE}


def test_scott_axis_is_preferred_even_when_shorter():
    """Скоттова ось выбирается, даже если свободная длиннее.

    Здесь m длиннее s (5 против 3), но разбиение по m потребовало бы
    реплицировать A на все узлы, а разбиение по s не требует ничего.
    """
    sizes = {"l": 2, "s": 3, "c": 4, "m": 5}
    assert split_axis(binary_step(), sizes) == "s"


def test_replication_factor_justifies_the_preference():
    step = binary_step()
    assert replication_factor(step, "s") == 0, "скоттова ось режет оба операнда"
    assert replication_factor(step, "l") == 1, "свободная ось реплицирует второй операнд"
    assert replication_factor(step, "m") == 1


def test_falls_back_to_free_axis_without_scott_indices():
    """При λ = 0 скоттовых осей нет — разбиваем по самой длинной свободной."""
    ops = [
        Operand(ARRAY, "A", ("l", "c"), np.ones((2, 4))),
        Operand(ARRAY, "B", ("c", "m"), np.ones((4, 7))),
    ]
    step = EinsumStep(ops, ("l", "m"))
    assert set(axis_kinds(step).values()) == {FREE}
    assert split_axis(step, {"l": 2, "c": 4, "m": 7}) == "m"


def test_returns_none_when_output_is_scalar():
    """Скалярный агрегат: делить нечего, параллелизм только с редукцией."""
    ops = [Operand(ARRAY, "A", ("c",), np.ones(4)), Operand(ARRAY, "B", ("c",), np.ones(4))]
    step = EinsumStep(ops, ())
    assert split_axis(step, {"c": 4}) is None


def test_chunk_ranges_cover_the_whole_axis_without_gaps():
    ranges = chunk_ranges(100, 4)
    assert ranges[0][0] == 0 and ranges[-1][1] == 100
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert sum(b - a for a, b in ranges) == 100


def test_chunk_ranges_never_produce_empty_parts():
    assert chunk_ranges(3, 10) == [(0, 1), (1, 2), (2, 3)]
    with pytest.raises(ValueError):
        chunk_ranges(10, 0)


def test_split_and_reassemble_reproduces_the_whole_result():
    """Разбиение по скоттовой оси даёт независимые подзадачи; склейка точна."""
    rng = np.random.default_rng(0)
    A = rng.random((2, 6, 4))     # [l, s, c]
    B = rng.random((6, 4, 5))     # [s, c, m]
    whole = np.einsum("lsc,scm->lsm", A, B)

    parts = [np.einsum("lsc,scm->lsm", A[:, lo:hi], B[lo:hi]) for lo, hi in chunk_ranges(6, 3)]
    assert np.allclose(concat_chunks(parts, axis_pos=1), whole)
