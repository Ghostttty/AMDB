# -*- coding: utf-8 -*-
"""Разбиение работы между узлами — основа масштабирования.

Разбивать можно по любой оси, присутствующей в результате: подзадачи выходят
независимыми, и межузловая редукция не нужна — результаты просто склеиваются.
Разбиение по свёртываемой (кэлиевой, μ) оси потребовало бы Allreduce, поэтому
его следует избегать.

Но не все оси результата равноценны по объёму пересылок, и предпочтение здесь
отдаётся **скоттовым (λ) осям** — общим для обоих сомножителей и сохраняемым
в результате:

* разбиение по скоттовой оси режет **оба** операнда: узел получает только свои
  срезы A и B, репликации нет;
* разбиение по свободной оси режет лишь один операнд, а второй приходится
  реплицировать на все узлы целиком.

Основание — утверждение из работы Мунермана и Мунермана (2022): при λ > 0
результат (λ, μ)-свёрнутого произведения составлен из сечений ориентации
(s₁, …, s_λ), каждое из которых есть произведение соответствующих сечений
сомножителей с тем же набором значений скоттовых индексов. Иначе говоря, одно
(λ, μ)-произведение распадается на n^λ независимых (0, μ)-произведений над
сечениями.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

from ..ql.planner import EinsumStep

#: Оценка кратности репликации при разбиении по оси данного типа.
SCOTT = "скоттова (λ)"
FREE = "свободная"


def chunk_ranges(length: int, n_chunks: int) -> list[tuple[int, int]]:
    """Равномерное разбиение диапазона [0, length) на n частей."""
    if n_chunks < 1:
        raise ValueError("число частей должно быть >= 1")
    n_chunks = min(n_chunks, max(length, 1))
    bounds = np.linspace(0, length, n_chunks + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def axis_kinds(step: EinsumStep) -> dict[str, str]:
    """Классифицирует оси результата: скоттовы (общие) и свободные."""
    counts: dict[str, int] = {}
    for o in step.operands:
        for a in set(o.axes):
            counts[a] = counts.get(a, 0) + 1
    return {a: (SCOTT if counts.get(a, 0) > 1 else FREE) for a in step.output}


def split_axis(step: EinsumStep, sizes: dict[str, int]) -> str | None:
    """Выбирает ось для распараллеливания.

    Порядок предпочтения:

    1. **скоттова ось результата** — режет оба операнда, репликации нет;
    2. свободная ось результата — режет один операнд, второй реплицируется;
    3. ничего (None) — тогда параллелизм возможен лишь по кэлиевой оси
       с последующей редукцией.

    Внутри каждой категории берётся самая длинная ось: чем она длиннее, тем
    равномернее делится работа между узлами.
    """
    kinds = axis_kinds(step)
    scott = [a for a, k in kinds.items() if k == SCOTT]
    free = [a for a, k in kinds.items() if k == FREE]
    for group in (scott, free):
        if group:
            return max(group, key=lambda a: sizes.get(a, 0))
    return None


def replication_factor(step: EinsumStep, axis: str) -> int:
    """Сколько операндов придётся реплицировать при разбиении по данной оси.

    Ноль означает, что реплицировать не нужно ничего: ось режет все операнды,
    которые её содержат, а операнды без неё в подзадаче и не участвовали бы.
    """
    return sum(1 for o in step.operands if axis not in o.axes and o.axes)


def iter_slices(axis: str, length: int, n_chunks: int) -> Iterator[tuple[str, slice]]:
    for lo, hi in chunk_ranges(length, n_chunks):
        yield axis, slice(lo, hi)


def concat_chunks(parts: Sequence[np.ndarray], axis_pos: int) -> np.ndarray:
    """Склейка результатов подзадач вдоль оси разбиения."""
    return np.concatenate(list(parts), axis=axis_pos)
