# -*- coding: utf-8 -*-
"""Операции над гиперкубом, не выражаемые через einsum.

SUM и COUNT линейны и исполняются свёрткой. MIN/MAX и оконные функции с
не-линейным ядром требуют отдельных векторизованных редукций — это ограничение
многомерно-матричной модели, а не реализации.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .mdm import MultidimensionalMatrix

_REDUCERS = {
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "prod": np.prod,
}


def reduce_axes(
    cube: MultidimensionalMatrix, axes: Iterable[str], how: str = "max"
) -> MultidimensionalMatrix:
    """Редукция по осям (MIN/MAX/MEAN и т.п.)."""
    axes = tuple(axes)
    if how not in _REDUCERS:
        raise ValueError(f"неизвестная редукция '{how}'")
    if not axes:
        return cube
    unknown = set(axes) - set(cube.axes)
    if unknown:
        raise KeyError(f"нет таких осей: {sorted(unknown)}")
    pos = tuple(cube.axes.index(a) for a in axes)
    keep = tuple(a for a in cube.axes if a not in axes)
    return MultidimensionalMatrix(_REDUCERS[how](cube.data, axis=pos), keep)


def internal_convolution(matrix, axis1: int | str, axis2: int | str):
    """Внутренняя свёртка многомерной матрицы по паре собственных индексов.

    Обобщение следа: суммирование по диагонали двух осей одинаковой длины,
    ранг результата на 2 меньше исходного. У Соколова это самостоятельная
    операция алгебры, не сводимая к (λ, μ)-произведению двух матриц: здесь
    свёртываются индексы одной и той же матрицы.

        b[…] = Σ_i a[…, i, …, i, …]

    Принимает как ``ndarray`` (оси задаются номерами), так и
    :class:`MultidimensionalMatrix` (оси задаются именами).
    """
    named = isinstance(matrix, MultidimensionalMatrix)
    data = matrix.data if named else np.asarray(matrix)
    if named:
        i1 = matrix.axes.index(axis1) if isinstance(axis1, str) else int(axis1)
        i2 = matrix.axes.index(axis2) if isinstance(axis2, str) else int(axis2)
    else:
        i1, i2 = int(axis1), int(axis2)
    if i1 == i2:
        raise ValueError("нужны две различные оси")
    if data.shape[i1] != data.shape[i2]:
        raise ValueError(
            f"свёртка по паре индексов требует осей одинаковой длины: "
            f"{data.shape[i1]} и {data.shape[i2]}"
        )
    # np.diagonal переносит диагональ в конец — суммируем именно её.
    out = np.diagonal(data, axis1=i1, axis2=i2).sum(axis=-1)
    if not named:
        return out
    keep = tuple(a for i, a in enumerate(matrix.axes) if i not in (i1, i2))
    return MultidimensionalMatrix(out, keep)


def indicator(cube: MultidimensionalMatrix, dtype=np.float32) -> MultidimensionalMatrix:
    """Индикаторный куб: 1 там, где ячейка непуста. Основа для COUNT."""
    return MultidimensionalMatrix((cube.data != 0).astype(dtype), cube.axes)


def rollup_matrix(child_ordinals: np.ndarray, n_parent: int,
                  dtype=np.float32) -> np.ndarray:
    """Матрица перехода [child × parent] для иерархии измерений.

    ``child_ordinals[i]`` — индекс родителя для i-го значения дочернего измерения.
    Благодаря такому представлению ROLLUP становится обычной (0,1)-свёрткой.
    """
    child_ordinals = np.asarray(child_ordinals, dtype=np.int64)
    if child_ordinals.ndim != 1:
        raise ValueError("child_ordinals должен быть одномерным")
    if child_ordinals.size and (child_ordinals.min() < 0 or child_ordinals.max() >= n_parent):
        raise ValueError("ординал родителя вне диапазона")
    m = np.zeros((child_ordinals.size, n_parent), dtype=dtype)
    m[np.arange(child_ordinals.size), child_ordinals] = 1
    return m


def rollup_reduce(
    cube: MultidimensionalMatrix,
    axis: str,
    child_ordinals: np.ndarray,
    n_parent: int,
    parent_axis: str,
    how: str = "max",
) -> MultidimensionalMatrix:
    """Сегментная редукция по иерархии для не-линейных агрегатов.

    Для SUM/COUNT вместо этого используется умножение на матрицу перехода.
    """
    pos = cube.axes.index(axis)
    moved = np.moveaxis(cube.data, pos, 0)
    fill = -np.inf if how == "max" else np.inf
    out = np.full((n_parent,) + moved.shape[1:], fill, dtype=np.float64)
    fn = np.maximum if how == "max" else np.minimum
    for child, parent in enumerate(np.asarray(child_ordinals)):
        out[parent] = fn(out[parent], moved[child])
    out[~np.isfinite(out)] = 0
    axes = (parent_axis,) + tuple(a for a in cube.axes if a != axis)
    return MultidimensionalMatrix(out.astype(cube.dtype), axes)


def masked(
    cube: MultidimensionalMatrix, masks: dict[str, np.ndarray]
) -> MultidimensionalMatrix:
    """Применяет 0/1-маски срезом (а не умножением).

    Для MIN/MAX умножение на ноль некорректно — ноль стал бы ложным минимумом,
    поэтому фильтрация выполняется выборкой разрешённых индексов.
    """
    out = cube
    for name, mask in masks.items():
        if name not in out.axes:
            continue
        idx = np.flatnonzero(np.asarray(mask))
        out = out.slice(**{name: idx})
    return out


def running_sum(
    cube: MultidimensionalMatrix, axis: str, window: int | None = None
) -> MultidimensionalMatrix:
    """Оконная накопительная сумма вдоль оси.

    Реализована умножением на нижнетреугольную (или ленточную) матрицу — то есть
    снова свёрткой. ``window=None`` даёт UNBOUNDED PRECEDING, целое значение —
    скользящее окно указанной ширины.
    """
    n = cube.axis_length(axis)
    ones = np.ones((n, n), dtype=np.float64)
    # Матрица окна индексируется как [from, to]: единица при from <= to.
    tri = np.triu(ones)
    if window is not None:
        if window < 1:
            raise ValueError("ширина окна должна быть >= 1")
        tri = tri - np.triu(ones, window)
    return weighted_window(cube, axis, tri)


def weighted_window(
    cube: MultidimensionalMatrix, axis: str, weights: np.ndarray
) -> MultidimensionalMatrix:
    """Произвольное взвешенное окно вдоль оси: матрица [from × to]."""
    weights = np.asarray(weights)
    n = cube.axis_length(axis)
    if weights.shape != (n, n):
        raise ValueError(f"матрица окна должна быть {n}×{n}, получена {weights.shape}")
    pos = cube.axes.index(axis)
    moved = np.moveaxis(cube.data, pos, -1)
    out = np.einsum("...m,mn->...n", moved, weights.astype(np.float64))
    return MultidimensionalMatrix(
        np.moveaxis(out, -1, pos).astype(np.result_type(cube.dtype, np.float32)),
        cube.axes,
    )


def elementwise(
    a: MultidimensionalMatrix, b: MultidimensionalMatrix, op: str
) -> MultidimensionalMatrix:
    """Поэлементные операции над гиперкубами с выравниванием осей."""
    ops = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
        "/": lambda x, y: x / y,
    }
    if op not in ops:
        raise ValueError(f"неизвестная операция '{op}'")
    return ops[op](a, b)


def slice_by_labels(
    cube: MultidimensionalMatrix, dimensions: dict, **labels: Sequence
) -> MultidimensionalMatrix:
    """Срез по значениям измерений, а не по индексам осей."""
    fixed = {}
    for axis, values in labels.items():
        dim = dimensions[axis]
        if isinstance(values, (list, tuple, np.ndarray)):
            fixed[axis] = np.array([dim.ordinal(v) for v in values], dtype=np.int64)
        else:
            fixed[axis] = dim.ordinal(values)
    return cube.slice(**fixed)
