# -*- coding: utf-8 -*-
"""Многомерная матрица с именованными осями.

Имена осей — не украшение: по ним планировщик автоматически выводит роли
индексов (λ / μ / свободные) при соединении кубов, вместо ручного согласования
порядка осей.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np

from .convolve import build_spec


@dataclass(frozen=True)
class MultidimensionalMatrix:
    """Многомерная матрица: массив + имена измерений по осям."""

    data: np.ndarray
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.asarray(self.data))
        object.__setattr__(self, "axes", tuple(self.axes))
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"имён осей {len(self.axes)}, ранг массива {self.data.ndim}"
            )
        if len(set(self.axes)) != len(self.axes):
            raise ValueError(f"имена осей должны быть уникальны: {self.axes}")

    # -- свойства -----------------------------------------------------------
    @property
    def rank(self) -> int:
        return self.data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def axis_length(self, name: str) -> int:
        return self.data.shape[self.axes.index(name)]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self.data if dtype is None else self.data.astype(dtype)
        return arr.copy() if copy else arr

    def __repr__(self) -> str:
        dims = ", ".join(f"{a}={n}" for a, n in zip(self.axes, self.shape))
        return f"MDM[{dims}] dtype={self.data.dtype}"

    # -- операции над гиперкубом (ТЗ 3.2.3) ---------------------------------
    def transpose(self, order: Sequence[str]) -> "MultidimensionalMatrix":
        """Перестановка осей по именам."""
        order = tuple(order)
        if set(order) != set(self.axes):
            raise ValueError(f"перестановка {order} не соответствует осям {self.axes}")
        perm = [self.axes.index(a) for a in order]
        return MultidimensionalMatrix(self.data.transpose(perm), order)

    def project(
        self, drop: Iterable[str], how: str = "sum"
    ) -> "MultidimensionalMatrix":
        """Проекция: удаление измерений с агрегацией."""
        drop = tuple(drop)
        unknown = set(drop) - set(self.axes)
        if unknown:
            raise KeyError(f"нет таких осей: {sorted(unknown)}")
        fn = {"sum": np.sum, "max": np.max, "min": np.min, "mean": np.mean}[how]
        ax = tuple(self.axes.index(a) for a in drop)
        keep = tuple(a for a in self.axes if a not in drop)
        return MultidimensionalMatrix(fn(self.data, axis=ax), keep)

    def keep_only(self, keep: Sequence[str], how: str = "sum") -> "MultidimensionalMatrix":
        """Оставить только перечисленные оси (в указанном порядке)."""
        result = self.project([a for a in self.axes if a not in keep], how=how)
        return result.transpose(tuple(keep))

    def slice(self, **fixed: Any) -> "MultidimensionalMatrix":
        """Срез: выделение подгиперкуба.

        ``cube.slice(customer=5)`` фиксирует значение (ось исчезает);
        ``cube.slice(month=slice(0, 3))`` или список индексов — ось сохраняется.
        """
        unknown = set(fixed) - set(self.axes)
        if unknown:
            raise KeyError(f"нет таких осей: {sorted(unknown)}")
        # Продвинутая индексация несколькими списками сразу ведёт себя не так,
        # как последовательные срезы, поэтому применяем оси по одной.
        data = self.data
        axes = list(self.axes)
        for name, key in fixed.items():
            pos = axes.index(name)
            idx: list[Any] = [slice(None)] * len(axes)
            idx[pos] = key
            data = data[tuple(idx)]
            if np.isscalar(key) or isinstance(key, (int, np.integer)):
                axes.pop(pos)
        return MultidimensionalMatrix(data, tuple(axes))

    def rename(self, **mapping: str) -> "MultidimensionalMatrix":
        return replace(self, axes=tuple(mapping.get(a, a) for a in self.axes))

    def astype(self, dtype: Any) -> "MultidimensionalMatrix":
        return replace(self, data=self.data.astype(dtype))

    # -- поэлементные операции ----------------------------------------------
    def _elementwise(self, other: Any, op) -> "MultidimensionalMatrix":
        if isinstance(other, MultidimensionalMatrix):
            aligned = align(self, other)
            return MultidimensionalMatrix(op(aligned[0].data, aligned[1].data), aligned[0].axes)
        return MultidimensionalMatrix(op(self.data, other), self.axes)

    def __add__(self, other): return self._elementwise(other, np.add)
    def __sub__(self, other): return self._elementwise(other, np.subtract)
    def __mul__(self, other): return self._elementwise(other, np.multiply)
    def __truediv__(self, other):
        return self._elementwise(other, lambda a, b: np.divide(
            a, b, out=np.zeros_like(np.asarray(a, dtype=np.float64)), where=np.asarray(b) != 0))
    __radd__ = __add__
    __rmul__ = __mul__


def align(
    a: MultidimensionalMatrix, b: MultidimensionalMatrix
) -> tuple[MultidimensionalMatrix, MultidimensionalMatrix]:
    """Приводит два куба к общему набору осей (broadcast по недостающим)."""
    axes = list(a.axes) + [x for x in b.axes if x not in a.axes]

    def expand(m: MultidimensionalMatrix) -> MultidimensionalMatrix:
        m = m.transpose(tuple(x for x in axes if x in m.axes))
        data = m.data
        for i, name in enumerate(axes):
            if name not in m.axes:
                data = np.expand_dims(data, i)
        return MultidimensionalMatrix(data, tuple(axes))

    return expand(a), expand(b)


def convolve_named(
    A: MultidimensionalMatrix,
    B: MultidimensionalMatrix,
    keep: Iterable[str] = (),
    optimize: bool | str | list = True,
) -> MultidimensionalMatrix:
    """(λ, μ)-свёртка с автоматическим выводом ролей индексов.

    Общие оси, попавшие в ``keep``, становятся λ (сохраняются);
    остальные общие оси становятся μ (суммируются).
    """
    keep = set(keep)
    common = [a for a in A.axes if a in B.axes]
    for name in common:
        if A.axis_length(name) != B.axis_length(name):
            raise ValueError(
                f"ось '{name}': {A.axis_length(name)} != {B.axis_length(name)}"
            )
    S = [a for a in common if a in keep]                  # λ
    L = [a for a in A.axes if a not in B.axes]
    M = [a for a in B.axes if a not in A.axes]
    out = tuple(L + S + M)
    spec = build_spec([A.axes, B.axes], out)
    return MultidimensionalMatrix(
        np.einsum(spec, A.data, B.data, optimize=optimize), out
    )


def lambda_mu(A: MultidimensionalMatrix, B: MultidimensionalMatrix,
              keep: Iterable[str] = ()) -> tuple[int, int]:
    """Возвращает (λ, μ) для пары кубов при заданном наборе сохраняемых осей."""
    keep = set(keep)
    common = [a for a in A.axes if a in B.axes]
    lam = sum(1 for a in common if a in keep)
    return lam, len(common) - lam
