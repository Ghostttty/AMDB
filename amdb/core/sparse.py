# -*- coding: utf-8 -*-
"""Разреженный гиперкуб (COO) и гибридные свёртки.

``einsum`` работает только с плотными массивами, поэтому разреженный путь —
собственный код. Это самый рискованный элемент ядра (риск R1 из техпроекта):
без него плотный гиперкуб взрывается комбинаторно на реальных данных.

Стратегия свёртки:
  * COO × dense  — прямая выборка значений плотного операнда по координатам;
  * COO × COO    — хеш-соединение по общим осям с последующей группировкой;
  * если результат достаточно плотный, он материализуется в ``ndarray``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .mdm import MultidimensionalMatrix


@dataclass
class COOCube:
    """Разреженный гиперкуб в координатном формате.

    coords: массив [nnz × rank] целочисленных координат;
    values: массив [nnz] значений;
    axes:   имена измерений; shape: длины осей.
    """

    coords: np.ndarray
    values: np.ndarray
    axes: tuple[str, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values).reshape(-1)
        coords = np.asarray(self.coords, dtype=np.int64)
        if len(self.axes):
            self.coords = coords.reshape(-1, len(self.axes))
        else:
            # Куб ранга 0 — скаляр: координат нет вовсе. reshape(-1, 0) вывести
            # число строк не может, поэтому оно берётся из значений.
            self.coords = coords.reshape(self.values.shape[0], 0)
        self.axes = tuple(self.axes)
        self.shape = tuple(int(s) for s in self.shape)
        if self.coords.shape[0] != self.values.shape[0]:
            raise ValueError("число координат и значений не совпадает")
        if len(self.axes) != len(self.shape):
            raise ValueError("число имён осей и длин осей не совпадает")

    # -- свойства -----------------------------------------------------------
    @property
    def nnz(self) -> int:
        return int(self.values.size)

    @property
    def rank(self) -> int:
        return len(self.axes)

    @property
    def total_cells(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def fill_factor(self) -> float:
        return self.nnz / max(self.total_cells, 1)

    def __repr__(self) -> str:
        dims = ", ".join(f"{a}={n}" for a, n in zip(self.axes, self.shape))
        return f"COOCube[{dims}] nnz={self.nnz} fill={self.fill_factor:.2%}"

    # -- конструкторы -------------------------------------------------------
    @classmethod
    def from_dense(cls, cube: MultidimensionalMatrix) -> "COOCube":
        nz = np.nonzero(cube.data)
        return cls(np.stack(nz, axis=1), cube.data[nz], cube.axes, cube.data.shape)

    def to_dense(self) -> MultidimensionalMatrix:
        out = np.zeros(self.shape, dtype=self.values.dtype)
        if self.nnz:
            flat = _key(self.coords, self.shape)
            np.add.at(out.reshape(-1), flat, self.values)
        return MultidimensionalMatrix(out, self.axes)

    # -- операции -----------------------------------------------------------
    def coalesce(self) -> "COOCube":
        """Складывает дубликаты координат и отбрасывает нули."""
        if not self.nnz:
            return self
        flat = _key(self.coords, self.shape)
        order = np.argsort(flat, kind="stable")
        flat, vals = flat[order], self.values[order]
        uniq, start = np.unique(flat, return_index=True)
        summed = np.add.reduceat(vals, start)
        keep = summed != 0
        uniq, summed = uniq[keep], summed[keep]
        if self.rank:
            coords = np.stack(np.unravel_index(uniq, self.shape), axis=1)
        else:
            # Скаляр: все слагаемые попадают в единственную ячейку.
            coords = np.zeros((uniq.size, 0), dtype=np.int64)
        return COOCube(coords, summed, self.axes, self.shape)

    def project(self, drop: Iterable[str]) -> "COOCube":
        """Проекция с суммированием — без материализации плотного куба."""
        drop = set(drop)
        keep = [a for a in self.axes if a not in drop]
        pos = [self.axes.index(a) for a in keep]
        shape = tuple(self.shape[i] for i in pos)
        return COOCube(self.coords[:, pos], self.values, tuple(keep), shape).coalesce()

    def mask(self, axis: str, allowed: np.ndarray) -> "COOCube":
        """Фильтрация по 0/1-маске вдоль оси без разрежения в плотный вид."""
        pos = self.axes.index(axis)
        keep = np.asarray(allowed, dtype=bool)[self.coords[:, pos]]
        return COOCube(self.coords[keep], self.values[keep], self.axes, self.shape)


def _key(coords: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    if coords.shape[1] == 0:
        return np.zeros(coords.shape[0], dtype=np.int64)
    return np.ravel_multi_index(tuple(coords.T), tuple(shape))


def convolve_sparse(
    A: COOCube,
    B: COOCube | MultidimensionalMatrix,
    keep: Iterable[str] = (),
    densify_threshold: float = 0.25,
) -> COOCube | MultidimensionalMatrix:
    """(λ, μ)-свёртка с разреженным левым операндом.

    ``keep`` задаёт общие оси, которые остаются в результате (λ); прочие общие
    оси суммируются (μ). Результат материализуется в плотный вид, если его
    фактор заполнения превышает ``densify_threshold``.
    """
    keep = set(keep)
    if isinstance(B, MultidimensionalMatrix):
        result = _convolve_sparse_dense(A, B, keep)
    else:
        result = _convolve_sparse_sparse(A, B, keep)
    if result.fill_factor >= densify_threshold:
        return result.to_dense()
    return result


def _out_axes(A_axes, B_axes, keep) -> list[str]:
    common = [a for a in A_axes if a in B_axes]
    L = [a for a in A_axes if a not in B_axes]
    S = [a for a in common if a in keep]
    M = [a for a in B_axes if a not in A_axes]
    return L + S + M


def _convolve_sparse_dense(
    A: COOCube, B: MultidimensionalMatrix, keep: set[str]
) -> COOCube:
    common = [a for a in A.axes if a in B.axes]
    for name in common:
        if A.shape[A.axes.index(name)] != B.axis_length(name):
            raise ValueError(f"ось '{name}': несогласованные длины")
    M = [a for a in B.axes if a not in A.axes]
    out_axes = _out_axes(A.axes, B.axes, keep)
    if not A.nnz:
        shape = tuple(_len(a, A, B) for a in out_axes)
        return COOCube(np.zeros((0, len(out_axes)), np.int64), np.zeros(0), tuple(out_axes), shape)

    # Значения B, отвечающие каждой ненулевой ячейке A: [nnz × |M|]
    Bt = B.transpose(tuple(common + M))
    idx = tuple(A.coords[:, A.axes.index(a)] for a in common)
    gathered = Bt.data[idx] if common else np.broadcast_to(Bt.data, (A.nnz,) + Bt.data.shape)
    gathered = gathered.reshape(A.nnz, -1)
    products = gathered * A.values[:, None]

    m_shape = tuple(B.axis_length(a) for a in M)
    left_axes = [a for a in out_axes if a not in M]
    left_pos = [A.axes.index(a) for a in left_axes]
    left_coords = A.coords[:, left_pos]
    n_m = int(np.prod(m_shape)) if m_shape else 1

    coords = np.repeat(left_coords, n_m, axis=0)
    if m_shape:
        m_grid = np.stack(np.unravel_index(np.arange(n_m), m_shape), axis=1)
        m_coords = np.tile(m_grid, (A.nnz, 1))
        coords = np.concatenate([coords, m_coords], axis=1)
    values = products.reshape(-1)

    shape = tuple(_len(a, A, B) for a in out_axes)
    order = [out_axes.index(a) for a in (left_axes + M)]
    inv = np.argsort(order)
    return COOCube(coords[:, inv], values, tuple(out_axes), shape).coalesce()


def _convolve_sparse_sparse(A: COOCube, B: COOCube, keep: set[str]) -> COOCube:
    common = [a for a in A.axes if a in B.axes]
    out_axes = _out_axes(A.axes, B.axes, keep)
    shape = tuple(_len(a, A, B) for a in out_axes)
    if not A.nnz or not B.nnz:
        return COOCube(np.zeros((0, len(out_axes)), np.int64), np.zeros(0), tuple(out_axes), shape)

    common_shape = [A.shape[A.axes.index(a)] for a in common]
    ka = _key(A.coords[:, [A.axes.index(a) for a in common]], common_shape)
    kb = _key(B.coords[:, [B.axes.index(a) for a in common]], common_shape)

    # Хеш-соединение по общим осям: для каждого ключа — декартово произведение.
    oa, ob = np.argsort(ka, kind="stable"), np.argsort(kb, kind="stable")
    ka_s, kb_s = ka[oa], kb[ob]
    uniq = np.intersect1d(ka_s, kb_s)
    if uniq.size == 0:
        return COOCube(np.zeros((0, len(out_axes)), np.int64), np.zeros(0), tuple(out_axes), shape)

    a_start, a_end = np.searchsorted(ka_s, uniq, "left"), np.searchsorted(ka_s, uniq, "right")
    b_start, b_end = np.searchsorted(kb_s, uniq, "left"), np.searchsorted(kb_s, uniq, "right")
    a_cnt, b_cnt = a_end - a_start, b_end - b_start
    pair_counts = a_cnt * b_cnt
    total = int(pair_counts.sum())

    a_rows = np.empty(total, dtype=np.int64)
    b_rows = np.empty(total, dtype=np.int64)
    off = 0
    for i in range(uniq.size):
        na, nb = int(a_cnt[i]), int(b_cnt[i])
        if not na or not nb:
            continue
        ai = oa[a_start[i]:a_end[i]]
        bi = ob[b_start[i]:b_end[i]]
        n = na * nb
        a_rows[off:off + n] = np.repeat(ai, nb)
        b_rows[off:off + n] = np.tile(bi, na)
        off += n
    a_rows, b_rows = a_rows[:off], b_rows[:off]

    values = A.values[a_rows] * B.values[b_rows]
    cols = []
    for name in out_axes:
        if name in A.axes:
            cols.append(A.coords[a_rows, A.axes.index(name)])
        else:
            cols.append(B.coords[b_rows, B.axes.index(name)])
    coords = np.stack(cols, axis=1) if cols else np.zeros((off, 0), np.int64)
    return COOCube(coords, values, tuple(out_axes), shape).coalesce()


def _len(name: str, A: COOCube, B) -> int:
    if name in A.axes:
        return A.shape[A.axes.index(name)]
    if isinstance(B, COOCube):
        return B.shape[B.axes.index(name)]
    return B.axis_length(name)
