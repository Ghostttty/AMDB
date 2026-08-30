# -*- coding: utf-8 -*-
"""Загрузка реляционных данных в гиперкубы.

Ключевой приём — векторизованная свёртка координат в линейный индекс и
``np.bincount`` вместо построчного цикла: 500 000 строк превращаются в
гиперкуб 100×100×100 за ~30 мс.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.mdm import MultidimensionalMatrix
from ..core.sparse import COOCube
from .catalog import Catalog, Cube
from .dimension import Dimension, Hierarchy, isna
from .policy import DENSE, SPARSE_COO, choose_layout, estimate_bytes

try:  # pragma: no cover
    import pandas as pd
    HAVE_PANDAS = True
except Exception:  # pragma: no cover
    pd = None
    HAVE_PANDAS = False


def _column(frame: Any, name: str) -> np.ndarray:
    """Столбец из DataFrame или из dict[str, sequence]."""
    if HAVE_PANDAS and isinstance(frame, pd.DataFrame):
        if name not in frame.columns:
            raise KeyError(f"нет столбца '{name}'; есть: {list(frame.columns)}")
        return frame[name].to_numpy()
    if name not in frame:
        raise KeyError(f"нет столбца '{name}'; есть: {list(frame)}")
    return np.asarray(frame[name])


def _encode(dim: "Dimension", col: np.ndarray) -> np.ndarray:
    """Ординалы столбца; отсутствующие значения получают ординал NULL.

    Кодируются они отдельно, чтобы непустая часть столбца шла быстрым путём
    двоичного поиска: словарь с NULL разнороден по типу и на общий путь не
    ложится.
    """
    missing = isna(col)
    if not missing.any():
        return dim.encode(col)
    out = np.empty(len(col), dtype=np.int64)
    out[~missing] = dim.encode(col[~missing])
    out[missing] = dim.ensure_null()
    return out


def _nrows(frame: Any, cols: Sequence[str]) -> int:
    return len(_column(frame, cols[0])) if cols else 0


def load_fact(
    catalog: Catalog,
    frame: Any,
    dim_cols: Sequence[str],
    measure: str,
    cube_name: str | None = None,
    measure_col: str | None = None,
    agg: str = "sum",
    layout: str | None = None,
    dtype: Any = np.float64,
    ordered_dims: Iterable[str] = (),
    with_count: bool = True,
) -> Cube:
    """Таблица фактов -> гиперкуб.

    Дубликаты координат агрегируются (SUM). Измерения создаются или дополняются
    в каталоге; словари append-only, поэтому существующие кубы не ломаются.

    ``with_count`` создаёт спутниковый счётный куб ``<имя>__count``: без него
    COUNT считает непустые ячейки, а не исходные строки, и AVG получается
    средним по ячейкам, а не по записям.
    """
    dim_cols = list(dim_cols)
    cube_name = cube_name or measure
    measure_col = measure_col or measure
    ordered = set(ordered_dims)

    dims: list[Dimension] = []
    columns: list[np.ndarray] = []
    for c in dim_cols:
        col = _column(frame, c)
        missing = isna(col)
        uniq = np.unique(col[~missing] if missing.any() else col)
        d = catalog.ensure_dimension(c, uniq.tolist(), ordered=c in ordered)
        if missing.any():
            d.ensure_null()
        dims.append(d)
        columns.append(col)

    shape = tuple(len(d) for d in dims)
    values = np.asarray(_column(frame, measure_col), dtype=np.float64)
    # NULL в мере — не значение, а его отсутствие. Нейтральный по ⊕ нуль
    # оставляет SUM неизменной, а исключение такой строки из счётного куба
    # даёт COUNT и AVG ровно по правилам SQL (см. §1 статьи).
    absent = np.isnan(values)
    if absent.any():
        values = np.where(absent, 0.0, values)
    coords = np.stack(
        [_encode(d, col) for d, col in zip(dims, columns)], axis=1
    ) if dim_cols else np.zeros((len(values), 0), dtype=np.int64)

    nnz_estimate = int(np.unique(
        np.ravel_multi_index(tuple(coords.T), shape) if dim_cols else np.zeros(1)
    ).size)
    chosen = layout or choose_layout(nnz_estimate, shape, np.dtype(dtype).itemsize)

    if chosen == DENSE:
        cells = int(np.prod(shape)) if shape else 1
        if estimate_bytes(shape, np.dtype(dtype).itemsize) > 2**31:
            chosen = SPARSE_COO
        else:
            flat = (np.ravel_multi_index(tuple(coords.T), shape)
                    if dim_cols else np.zeros(len(values), dtype=np.int64))
            if agg == "sum":
                acc = np.bincount(flat, weights=values, minlength=cells)
            elif agg in ("max", "min"):
                acc = np.full(cells, -np.inf if agg == "max" else np.inf)
                fn = np.maximum if agg == "max" else np.minimum
                fn.at(acc, flat, values)
                acc[~np.isfinite(acc)] = 0
            elif agg == "count":
                acc = np.bincount(flat, minlength=cells).astype(np.float64)
            elif agg == "mean":
                s = np.bincount(flat, weights=values, minlength=cells)
                n = np.bincount(flat, minlength=cells)
                acc = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
            else:
                raise ValueError(f"неизвестная агрегация '{agg}'")
            matrix: Any = MultidimensionalMatrix(
                acc.reshape(shape).astype(dtype), tuple(dim_cols)
            )
            cube = catalog.add_cube(Cube(cube_name, measure, matrix, DENSE, agg))
            if with_count:
                counts = np.bincount(
                    flat, weights=None if not absent.any() else (~absent).astype(np.float64),
                    minlength=cells).astype(dtype)
                catalog.add_cube(Cube(
                    f"{cube_name}__count", f"{cube_name}__count",
                    MultidimensionalMatrix(counts.reshape(shape), tuple(dim_cols)),
                    DENSE, "sum"))
            return cube

    # Разреженный путь: плотный массив не материализуется вовсе.
    matrix = COOCube(coords, values.astype(dtype), tuple(dim_cols), shape).coalesce()
    cube = catalog.add_cube(Cube(cube_name, measure, matrix, SPARSE_COO, agg))
    if with_count:
        counter = COOCube(coords, (~absent).astype(dtype),
                          tuple(dim_cols), shape).coalesce()
        catalog.add_cube(Cube(f"{cube_name}__count", f"{cube_name}__count",
                              counter, SPARSE_COO, "sum"))
    return cube


def load_dimension_table(
    catalog: Catalog,
    frame: Any,
    key_col: str,
    dim_name: str | None = None,
    attributes: Sequence[str] = (),
    measures: Sequence[str] = (),
    ordered: bool = False,
    dtype: Any = np.float64,
) -> Dimension:
    """Справочник -> измерение с атрибутами (+ кубы ранга 1 для его мер).

    Например, таблица products(id, category, price) даёт измерение ``product``
    с атрибутом ``category`` и куб ``price[product]``.
    """
    dim_name = dim_name or key_col
    keys = _column(frame, key_col)
    dim = catalog.ensure_dimension(dim_name, np.unique(keys).tolist(), ordered=ordered)

    order = dim.encode(keys)
    for attr in attributes:
        col = _column(frame, attr)
        arranged = np.empty(len(dim), dtype=object)
        arranged[:] = None
        arranged[order] = col
        dim.set_attribute(attr, arranged)

    for m in measures:
        col = np.asarray(_column(frame, m), dtype=np.float64)
        arr = np.zeros(len(dim), dtype=np.float64)
        arr[order] = col
        catalog.add_cube(
            Cube(m, m, MultidimensionalMatrix(arr.astype(dtype), (dim_name,)), DENSE, "sum")
        )
    return dim


def add_hierarchy(
    catalog: Catalog,
    child: str,
    parent: str,
    mapping: Sequence[Any] | dict[Any, Any],
    name: str | None = None,
    parent_ordered: bool = False,
) -> Hierarchy:
    """Регистрирует иерархию child -> parent.

    ``mapping`` — либо последовательность родителей в порядке ординалов
    дочернего измерения, либо словарь {значение_child: значение_parent}.
    """
    child_dim = catalog.dimension(child)
    if isinstance(mapping, dict):
        mapping = [mapping[v] for v in child_dim.labels()]
    parent_dim = catalog.ensure_dimension(
        parent, sorted(set(mapping), key=lambda x: (x is None, x)), ordered=parent_ordered
    )
    h = Hierarchy(name or f"{child}->{parent}", child_dim, parent_dim, list(mapping))
    return catalog.add_hierarchy(h)


def read_csv(path: str | Path, **kwargs: Any) -> Any:
    """CSV -> DataFrame (pandas) или dict столбцов, если pandas недоступен."""
    if HAVE_PANDAS:
        return pd.read_csv(path, **kwargs)
    import csv as _csv

    with open(path, newline="", encoding=kwargs.get("encoding", "utf-8")) as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return {}
    out: dict[str, list[Any]] = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try:
                out[k].append(int(v))
            except (TypeError, ValueError):
                try:
                    out[k].append(float(v))
                except (TypeError, ValueError):
                    out[k].append(v)
    return {k: np.asarray(v) for k, v in out.items()}


def read_sql(query: str, connection: Any, chunksize: int | None = None) -> Any:
    """Потоковое чтение из PostgreSQL/ClickHouse/SQLite через DB-API соединение."""
    if HAVE_PANDAS:
        if chunksize:
            return pd.concat(pd.read_sql(query, connection, chunksize=chunksize),
                             ignore_index=True)
        return pd.read_sql(query, connection)
    cur = connection.cursor()
    cur.execute(query)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}


def load_from_sql(
    catalog: Catalog,
    connection: Any,
    query: str,
    dim_cols: Sequence[str],
    measure: str,
    **kwargs: Any,
) -> Cube:
    """Реляционная СУБД -> гиперкуб одной операцией."""
    return load_fact(catalog, read_sql(query, connection), dim_cols, measure, **kwargs)
