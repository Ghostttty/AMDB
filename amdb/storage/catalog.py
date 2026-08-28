# -*- coding: utf-8 -*-
"""Каталог метаданных: измерения, кубы, иерархии, права, статистика.

Каталог — единственный источник истины для транслятора и оптимизатора.
Персистентность: метаданные в SQLite (схема из техпроекта, §5), массивы — в
HDF5 (если доступен h5py) или в .npz.
"""
from __future__ import annotations

import json
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.mdm import MultidimensionalMatrix
from ..core.sparse import COOCube
from .dimension import Dimension, Hierarchy
from .policy import DENSE, SPARSE_COO, choose_layout

SCHEMA = """
CREATE TABLE IF NOT EXISTS dimension (
    dim_id      INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    value_type  TEXT NOT NULL,
    cardinality INTEGER NOT NULL,
    is_ordered  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS dim_member (
    dim_id  INTEGER NOT NULL REFERENCES dimension(dim_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    key_json TEXT NOT NULL,
    PRIMARY KEY (dim_id, ordinal)
);
CREATE TABLE IF NOT EXISTS dim_attribute (
    dim_id INTEGER NOT NULL REFERENCES dimension(dim_id) ON DELETE CASCADE,
    name   TEXT NOT NULL,
    values_json TEXT NOT NULL,
    PRIMARY KEY (dim_id, name)
);
CREATE TABLE IF NOT EXISTS cube (
    cube_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    measure     TEXT NOT NULL,
    dtype       TEXT NOT NULL,
    layout      TEXT NOT NULL,
    default_agg TEXT NOT NULL DEFAULT 'sum',
    storage_key TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS cube_axis (
    cube_id  INTEGER NOT NULL REFERENCES cube(cube_id) ON DELETE CASCADE,
    axis_pos INTEGER NOT NULL,
    dim_id   INTEGER NOT NULL REFERENCES dimension(dim_id),
    PRIMARY KEY (cube_id, axis_pos),
    UNIQUE (cube_id, dim_id)
);
CREATE TABLE IF NOT EXISTS hierarchy (
    hier_id       INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    child_dim_id  INTEGER NOT NULL REFERENCES dimension(dim_id),
    parent_dim_id INTEGER NOT NULL REFERENCES dimension(dim_id),
    map_json      TEXT NOT NULL,
    UNIQUE (child_dim_id, parent_dim_id, name)
);
CREATE TABLE IF NOT EXISTS cube_stats (
    cube_id     INTEGER PRIMARY KEY REFERENCES cube(cube_id) ON DELETE CASCADE,
    nnz         INTEGER NOT NULL,
    total_cells INTEGER NOT NULL,
    fill_factor REAL NOT NULL,
    bytes_in_memory INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS role (
    role_id INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS dim_grant (
    role_id      INTEGER NOT NULL REFERENCES role(role_id) ON DELETE CASCADE,
    dim_id       INTEGER NOT NULL REFERENCES dimension(dim_id) ON DELETE CASCADE,
    allowed_json TEXT,
    can_project  INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (role_id, dim_id)
);
"""


@dataclass
class Cube:
    """Гиперкуб факта: массив данных + метаданные."""

    name: str
    measure: str
    matrix: MultidimensionalMatrix | COOCube
    layout: str = DENSE
    default_agg: str = "sum"
    version: int = 1

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self.matrix.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.matrix.shape)

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.matrix, COOCube)

    def dense(self) -> MultidimensionalMatrix:
        return self.matrix.to_dense() if self.is_sparse else self.matrix

    @property
    def nnz(self) -> int:
        return self.matrix.nnz if self.is_sparse else int(np.count_nonzero(self.matrix.data))

    def __repr__(self) -> str:
        dims = ", ".join(f"{a}={n}" for a, n in zip(self.axes, self.shape))
        return f"Cube({self.name!r}, measure={self.measure!r}, [{dims}], {self.layout})"


@dataclass
class Grant:
    """Право роли на измерение."""

    allowed: np.ndarray | None = None      # 0/1-вектор; None = полный доступ
    can_project: bool = True


@dataclass
class Role:
    name: str
    grants: dict[str, Grant] = field(default_factory=dict)
    #: Если True, измерения без явного гранта доступны полностью.
    permissive: bool = True


class Catalog:
    """Реестр измерений, кубов, иерархий и прав доступа."""

    def __init__(self) -> None:
        self.dimensions: dict[str, Dimension] = {}
        self.cubes: dict[str, Cube] = {}
        self.hierarchies: list[Hierarchy] = []
        self.roles: dict[str, Role] = {}
        self._measure_index: dict[str, list[str]] = {}

    # -- измерения ----------------------------------------------------------
    def add_dimension(self, dim: Dimension) -> Dimension:
        existing = self.dimensions.get(dim.name)
        if existing is not None and existing is not dim:
            raise ValueError(f"измерение '{dim.name}' уже зарегистрировано")
        self.dimensions[dim.name] = dim
        return dim

    def dimension(self, name: str) -> Dimension:
        try:
            return self.dimensions[name]
        except KeyError:
            raise KeyError(f"неизвестное измерение '{name}'") from None

    def ensure_dimension(self, name: str, values: Iterable[Any] = (),
                         ordered: bool = False) -> Dimension:
        if name not in self.dimensions:
            self.dimensions[name] = Dimension(name, values, ordered=ordered)
        else:
            self.dimensions[name].extend(values)
        return self.dimensions[name]

    # -- кубы ---------------------------------------------------------------
    def add_cube(self, cube: Cube) -> Cube:
        for axis, length in zip(cube.axes, cube.shape):
            dim = self.dimension(axis)
            if len(dim) != length:
                raise ValueError(
                    f"куб '{cube.name}': ось '{axis}' длины {length}, "
                    f"мощность измерения {len(dim)}"
                )
        if cube.name in self.cubes:
            cube.version = self.cubes[cube.name].version + 1
        self.cubes[cube.name] = cube
        self._measure_index.setdefault(cube.measure, [])
        if cube.name not in self._measure_index[cube.measure]:
            self._measure_index[cube.measure].append(cube.name)
        return cube

    def cube(self, name: str) -> Cube:
        try:
            return self.cubes[name]
        except KeyError:
            raise KeyError(f"неизвестный куб '{name}'") from None

    def axes_of(self, cube_name: str) -> tuple[str, ...]:
        return self.cube(cube_name).axes

    def cube_for_measure(self, measure: str, candidates: Sequence[str] | None = None) -> Cube:
        """Находит куб, содержащий указанную меру.

        ``candidates`` ограничивает поиск кубами из FROM/JOIN текущего запроса.
        """
        names = self._measure_index.get(measure, [])
        if candidates is not None:
            names = [n for n in names if n in candidates]
        if not names:
            raise KeyError(f"мера '{measure}' не найдена среди кубов {list(candidates or self.cubes)}")
        if len(names) > 1:
            raise KeyError(
                f"мера '{measure}' неоднозначна: {names}; уточните кубом (cube.measure)"
            )
        return self.cube(names[0])

    def measures(self) -> dict[str, str]:
        return {c.measure: c.name for c in self.cubes.values()}

    # -- иерархии -----------------------------------------------------------
    def add_hierarchy(self, hier: Hierarchy) -> Hierarchy:
        self.add_dimension(hier.child) if hier.child.name not in self.dimensions else None
        self.add_dimension(hier.parent) if hier.parent.name not in self.dimensions else None
        self.hierarchies.append(hier)
        return hier

    def hierarchy_path(self, child: str, parent: str) -> list[Hierarchy] | None:
        """Кратчайший путь по иерархиям от дочернего измерения к родительскому."""
        if child == parent:
            return []
        seen = {child}
        queue: deque[tuple[str, list[Hierarchy]]] = deque([(child, [])])
        while queue:
            node, path = queue.popleft()
            for h in self.hierarchies:
                if h.child.name != node or h.parent.name in seen:
                    continue
                new = path + [h]
                if h.parent.name == parent:
                    return new
                seen.add(h.parent.name)
                queue.append((h.parent.name, new))
        return None

    def rollup_matrix(self, child: str, parent: str) -> np.ndarray | None:
        """Матрица перехода child → parent (композиция по цепочке иерархий)."""
        path = self.hierarchy_path(child, parent)
        if path is None:
            return None
        if not path:
            return None
        m = path[0].matrix()
        for h in path[1:]:
            m = m @ h.matrix()
        return m

    # -- права --------------------------------------------------------------
    def add_role(self, role: Role) -> Role:
        self.roles[role.name] = role
        return role

    def role(self, name: str) -> Role:
        try:
            return self.roles[name]
        except KeyError:
            raise KeyError(f"неизвестная роль '{name}'") from None

    def grant(self, role: Role | str | None, axis: str) -> Grant | None:
        """Право роли на измерение; None означает отказ в доступе."""
        if role is None:
            return Grant()
        if isinstance(role, str):
            role = self.role(role)
        if axis in role.grants:
            return role.grants[axis]
        return Grant() if role.permissive else None

    # -- статистика ---------------------------------------------------------
    def stats(self, cube_name: str) -> dict[str, Any]:
        c = self.cube(cube_name)
        cells = int(np.prod(c.shape)) if c.shape else 1
        nnz = c.nnz
        itemsize = 4 if c.is_sparse else c.matrix.data.dtype.itemsize
        return {
            "cube": c.name,
            "measure": c.measure,
            "axes": c.axes,
            "shape": c.shape,
            "nnz": nnz,
            "total_cells": cells,
            "fill_factor": nnz / max(cells, 1),
            "layout": c.layout,
            "bytes_in_memory": (
                c.matrix.values.nbytes + c.matrix.coords.nbytes
                if c.is_sparse else c.matrix.data.nbytes
            ),
            "recommended_layout": choose_layout(nnz, c.shape, itemsize),
            "version": c.version,
        }

    def summary(self) -> str:
        lines = ["Измерения:"]
        for d in self.dimensions.values():
            attrs = f", атрибуты: {sorted(d.attributes)}" if d.attributes else ""
            lines.append(f"  {d.name}: {len(d)}{' (упорядочено)' if d.ordered else ''}{attrs}")
        lines.append("Кубы:")
        for c in self.cubes.values():
            lines.append(f"  {c!r}")
        if self.hierarchies:
            lines.append("Иерархии:")
            for h in self.hierarchies:
                lines.append(f"  {h.child.name} -> {h.parent.name} ({h.name})")
        return "\n".join(lines)

    # -- персистентность ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Сохраняет каталог: метаданные в SQLite, массивы в HDF5/NPZ."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        db = path / "catalog.sqlite"
        if db.exists():
            db.unlink()
        con = sqlite3.connect(db)
        try:
            con.executescript(SCHEMA)
            dim_ids: dict[str, int] = {}
            for i, d in enumerate(self.dimensions.values(), start=1):
                dim_ids[d.name] = i
                vtype = "utf8" if d._is_text else "int64"
                con.execute(
                    "INSERT INTO dimension (dim_id, name, value_type, cardinality, is_ordered)"
                    " VALUES (?,?,?,?,?)",
                    (i, d.name, vtype, len(d), int(d.ordered)),
                )
                con.executemany(
                    "INSERT INTO dim_member (dim_id, ordinal, key_json) VALUES (?,?,?)",
                    [(i, o, json.dumps(v, default=str)) for o, v in enumerate(d.labels())],
                )
                for aname, avals in d.attributes.items():
                    con.execute(
                        "INSERT INTO dim_attribute (dim_id, name, values_json) VALUES (?,?,?)",
                        (i, aname, json.dumps(list(avals), default=str)),
                    )
            for i, c in enumerate(self.cubes.values(), start=1):
                dtype = str(c.matrix.values.dtype if c.is_sparse else c.matrix.data.dtype)
                con.execute(
                    "INSERT INTO cube (cube_id, name, measure, dtype, layout, default_agg,"
                    " storage_key, version) VALUES (?,?,?,?,?,?,?,?)",
                    (i, c.name, c.measure, dtype, c.layout, c.default_agg, c.name, c.version),
                )
                con.executemany(
                    "INSERT INTO cube_axis (cube_id, axis_pos, dim_id) VALUES (?,?,?)",
                    [(i, p, dim_ids[a]) for p, a in enumerate(c.axes)],
                )
                s = self.stats(c.name)
                con.execute(
                    "INSERT INTO cube_stats (cube_id, nnz, total_cells, fill_factor,"
                    " bytes_in_memory) VALUES (?,?,?,?,?)",
                    (i, s["nnz"], s["total_cells"], s["fill_factor"], s["bytes_in_memory"]),
                )
            for i, h in enumerate(self.hierarchies, start=1):
                con.execute(
                    "INSERT INTO hierarchy (hier_id, name, child_dim_id, parent_dim_id, map_json)"
                    " VALUES (?,?,?,?,?)",
                    (i, h.name, dim_ids[h.child.name], dim_ids[h.parent.name],
                     json.dumps([int(x) for x in h.child_ordinals])),
                )
            for i, r in enumerate(self.roles.values(), start=1):
                con.execute("INSERT INTO role (role_id, name) VALUES (?,?)", (i, r.name))
                for axis, g in r.grants.items():
                    con.execute(
                        "INSERT INTO dim_grant (role_id, dim_id, allowed_json, can_project)"
                        " VALUES (?,?,?,?)",
                        (i, dim_ids[axis],
                         None if g.allowed is None else json.dumps([float(x) for x in g.allowed]),
                         int(g.can_project)),
                    )
            con.commit()
        finally:
            con.close()
        from .backends import write_arrays
        write_arrays(path, self.cubes)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        path = Path(path)
        con = sqlite3.connect(path / "catalog.sqlite")
        con.row_factory = sqlite3.Row
        cat = cls()
        try:
            dim_by_id: dict[int, Dimension] = {}
            for row in con.execute("SELECT * FROM dimension ORDER BY dim_id"):
                members = [
                    json.loads(r["key_json"])
                    for r in con.execute(
                        "SELECT key_json FROM dim_member WHERE dim_id=? ORDER BY ordinal",
                        (row["dim_id"],))
                ]
                d = Dimension(row["name"], members, ordered=bool(row["is_ordered"]))
                for a in con.execute(
                        "SELECT name, values_json FROM dim_attribute WHERE dim_id=?",
                        (row["dim_id"],)):
                    d.set_attribute(a["name"], json.loads(a["values_json"]))
                dim_by_id[row["dim_id"]] = cat.add_dimension(d)
            for row in con.execute("SELECT * FROM hierarchy ORDER BY hier_id"):
                child, parent = dim_by_id[row["child_dim_id"]], dim_by_id[row["parent_dim_id"]]
                ordinals = json.loads(row["map_json"])
                h = Hierarchy.__new__(Hierarchy)
                h.name, h.child, h.parent = row["name"], child, parent
                h.child_ordinals = np.asarray(ordinals, dtype=np.int64)
                cat.hierarchies.append(h)
            from .backends import read_arrays
            arrays = read_arrays(path)
            for row in con.execute("SELECT * FROM cube ORDER BY cube_id"):
                axes = tuple(
                    dim_by_id[r["dim_id"]].name
                    for r in con.execute(
                        "SELECT dim_id FROM cube_axis WHERE cube_id=? ORDER BY axis_pos",
                        (row["cube_id"],))
                )
                payload = arrays[row["storage_key"]]
                if row["layout"] == SPARSE_COO:
                    shape = tuple(int(x) for x in payload["shape"])
                    matrix: Any = COOCube(payload["coords"], payload["values"], axes, shape)
                else:
                    matrix = MultidimensionalMatrix(payload["data"], axes)
                cat.add_cube(Cube(row["name"], row["measure"], matrix, row["layout"],
                                  row["default_agg"], row["version"]))
            for row in con.execute("SELECT * FROM role ORDER BY role_id"):
                role = Role(row["name"], {})
                for g in con.execute(
                        "SELECT * FROM dim_grant WHERE role_id=?", (row["role_id"],)):
                    allowed = (None if g["allowed_json"] is None
                               else np.array(json.loads(g["allowed_json"]), dtype=np.float32))
                    role.grants[dim_by_id[g["dim_id"]].name] = Grant(allowed, bool(g["can_project"]))
                cat.add_role(role)
        finally:
            con.close()
        return cat
