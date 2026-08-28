# -*- coding: utf-8 -*-
"""Бэкенды физического хранения массивов гиперкубов.

v1: HDF5 (чанкованное хранение со сжатием) при наличии h5py, иначе .npz.
Интерфейс намеренно узкий, чтобы v2 (TileDB) и v3 (Zarr/S3) подключались
заменой двух функций.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..core.sparse import COOCube

try:  # pragma: no cover - зависит от окружения
    import h5py
    HAVE_HDF5 = True
except Exception:  # pragma: no cover
    h5py = None
    HAVE_HDF5 = False

ARRAYS_H5 = "arrays.h5"
ARRAYS_NPZ = "arrays.npz"


def _chunks(shape: tuple[int, ...]) -> tuple[int, ...] | None:
    """Чанк ~1 МиБ: последняя ось целиком, остальные режутся."""
    if not shape or int(np.prod(shape)) * 4 < 2**20:
        return None
    chunk = list(shape)
    target = 2**18  # ~256K элементов
    while int(np.prod(chunk)) > target:
        biggest = int(np.argmax(chunk))
        if chunk[biggest] <= 1:
            break
        chunk[biggest] = max(1, chunk[biggest] // 2)
    return tuple(chunk)


def write_arrays(path: str | Path, cubes: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if HAVE_HDF5:
        target = path / ARRAYS_H5
        with h5py.File(target, "w") as f:
            for name, cube in cubes.items():
                g = f.create_group(name)
                if isinstance(cube.matrix, COOCube):
                    g.create_dataset("coords", data=cube.matrix.coords, compression="gzip")
                    g.create_dataset("values", data=cube.matrix.values, compression="gzip")
                    g.create_dataset("shape", data=np.array(cube.matrix.shape, dtype=np.int64))
                else:
                    data = cube.matrix.data
                    g.create_dataset("data", data=data, compression="gzip",
                                     chunks=_chunks(data.shape))
        return target
    target = path / ARRAYS_NPZ
    payload: dict[str, np.ndarray] = {}
    for name, cube in cubes.items():
        if isinstance(cube.matrix, COOCube):
            payload[f"{name}/coords"] = cube.matrix.coords
            payload[f"{name}/values"] = cube.matrix.values
            payload[f"{name}/shape"] = np.array(cube.matrix.shape, dtype=np.int64)
        else:
            payload[f"{name}/data"] = cube.matrix.data
    np.savez_compressed(target, **payload)
    return target


def read_arrays(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    path = Path(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    h5 = path / ARRAYS_H5
    if HAVE_HDF5 and h5.exists():
        with h5py.File(h5, "r") as f:
            for name in f:
                out[name] = {k: np.asarray(v) for k, v in f[name].items()}
        return out
    npz = path / ARRAYS_NPZ
    if not npz.exists():
        raise FileNotFoundError(f"в {path} нет ни {ARRAYS_H5}, ни {ARRAYS_NPZ}")
    with np.load(npz, allow_pickle=False) as z:
        for key in z.files:
            name, _, field = key.partition("/")
            out.setdefault(name, {})[field] = z[key]
    return out


def open_dense(path: str | Path, cube_name: str) -> np.ndarray:
    """Ленивое чтение плотного куба (в HDF5 — без загрузки всего массива)."""
    path = Path(path)
    h5 = path / ARRAYS_H5
    if HAVE_HDF5 and h5.exists():
        f = h5py.File(h5, "r")
        return f[cube_name]["data"]  # h5py-датасет поддерживает срезы
    return read_arrays(path)[cube_name]["data"]
