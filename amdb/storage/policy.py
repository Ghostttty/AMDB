# -*- coding: utf-8 -*-
"""Политика физического представления гиперкуба.

Выбор «плотно или разреженно» принимается по фактору заполнения и по бюджету
памяти, а не задаётся вручную: реальные OLAP-факты имеют заполненность
10⁻³…10⁻⁶, и плотное хранение для них означает гарантированный отказ.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: Ниже этого фактора заполнения плотное хранение невыгодно.
DENSE_THRESHOLD = 0.02
#: Бюджет на один плотный куб в памяти.
MAX_DENSE_BYTES = 8 * 2**30

DENSE = "dense"
SPARSE_COO = "sparse_coo"
CHUNKED_SPARSE = "chunked_sparse"


def choose_layout(
    nnz: int,
    shape: Sequence[int],
    itemsize: int = 4,
    dense_threshold: float = DENSE_THRESHOLD,
    max_dense_bytes: int = MAX_DENSE_BYTES,
) -> str:
    """Возвращает 'dense' | 'sparse_coo' | 'chunked_sparse'."""
    cells = int(np.prod(shape)) if len(shape) else 1
    dense_bytes = cells * itemsize
    coo_bytes = nnz * (itemsize + 8 * len(shape))  # значение + int64-координаты
    if dense_bytes > max_dense_bytes:
        return CHUNKED_SPARSE if coo_bytes > max_dense_bytes else SPARSE_COO
    if cells and nnz / cells < dense_threshold and coo_bytes < dense_bytes:
        return SPARSE_COO
    return DENSE


def estimate_bytes(shape: Sequence[int], itemsize: int = 4) -> int:
    return int(np.prod(shape)) * itemsize if len(shape) else itemsize


def check_budget(shape: Sequence[int], itemsize: int = 4,
                 budget: int = MAX_DENSE_BYTES) -> None:
    """Бросает MemoryError до аллокации, а не во время неё."""
    need = estimate_bytes(shape, itemsize)
    if need > budget:
        raise MemoryError(
            f"плотный куб {tuple(shape)} требует {need / 2**30:.1f} ГиБ "
            f"при бюджете {budget / 2**30:.1f} ГиБ; используйте разреженное хранение"
        )
