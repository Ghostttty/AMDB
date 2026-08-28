# -*- coding: utf-8 -*-
"""Исполнение планов: движки, чанкование, сборка результата."""
from .chunker import (
    axis_kinds,
    chunk_ranges,
    concat_chunks,
    iter_slices,
    replication_factor,
    split_axis,
)
from .engine import (
    Engine,
    NumpyEngine,
    TorchEngine,
    blas_info,
    gpu_available,
    pick_engine,
    warn_if_reference_blas,
)
from .executor import Executor
from .result import ResultSet

__all__ = [
    "Engine",
    "Executor",
    "axis_kinds",
    "NumpyEngine",
    "ResultSet",
    "TorchEngine",
    "blas_info",
    "chunk_ranges",
    "concat_chunks",
    "gpu_available",
    "iter_slices",
    "pick_engine",
    "replication_factor",
    "split_axis",
    "warn_if_reference_blas",
]
