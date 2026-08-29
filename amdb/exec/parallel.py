# -*- coding: utf-8 -*-
"""Параллельное исполнение свёртки по скоттовым сечениям.

Реализует первое разбиение из работы Мунермана и Мунермана (2022): при λ > 0
результат (λ, μ)-свёрнутого произведения составлен из сечений ориентации
(s_1, …, s_λ), каждое из которых есть (0, μ)-произведение соответствующих
сечений сомножителей. Сечения **независимы** — обмена между процессами не
требуется, и обобщённый алгоритм Кэннона с циркуляцией блоков нужен лишь на
следующем уровне, для распределённой памяти. В пределах одного узла с общей
памятью циркулировать нечего: блоки и так доступны всем.

Два обстоятельства определяют устройство модуля.

**Потоки не годятся.** einsum не освобождает GIL, поэтому разбиение по потокам
на нём даёт замедление (замерено: 0.18–0.56 от однопоточного времени). Нужны
процессы.

**BLAS уже занимает все ядра.** Однопоточный на вид вызов einsum внутри
использует многопоточный OpenBLAS, то есть машина загружена и без нас. Если
запустить процессы, не ограничив в них число потоков BLAS, они станут
конкурировать за одни и те же ядра. Поэтому дочерним процессам передаётся
окружение с одним потоком BLAS: параллелизм по сечениям заменяет
внутриблочный, а не добавляется к нему.

Отсюда и область применимости, измеряемая стендом ``bench/bench_parallel.py``:
выигрыш возможен там, где BLAS распараллеливает плохо, и отсутствует там, где
он справляется сам.
"""
from __future__ import annotations

import os
from multiprocessing import get_context, shared_memory
from typing import Any, Sequence

import numpy as np

from .engine import NumpyEngine

#: Ниже этого суммарного числа ячеек операндов накладные расходы на запуск
#: процессов и перенос в общую память заведомо больше выигрыша.
MIN_PARALLEL_CELLS: int = 4_000_000

#: Переменные, которыми ограничивается число потоков BLAS в дочерних процессах.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

#: Кэш присоединённых блоков общей памяти внутри дочернего процесса.
_ATTACHED: dict[str, Any] = {}

#: Движок рабочего процесса — тот же, что и на однопоточном пути.
_ENGINE = NumpyEngine()


def _attach(name: str):
    block = _ATTACHED.get(name)
    if block is None:
        block = shared_memory.SharedMemory(name=name)
        _ATTACHED[name] = block
    return block


def _worker(task):
    """Считает одно сечение прямо в общую память результата.

    Возвращать массив через очередь нельзя: он сериализуется, и на результате
    в десятки мегабайт пересылка стоит дороже самого счёта (замерено).
    """
    spec, blocks, lo, hi, axis_pos, out_name, out_shape, out_dtype = task
    operands = []
    for name, shape, dtype, pos in blocks:
        arr = np.ndarray(shape, dtype=np.dtype(dtype), buffer=_attach(name).buf)
        if pos is None:
            operands.append(arr)
        else:
            operands.append(arr[(slice(None),) * pos + (slice(lo, hi),)])
    out = np.ndarray(out_shape, dtype=np.dtype(out_dtype), buffer=_attach(out_name).buf)
    view = out[(slice(None),) * axis_pos + (slice(lo, hi),)]
    # Через NumpyEngine, а не через np.einsum напрямую: иначе рабочий процесс
    # считал бы обычной свёрткой, тогда как однопоточный путь идёт разложением
    # на сечения с пакетным gemm, и сравнивались бы разные алгоритмы.
    view[...] = _ENGINE.einsum(spec, *operands)
    return None


def scott_axes(spec: str) -> list[str]:
    """Индексы, годные для разбиения: общие сомножителям и сохраняемые в результате.

    Разбиение по такому индексу режет **оба** сомножителя, поэтому
    реплицировать ничего не приходится; по свободной оси пришлось бы копировать
    второй операнд целиком (§3.4 статьи).
    """
    if "->" not in spec:
        return []
    lhs, out = spec.split("->")
    subs = lhs.split(",")
    counts: dict[str, int] = {}
    for sub in subs:
        for letter in set(sub):
            counts[letter] = counts.get(letter, 0) + 1
    return [x for x in out if counts.get(x, 0) > 1]


class ParallelEngine:
    """Движок, исполняющий свёртку по независимым скоттовым сечениям.

    ``resident`` оставляет операнды в общей памяти между вызовами: при
    повторных запросах к неизменному гиперкубу перенос выполняется однократно.
    Это тот же приём, что и резидентные операнды на ускорителе, и по той же
    причине — перенос сопоставим по времени со счётом.
    """

    name = "parallel"

    def __init__(self, workers: int | None = None,
                 min_cells: int = MIN_PARALLEL_CELLS,
                 blas_threads: int = 1, resident: bool = True):
        self.workers = workers or (os.cpu_count() or 2)
        self.min_cells = min_cells
        self.blas_threads = blas_threads
        self.resident = resident
        self._fallback = NumpyEngine()
        self._pool = None
        self._shared: dict[int, tuple[np.ndarray, Any]] = {}

    # -- общая память ---------------------------------------------------------
    def _publish(self, array: np.ndarray):
        """Кладёт массив в общую память; при resident — один раз на массив."""
        array = np.ascontiguousarray(array)
        if self.resident:
            hit = self._shared.get(id(array))
            if hit is not None and hit[0] is array:
                return hit[1], False
        block = shared_memory.SharedMemory(create=True, size=max(array.nbytes, 1))
        np.ndarray(array.shape, dtype=array.dtype, buffer=block.buf)[...] = array
        if self.resident:
            self._shared[id(array)] = (array, block)
            return block, False
        return block, True

    def close(self) -> None:
        """Освобождает пул и общую память. Вызывать обязательно: блоки живут
        дольше процесса, если их не удалить явно."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None
        for _, block in self._shared.values():
            block.close()
            block.unlink()
        self._shared.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- пул -------------------------------------------------------------------
    def _ensure_pool(self, n: int):
        if self._pool is not None:
            return self._pool
        saved = {v: os.environ.get(v) for v in _THREAD_VARS}
        for v in _THREAD_VARS:
            os.environ[v] = str(self.blas_threads)
        try:
            # spawn: дочерний процесс импортирует numpy заново и подхватывает
            # ограничение на число потоков BLAS из окружения.
            self._pool = get_context("spawn").Pool(n)
        finally:
            for v, old in saved.items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old
        return self._pool

    # -- вычисление -------------------------------------------------------------
    def einsum(self, spec: str, *ops: np.ndarray, path: Any = None) -> np.ndarray:
        arrays = [np.asarray(o) for o in ops]
        plan = self._split_plan(spec, arrays)
        if plan is None:
            return self._fallback.einsum(spec, *arrays, path=path)
        letter, axis_out_pos, bounds = plan

        lhs = spec.split("->")[0].split(",")
        blocks, temporary = [], []
        for sub, arr in zip(lhs, arrays):
            block, is_temp = self._publish(arr)
            blocks.append((block.name, arr.shape, arr.dtype.str,
                           sub.index(letter) if letter in sub else None))
            if is_temp:
                temporary.append(block)

        out_shape = self._output_shape(spec, arrays)
        dtype = np.result_type(*arrays)
        out_block = shared_memory.SharedMemory(
            create=True, size=max(int(np.prod(out_shape)) * dtype.itemsize, 1))
        tasks = [(spec, blocks, lo, hi, axis_out_pos,
                  out_block.name, out_shape, dtype.str) for lo, hi in bounds]
        try:
            pool = self._ensure_pool(len(tasks))
            pool.map(_worker, tasks)
            result = np.ndarray(out_shape, dtype=dtype, buffer=out_block.buf).copy()
        finally:
            for block in temporary:
                block.close()
                block.unlink()
            out_block.close()
            out_block.unlink()
        return result

    @staticmethod
    def _output_shape(spec: str, arrays: Sequence[np.ndarray]) -> tuple[int, ...]:
        lhs, out = spec.split("->")
        sizes: dict[str, int] = {}
        for sub, arr in zip(lhs.split(","), arrays):
            for letter, length in zip(sub, arr.shape):
                sizes[letter] = length
        return tuple(sizes[x] for x in out)

    def _split_plan(self, spec: str, arrays: Sequence[np.ndarray]):
        """Выбирает ось разбиения и границы кусков либо отказывается."""
        if "->" not in spec or len(arrays) < 2:
            return None
        if sum(a.size for a in arrays) < self.min_cells:
            return None
        candidates = scott_axes(spec)
        if not candidates:
            return None
        lhs, out = spec.split("->")
        subs = lhs.split(",")
        sizes: dict[str, int] = {}
        for sub, arr in zip(subs, arrays):
            for letter, length in zip(sub, arr.shape):
                sizes[letter] = length
        letter = max(candidates, key=lambda x: sizes.get(x, 0))
        length = sizes.get(letter, 0)
        n = min(self.workers, length)
        if n < 2:
            return None
        edges = np.linspace(0, length, n + 1).astype(int)
        bounds = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        if len(bounds) < 2:
            return None
        return letter, out.index(letter), bounds

    def __repr__(self) -> str:
        return (f"ParallelEngine(workers={self.workers}, "
                f"blas_threads={self.blas_threads})")
