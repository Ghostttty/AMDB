# -*- coding: utf-8 -*-
"""Исполнитель физического плана.

Каждый einsum-шаг выполняется выбранным движком; разреженные кубы идут по
собственному пути свёртки, не материализуя плотный массив.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from ..core.mdm import MultidimensionalMatrix, convolve_named
from ..core.ops import indicator, reduce_axes, rollup_reduce, running_sum
from ..core.sparse import COOCube, convolve_sparse
from ..ql.ast import Aggregate, BinOp, Column, Compare, Condition, Expr, Literal, Logical, Not, Star
from ..ql.binder import BindError
from ..ql.planner import ARRAY, CUBE, INDICATOR, AggPlan, EinsumStep, Operand, PhysicalPlan, ReduceStep
from .engine import Engine, NumpyEngine, gpu_available, pick_engine, spec_cost
from .result import ResultSet


class Executor:
    """Исполняет физический план.

    Движок выбирается для каждого шага отдельно. Явно переданный движок
    используется всегда — так работают тесты и стенды, которым нужен
    предсказуемый путь. Если движок не задан, а ускоритель доступен, решение
    принимает :func:`pick_engine` по оценке шага: перенос данных через шину
    сопоставим со временем самой свёртки, поэтому включать ускоритель по одному
    лишь объёму данных нельзя.
    """

    def __init__(self, catalog: Any, engine: Engine | None = None):
        self.catalog = catalog
        self.engine = engine or NumpyEngine()
        self._fixed_engine = engine
        self._accelerator: Engine | None = None
        self._engines_used: set[str] = set()

    def _engine_for(self, spec: str, arrays: Sequence[np.ndarray]) -> Engine:
        if self._fixed_engine is not None:
            return self._fixed_engine
        if not gpu_available():
            return self.engine
        chosen = pick_engine(*spec_cost(spec, *arrays))
        if chosen.name == "numpy":
            return self.engine
        if self._accelerator is None:
            self._accelerator = chosen
        return self._accelerator

    def _einsum(self, spec: str, arrays: Sequence[np.ndarray], path: Any) -> np.ndarray:
        engine = self._engine_for(spec, arrays)
        self._engines_used.add(engine.name)
        return engine.einsum(spec, *arrays, path=path)

    # -- публичный вход -----------------------------------------------------
    def run(self, plan: PhysicalPlan) -> ResultSet:
        t0 = time.perf_counter()
        # Один и тот же счётный шаг обычно нужен и для COUNT, и для знаменателя
        # AVG, и для определения непустых групп — считаем его один раз.
        self._step_cache: dict[tuple, np.ndarray] = {}
        self._engines_used = set()
        group_axes = plan.group_axes
        shape = tuple(len(self.catalog.dimension(a)) for a in group_axes)

        values: dict[int, np.ndarray] = {}
        for agg, node in zip(plan.aggregates, plan.query.select.aggregates()):
            values[id(node)] = self._run_aggregate(agg, group_axes, shape)

        presence = (self._run_step(plan.presence, group_axes)
                    if plan.presence is not None else None)
        t_compute = time.perf_counter() - t0

        result = self._assemble(plan, values, presence, group_axes, shape)
        result.stats.update({
            "compute_seconds": t_compute,
            "total_seconds": time.perf_counter() - t0,
            "engine": "+".join(sorted(self._engines_used)) or self.engine.name,
            "group_shape": shape,
        })
        return result

    # -- агрегаты -----------------------------------------------------------
    def _run_aggregate(self, agg: AggPlan, group_axes: Sequence[str],
                       shape: tuple[int, ...]) -> np.ndarray:
        if agg.distinct is not None:
            out = self._run_distinct(agg.distinct, group_axes)
        elif agg.reduce is not None:
            out = self._run_reduce(agg.reduce, group_axes)
        else:
            out = np.zeros(shape, dtype=np.float64)
            for coef, step in agg.terms:
                out = out + coef * self._run_step(step, group_axes)
            if agg.denominator is not None:
                denom = self._run_step(agg.denominator, group_axes)
                out = np.divide(out, denom, out=np.zeros_like(out), where=denom != 0)
        if agg.window is not None:
            out = self._apply_window(out, agg, group_axes)
        return out

    def _apply_window(self, data: np.ndarray, agg: AggPlan,
                      group_axes: Sequence[str]) -> np.ndarray:
        spec = agg.window
        if spec.order_by is None:
            raise BindError("оконная функция требует ORDER BY внутри OVER (...)")
        if spec.order_by not in group_axes:
            raise BindError(
                f"ORDER BY '{spec.order_by}' в окне должен быть среди измерений GROUP BY"
            )
        for p in spec.partition_by:
            if p not in group_axes:
                raise BindError(
                    f"PARTITION BY '{p}' должен быть среди измерений GROUP BY"
                )
        cube = MultidimensionalMatrix(data, tuple(group_axes))
        return running_sum(cube, spec.order_by, window=spec.frame_preceding).data

    # -- шаги ---------------------------------------------------------------
    def _materialize(self, op: Operand) -> tuple[tuple[str, ...], Any]:
        if op.kind == ARRAY:
            data: Any = np.asarray(op.array)
            full_axes = tuple(op.axes) + tuple(op.presum)
        else:
            cube = self.catalog.cube(op.name)
            full_axes = cube.axes
            if op.kind == INDICATOR:
                if cube.is_sparse:
                    m = cube.matrix
                    data = COOCube(m.coords, np.ones_like(m.values), m.axes, m.shape)
                else:
                    data = indicator(cube.matrix).data
            else:
                data = cube.matrix if cube.is_sparse else cube.matrix.data
        if op.presum:
            data = self._presum(data, full_axes, op.presum, op.axes)
        return op.axes, data

    @staticmethod
    def _presum(data: Any, full_axes: tuple[str, ...], drop: tuple[str, ...],
                keep: tuple[str, ...]) -> Any:
        """Предварительная свёртка операнда по его «висячим» осям.

        Алгебраически это (0, μ)-произведение с матрицей единиц; по теореме о
        разложении результат от такого переупорядочения не меняется, зато
        объём вычислений падает на порядок (см. optimizer.push_projections).
        """
        if isinstance(data, COOCube):
            reduced = data.project(drop)
            if tuple(reduced.axes) != tuple(keep):
                order = [reduced.axes.index(a) for a in keep]
                reduced = COOCube(reduced.coords[:, order], reduced.values, keep,
                                  tuple(reduced.shape[i] for i in order))
            return reduced
        pos = tuple(full_axes.index(a) for a in drop)
        out = np.sum(data, axis=pos)
        remaining = tuple(a for a in full_axes if a not in drop)
        if remaining != keep:
            out = np.transpose(out, [remaining.index(a) for a in keep])
        return out

    def _cache_key(self, step: EinsumStep) -> tuple:
        return (step.spec, step.output,
                tuple((o.kind, o.name, None if o.array is None else id(o.array))
                      for o in step.operands))

    def _run_step(self, step: EinsumStep, group_axes: Sequence[str]) -> np.ndarray:
        cache = getattr(self, "_step_cache", None)
        key = self._cache_key(step) if cache is not None else None
        if key is not None and key in cache:
            return cache[key]
        if step.chain:
            out = self._run_chain(step, group_axes)
        else:
            materialized = [self._materialize(o) for o in step.operands]
            if any(isinstance(arr, COOCube) for _, arr in materialized):
                out = self._run_sparse(step, materialized)
            else:
                arrays = [np.asarray(arr) for _, arr in materialized]
                out = np.asarray(self._einsum(step.spec, arrays, step.path),
                                 dtype=np.float64)
        if key is not None:
            cache[key] = out
        return out

    def _run_chain(self, step: EinsumStep, group_axes: Sequence[str]) -> np.ndarray:
        """Исполняет цепочку парных свёрток (план не помещался в 52 индекса)."""
        prev: np.ndarray | None = None
        assert step.chain is not None
        for sub in step.chain:
            arrays = []
            for o in sub.operands:
                if o.name.startswith("__tmp"):
                    if prev is None:
                        raise RuntimeError("нарушен порядок цепочки свёрток")
                    arrays.append(prev)
                else:
                    _, arr = self._materialize(o)
                    arrays.append(arr.to_dense().data if isinstance(arr, COOCube)
                                  else np.asarray(arr))
            prev = self._einsum(sub.spec, arrays, sub.path)
        return np.asarray(prev, dtype=np.float64)

    def _run_sparse(self, step: EinsumStep, materialized) -> np.ndarray:
        """Свёртка с разреженным операндом без материализации плотного куба."""
        sparse_positions = [i for i, (_, a) in enumerate(materialized)
                            if isinstance(a, COOCube)]
        if len(sparse_positions) > 1:
            # Несколько разреженных операндов: сворачиваем последовательно,
            # начиная с самого разреженного.
            sparse_positions.sort(key=lambda i: materialized[i][1].nnz)
        first = sparse_positions[0]
        order = [first] + [i for i in range(len(materialized)) if i != first]
        acc_axes, acc = materialized[order[0]]
        if not isinstance(acc, COOCube):
            acc = MultidimensionalMatrix(acc, acc_axes)

        for pos, i in enumerate(order[1:], start=1):
            axes, arr = materialized[i]
            later = {a for j in order[pos + 1:] for a in materialized[j][0]}
            needed = later | set(step.output)
            keep = (set(acc.axes) & set(axes)) & needed
            other = arr if isinstance(arr, COOCube) else MultidimensionalMatrix(arr, axes)
            if isinstance(acc, COOCube):
                acc = convolve_sparse(acc, other, keep=keep)
            else:
                dense_other = other.to_dense() if isinstance(other, COOCube) else other
                acc = convolve_named(acc, dense_other, keep=keep)
            drop = [a for a in acc.axes if a not in needed]
            if drop:
                acc = acc.project(drop)

        dense = acc.to_dense() if isinstance(acc, COOCube) else acc
        extra = [a for a in dense.axes if a not in step.output]
        if extra:
            dense = dense.project(extra)
        if step.output:
            dense = dense.transpose(tuple(step.output))
        return np.asarray(dense.data, dtype=np.float64)

    def _run_distinct(self, step, group_axes: Sequence[str]) -> np.ndarray:
        """COUNT DISTINCT: ∨-свёртка по лишним осям, затем подсчёт по оси значений.

        Маски применяются обнулением ячеек до ∨-свёртки: для булева полукольца
        умножение на ноль корректно, поскольку 0 — нейтраль ∨.
        """
        from ..core.semiring import count_distinct

        cube = self.catalog.cube(step.cube).dense()
        data = cube.data.astype(np.float64)
        for m in step.masks:
            idx: list[Any] = [None] * data.ndim
            for i, ax in enumerate(cube.axes):
                idx[i] = slice(None) if ax in m.axes else None
            mask = np.asarray(m.data, dtype=np.float64)
            order = [m.axes.index(ax) for ax in cube.axes if ax in m.axes]
            if len(order) > 1:
                mask = np.transpose(mask, order)
            data = data * mask[tuple(idx)]

        distinct_pos = cube.axes.index(step.distinct_axis)
        keep_pos = tuple(cube.axes.index(a) for a in step.output)
        return count_distinct(data, distinct_pos, keep_pos)

    def _run_reduce(self, step: ReduceStep, group_axes: Sequence[str]) -> np.ndarray:
        cube = self.catalog.cube(step.cube).dense()
        fill = -np.inf if step.how == "max" else np.inf
        data = cube.data.astype(np.float64)

        if step.masks:
            # Для MIN/MAX маскирование умножением на ноль некорректно: ноль стал бы
            # ложным экстремумом. Поэтому отфильтрованные ячейки заменяются на ±inf.
            keep = np.ones(data.shape, dtype=bool)
            for m in step.masks:
                bm = np.asarray(m.data, dtype=bool)
                idx: list[Any] = [None] * data.ndim
                for i, ax in enumerate(cube.axes):
                    idx[i] = slice(None) if ax in m.axes else None
                order = [m.axes.index(ax) for ax in cube.axes if ax in m.axes]
                bm = np.transpose(bm, order) if len(order) > 1 else bm
                keep &= bm[tuple(idx)]
            data = np.where(keep, data, fill)

        result = MultidimensionalMatrix(data, cube.axes)
        for op in step.rollups:
            child, parent = op.axes
            ordinals = np.argmax(np.asarray(op.array), axis=1)
            result = rollup_reduce(result, child, ordinals,
                                   int(np.asarray(op.array).shape[1]), parent, how=step.how)

        extra = [a for a in result.axes if a not in step.output]
        if extra:
            result = reduce_axes(result, extra, how=step.how)
        out = result.transpose(tuple(step.output)).data if step.output else result.data
        out = np.asarray(out, dtype=np.float64)
        out[~np.isfinite(out)] = 0.0
        return out

    # -- сборка результата --------------------------------------------------
    def _assemble(self, plan: PhysicalPlan, values: dict[int, np.ndarray],
                  presence: np.ndarray | None, group_axes: Sequence[str],
                  shape: tuple[int, ...]) -> ResultSet:
        select = plan.query.select
        n_cells = int(np.prod(shape)) if shape else 1

        if presence is None:
            flat = np.array([0])
        else:
            flat = np.flatnonzero(np.asarray(presence).reshape(-1) != 0)

        coords = (np.unravel_index(flat, shape) if shape
                  else tuple(np.zeros(1, dtype=np.int64) for _ in ()))

        columns: list[str] = []
        data_cols: list[Any] = []
        for item in select.items:
            if isinstance(item.expr, Star):
                for i, axis in enumerate(group_axes):
                    columns.append(axis)
                    data_cols.append(self._labels(axis, coords[i]))
                continue
            if isinstance(item.expr, Column):
                axis = item.expr.name
                i = list(group_axes).index(axis)
                columns.append(item.alias or axis)
                data_cols.append(self._labels(axis, coords[i]))
                continue
            arr = self._eval(item.expr, values, shape)
            columns.append(item.label)
            flat_arr = np.asarray(arr, dtype=np.float64).reshape(-1)
            if flat_arr.size == 1 and n_cells > 1:
                flat_arr = np.repeat(flat_arr, n_cells)
            data_cols.append(flat_arr[flat])

        # Пока не потребовалась фильтрация или сортировка, результат остаётся
        # в столбцовом виде: на широких выдачах сборка кортежей Python дороже
        # самой свёртки, и потребителю кадра pandas платить за неё незачем.
        needs_rows = (select.having is not None or select.order_by
                      or select.limit is not None)
        if not needs_rows:
            return ResultSet(columns, column_values=data_cols)

        rows = list(zip(*data_cols)) if data_cols else []
        if select.having is not None:
            rows = self._apply_having(select.having, columns, rows)
        rows = self._apply_order(select.order_by, columns, rows)
        if select.limit is not None:
            rows = rows[: select.limit]
        return ResultSet(columns, rows)

    def _labels(self, axis: str, ordinals: np.ndarray) -> np.ndarray:
        """Метки значений измерения. Возвращается массив, а не список Python:
        построение кадра pandas из массивов обходится без пересборки объектов."""
        dim = self.catalog.dimension(axis)
        table = np.empty(len(dim), dtype=object)
        table[:] = dim.labels()
        return table[np.asarray(ordinals, dtype=np.int64)]

    def _eval(self, expr: Expr, values: dict[int, np.ndarray],
              shape: tuple[int, ...]) -> Any:
        if isinstance(expr, Aggregate):
            return values[id(expr)]
        if isinstance(expr, Literal):
            return np.full(shape or (1,), float(expr.value))
        if isinstance(expr, BinOp):
            left = self._eval(expr.left, values, shape)
            right = self._eval(expr.right, values, shape)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                r = np.asarray(right, dtype=np.float64)
                l = np.asarray(left, dtype=np.float64)
                return np.divide(l, r, out=np.zeros_like(l, dtype=np.float64), where=r != 0)
        raise BindError(f"выражение {expr} не может быть вычислено над результатом")

    def _apply_having(self, cond: Condition, columns: list[str],
                      rows: list[tuple]) -> list[tuple]:
        def value_of(name: str, row: tuple) -> Any:
            if name in columns:
                return row[columns.index(name)]
            raise BindError(
                f"HAVING ссылается на '{name}', которого нет среди столбцов {columns}"
            )

        def test(c: Condition, row: tuple) -> bool:
            if isinstance(c, Logical):
                a, b = test(c.left, row), test(c.right, row)
                return (a and b) if c.op == "AND" else (a or b)
            if isinstance(c, Not):
                return not test(c.inner, row)
            if isinstance(c, Compare):
                if isinstance(c.value, Column):
                    raise BindError("в HAVING сравнение столбцов не поддерживается")
                v = value_of(c.column.name, row)
                ops = {"=": lambda x, y: x == y, "!=": lambda x, y: x != y,
                       "<": lambda x, y: x < y, "<=": lambda x, y: x <= y,
                       ">": lambda x, y: x > y, ">=": lambda x, y: x >= y}
                return bool(ops[c.op](v, c.value))
            raise BindError(f"условие {type(c).__name__} не поддерживается в HAVING")

        return [r for r in rows if test(cond, r)]

    def _apply_order(self, keys: Sequence[Any], columns: list[str],
                     rows: list[tuple]) -> list[tuple]:
        for key in reversed(list(keys)):
            if isinstance(key.key, int):
                idx = key.key - 1
                if not 0 <= idx < len(columns):
                    raise BindError(f"ORDER BY {key.key}: столбца с таким номером нет")
            elif key.key in columns:
                idx = columns.index(key.key)
            else:
                raise BindError(
                    f"ORDER BY '{key.key}': нет такого столбца среди {columns}"
                )
            rows = sorted(rows, key=lambda r, i=idx: _sort_key(r[i]),
                          reverse=key.descending)
        return rows


def _sort_key(v: Any) -> tuple[int, Any]:
    """Устойчивое сравнение разнотипных значений (числа < строки)."""
    if v is None:
        return (0, 0)
    if isinstance(v, (int, float, np.number)):
        return (1, float(v))
    return (2, str(v))
