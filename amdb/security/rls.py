# -*- coding: utf-8 -*-
"""Разграничение доступа на уровне измерений.

В матричной модели ограничение доступа — ещё один операнд-маска в той же
свёртке, а не фильтр после вычисления: запрещённые данные не участвуют
в арифметике вовсе.

Важное ограничение: маскирование умножением на ноль корректно для SUM и COUNT,
но не для MIN/MAX (ноль стал бы ложным экстремумом) и не для AVG (искажается
знаменатель). Для редукций права применяются выборкой разрешённых индексов —
это делает ReduceStep, получая те же маски.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..ql.binder import MaskTensor
from ..ql.planner import ARRAY, AggPlan, EinsumStep, Operand, PhysicalPlan


class AccessDenied(PermissionError):
    """Роль не имеет права на измерение или на агрегацию по нему."""


def apply_rls(plan: PhysicalPlan, catalog: Any, role: Any | None) -> PhysicalPlan:
    """Дополняет план масками прав доступа. Без роли план не меняется."""
    if role is None:
        return plan
    for agg in plan.aggregates:
        for _, step in agg.terms:
            _guard_step(step, catalog, role)
        if agg.denominator is not None:
            _guard_step(agg.denominator, catalog, role)
        if agg.reduce is not None:
            _guard_reduce(agg, catalog, role)
    if plan.presence is not None:
        _guard_step(plan.presence, catalog, role)
    return plan


def _axes_of(step: EinsumStep) -> list[str]:
    seen: list[str] = []
    for o in step.operands:
        for a in o.axes:
            if a not in seen:
                seen.append(a)
    return seen


def _grant_or_deny(catalog: Any, role: Any, axis: str, output: tuple[str, ...]):
    grant = catalog.grant(role, axis)
    name = getattr(role, "name", role)
    if grant is None:
        raise AccessDenied(f"роль '{name}' не имеет доступа к измерению '{axis}'")
    if axis not in output and not grant.can_project:
        raise AccessDenied(
            f"роль '{name}' не может агрегировать по измерению '{axis}': "
            "оно должно присутствовать в GROUP BY"
        )
    return grant


def _guard_step(step: EinsumStep, catalog: Any, role: Any) -> None:
    for axis in _axes_of(step):
        grant = _grant_or_deny(catalog, role, axis, step.output)
        if grant.allowed is None:
            continue
        step.operands.append(
            Operand(ARRAY, f"rls:{axis}", (axis,), np.asarray(grant.allowed, np.float32))
        )
    step.spec = ""
    step.__post_init__()
    step.path = None


def _guard_reduce(agg: AggPlan, catalog: Any, role: Any) -> None:
    reduce_step = agg.reduce
    assert reduce_step is not None
    cube = catalog.cube(reduce_step.cube)
    for axis in cube.axes:
        grant = _grant_or_deny(catalog, role, axis, reduce_step.output)
        if grant.allowed is None:
            continue
        reduce_step.masks.append(
            MaskTensor((axis,), np.asarray(grant.allowed, np.float32))
        )


def cell_suppression(values: np.ndarray, counts: np.ndarray,
                     k: int = 5) -> np.ndarray:
    """Подавление ячеек, построенных менее чем по k записям.

    Защита от косвенного раскрытия: агрегат по одной записи её и раскрывает.
    """
    return np.where(np.asarray(counts) >= k, values, np.nan)
