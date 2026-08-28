# -*- coding: utf-8 -*-
"""Связывание AST с каталогом: разрешение имён, нормализация выражений, маски.

Здесь же происходит главное преобразование модели: условия WHERE становятся
0/1-масками по осям, а выражения под агрегатами раскладываются в линейную
комбинацию произведений мер — именно та форма, которую принимает einsum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..storage.catalog import Catalog
from ..storage.dimension import Dimension
from .ast import (
    Aggregate,
    Between,
    BinOp,
    Column,
    Compare,
    Condition,
    Expr,
    InList,
    InSubquery,
    Literal,
    Logical,
    Not,
    Select,
    Star,
)


class BindError(ValueError):
    """Запрос синтаксически корректен, но не согласуется с каталогом."""


@dataclass
class MaskTensor:
    """0/1-маска по одной или нескольким осям (операнд einsum)."""

    axes: tuple[str, ...]
    data: np.ndarray

    def __post_init__(self) -> None:
        self.axes = tuple(self.axes)
        self.data = np.asarray(self.data, dtype=np.float32)

    def __repr__(self) -> str:
        return f"Mask{self.axes} selected={int(self.data.sum())}/{self.data.size}"


@dataclass
class Term:
    """Слагаемое нормализованного выражения: coef * произведение мер."""

    coef: float
    measures: tuple[str, ...] = ()

    def __repr__(self) -> str:
        body = " * ".join(self.measures) or "1"
        return f"{self.coef:g}*{body}" if self.coef != 1 else body


@dataclass
class BoundAggregate:
    func: str
    terms: list[Term]
    alias: str
    distinct: bool = False
    window: Any = None
    key: str = ""


@dataclass
class LogicalQuery:
    select: Select
    sources: tuple[str, ...]
    group_axes: tuple[str, ...]
    filters: list[MaskTensor]
    aggregates: list[BoundAggregate]
    catalog: Catalog = field(repr=False, default=None)  # type: ignore[assignment]

    def describe(self) -> str:
        lines = [f"источники: {', '.join(self.sources) or '-'}",
                 f"GROUP BY: {', '.join(self.group_axes) or '(нет — скалярный агрегат)'}"]
        for f_ in self.filters:
            lines.append(f"фильтр: {f_!r}")
        for a in self.aggregates:
            lines.append(f"агрегат: {a.func}({' + '.join(map(str, a.terms))}) AS {a.alias}")
        return "\n".join(lines)


# --- нормализация выражений ------------------------------------------------
def normalize(expr: Expr) -> list[Term]:
    """Раскладывает выражение в сумму произведений мер с коэффициентами.

    SUM(quantity * price - discount) -> [1*quantity*price, -1*discount],
    то есть две независимые свёртки, результаты которых складываются.
    """
    if isinstance(expr, Literal):
        if not isinstance(expr.value, (int, float)):
            raise BindError(f"нечисловая константа {expr.value!r} под агрегатом")
        return [Term(float(expr.value))]
    if isinstance(expr, Star):
        return [Term(1.0)]
    if isinstance(expr, Column):
        return [Term(1.0, (expr.name,))]
    if isinstance(expr, Aggregate):
        raise BindError("вложенные агрегаты не поддерживаются")
    if isinstance(expr, BinOp):
        if expr.op in ("+", "-"):
            left = normalize(expr.left)
            right = normalize(expr.right)
            sign = 1.0 if expr.op == "+" else -1.0
            return left + [Term(t.coef * sign, t.measures) for t in right]
        if expr.op == "*":
            out: list[Term] = []
            for a in normalize(expr.left):
                for b in normalize(expr.right):
                    out.append(Term(a.coef * b.coef, a.measures + b.measures))
            return out
        if expr.op == "/":
            denom = normalize(expr.right)
            if len(denom) != 1 or denom[0].measures:
                raise BindError(
                    "деление на меру внутри агрегата не выражается через свёртку; "
                    "вычислите отношение агрегатов: SUM(a) / SUM(b)"
                )
            if denom[0].coef == 0:
                raise BindError("деление на ноль в выражении")
            return [Term(t.coef / denom[0].coef, t.measures) for t in normalize(expr.left)]
    raise BindError(f"выражение {expr} не поддерживается под агрегатом")


# --- разрешение столбцов ---------------------------------------------------
def resolve_dimension(catalog: Catalog, col: Column) -> tuple[Dimension, str | None]:
    """Столбец -> (измерение, имя атрибута или None, если это само измерение)."""
    if col.qualifier and col.qualifier in catalog.dimensions:
        dim = catalog.dimension(col.qualifier)
        if col.name in dim.attributes:
            return dim, col.name
        if col.name == dim.name:
            return dim, None
        raise BindError(f"измерение '{dim.name}' не имеет атрибута '{col.name}'")
    if col.name in catalog.dimensions:
        return catalog.dimension(col.name), None
    owners = [d for d in catalog.dimensions.values() if col.name in d.attributes]
    if len(owners) == 1:
        return owners[0], col.name
    if len(owners) > 1:
        raise BindError(
            f"атрибут '{col.name}' есть у измерений {[d.name for d in owners]}; "
            "уточните запись как измерение.атрибут"
        )
    raise BindError(
        f"неизвестный столбец '{col}'; известные измерения: {sorted(catalog.dimensions)}"
    )


# --- условия -> маски ------------------------------------------------------
def bind_condition(cond: Condition | None, catalog: Catalog) -> list[MaskTensor]:
    """Условие -> список масок, соединяемых конъюнктивно."""
    if cond is None:
        return []
    if isinstance(cond, Logical) and cond.op == "AND":
        return bind_condition(cond.left, catalog) + bind_condition(cond.right, catalog)
    if isinstance(cond, Logical) and cond.op == "OR":
        left = _merge(bind_condition(cond.left, catalog), catalog)
        right = _merge(bind_condition(cond.right, catalog), catalog)
        return [_or(left, right)]
    if isinstance(cond, Not):
        inner = _merge(bind_condition(cond.inner, catalog), catalog)
        return [MaskTensor(inner.axes, 1.0 - inner.data)]
    return [_bind_predicate(cond, catalog)]


def _attribute_mask(dim: Dimension, attr: str, predicate, description: str) -> np.ndarray:
    """Маска по атрибуту с внятной ошибкой при несравнимых типах.

    Без этого сравнение текстового атрибута с числом всплывало бы наружу как
    голое TypeError из недр Python.
    """
    try:
        return dim.attribute_mask(attr, predicate)
    except TypeError as e:
        sample = next((v for v in dim.attributes[attr] if v is not None), None)
        raise BindError(
            f"условие {description} неприменимо к атрибуту '{attr}' измерения "
            f"'{dim.name}': его значения имеют тип {type(sample).__name__} "
            f"(например, {sample!r})"
        ) from e


def _bind_predicate(cond: Condition, catalog: Catalog) -> MaskTensor:
    if isinstance(cond, Compare):
        if isinstance(cond.value, Column):
            # Условие соединения (ON a.x = b.x): соединение в матричной модели
            # выполняется по совпадению осей, отдельная маска не нужна.
            _check_join_condition(cond, catalog)
            return MaskTensor((), np.ones((), dtype=np.float32))
        dim, attr = resolve_dimension(catalog, cond.column)
        return MaskTensor((dim.name,), _compare_mask(dim, attr, cond.op, cond.value))
    if isinstance(cond, InList):
        dim, attr = resolve_dimension(catalog, cond.column)
        if attr is None:
            m = dim.mask_of(cond.values)
        else:
            wanted = set(cond.values)
            m = dim.attribute_mask(attr, lambda v: v in wanted)
        return MaskTensor((dim.name,), 1.0 - m if cond.negated else m)
    if isinstance(cond, Between):
        dim, attr = resolve_dimension(catalog, cond.column)
        if attr is None:
            m = dim.mask_range(cond.low, cond.high)
        else:
            m = _attribute_mask(
                dim, attr, lambda v: v is not None and cond.low <= v <= cond.high,
                f"BETWEEN {cond.low!r} AND {cond.high!r}")
        return MaskTensor((dim.name,), 1.0 - m if cond.negated else m)
    if isinstance(cond, InSubquery):
        return _bind_subquery(cond, catalog)
    raise BindError(f"условие {type(cond).__name__} не поддерживается")


def _check_join_condition(cond: Compare, catalog: Catalog) -> None:
    left, right = cond.column, cond.value
    if cond.op != "=":
        raise BindError("условие соединения должно быть равенством")
    if left.name != right.name:
        raise BindError(
            f"соединение по разным измерениям ('{left.name}' и '{right.name}') "
            "в многомерно-матричной модели не определено: оси соединяются по имени"
        )
    if left.name not in catalog.dimensions:
        raise BindError(f"неизвестное измерение соединения '{left.name}'")


def _compare_mask(dim: Dimension, attr: str | None, op: str, value: Any) -> np.ndarray:
    if attr is not None:
        table = {
            "=": lambda v: v == value,
            "!=": lambda v: v != value,
            "<": lambda v: v is not None and v < value,
            "<=": lambda v: v is not None and v <= value,
            ">": lambda v: v is not None and v > value,
            ">=": lambda v: v is not None and v >= value,
        }
        return _attribute_mask(dim, attr, table[op], f"{op} {value!r}")
    if op == "=":
        return dim.mask_of([value])
    if op == "!=":
        return 1.0 - dim.mask_of([value])
    inclusive = op in ("<=", ">=")
    if op in ("<", "<="):
        return dim.mask_range(high=value, inclusive=(True, inclusive))
    return dim.mask_range(low=value, inclusive=(inclusive, True))


def _bind_subquery(cond: InSubquery, catalog: Catalog) -> MaskTensor:
    """IN (SELECT key FROM ... WHERE ...) -> маска по измерению внешнего столбца."""
    outer_dim, outer_attr = resolve_dimension(catalog, cond.column)
    if outer_attr is not None:
        raise BindError("подзапрос в IN допустим только для измерения, не для атрибута")
    sub = cond.subquery
    if sub.group_by or sub.aggregates() or sub.joins:
        raise BindError(
            "поддерживается только простой подзапрос вида "
            "IN (SELECT <измерение> FROM <источник> WHERE <условие>)"
        )
    masks = bind_condition(sub.where, catalog)
    merged = _merge(masks, catalog)
    if merged.axes and merged.axes != (outer_dim.name,):
        raise BindError(
            f"подзапрос ограничивает оси {merged.axes}, а внешний столбец — "
            f"'{outer_dim.name}'"
        )
    data = merged.data if merged.axes else outer_dim.mask_all()
    return MaskTensor((outer_dim.name,), 1.0 - data if cond.negated else data)


def _merge(masks: Sequence[MaskTensor], catalog: Catalog) -> MaskTensor:
    """Сводит конъюнкцию масок к одному тензору (нужно для OR и NOT)."""
    if not masks:
        return MaskTensor((), np.ones((), dtype=np.float32))
    out = masks[0]
    for m in masks[1:]:
        out = _combine(out, m, lambda a, b: a * b)
    return out


def _or(a: MaskTensor, b: MaskTensor) -> MaskTensor:
    return _combine(a, b, lambda x, y: 1.0 - (1.0 - x) * (1.0 - y))


def _combine(a: MaskTensor, b: MaskTensor, op) -> MaskTensor:
    axes = list(a.axes) + [x for x in b.axes if x not in a.axes]
    return MaskTensor(tuple(axes), op(_expand(a, axes), _expand(b, axes)))


def _expand(m: MaskTensor, axes: Sequence[str]) -> np.ndarray:
    data = m.data
    for i, name in enumerate(axes):
        if name not in m.axes:
            data = np.expand_dims(data, i)
    order = [m.axes.index(n) for n in axes if n in m.axes]
    if order and order != sorted(order):
        perm = [axes.index(n) for n in m.axes] + [
            i for i, n in enumerate(axes) if n not in m.axes]
        data = np.moveaxis(data, list(range(data.ndim)), perm)
    return data


# --- связывание запроса ----------------------------------------------------
def bind(select: Select, catalog: Catalog) -> LogicalQuery:
    """AST -> логический запрос, согласованный с каталогом."""
    sources = select.sources
    for s in sources:
        if s not in catalog.cubes and s not in catalog.dimensions:
            raise BindError(
                f"источник '{s}' не найден: нет ни куба, ни измерения с таким именем"
            )

    group_axes: list[str] = []
    for name in select.group_by:
        col = Column(*reversed(name.split("."))) if "." in name else Column(name)
        dim, attr = resolve_dimension(catalog, col)
        if attr is not None:
            raise BindError(
                f"GROUP BY по атрибуту '{attr}' не поддерживается; "
                f"создайте иерархию {dim.name} -> {attr}"
            )
        if dim.name in group_axes:
            raise BindError(f"измерение '{dim.name}' указано в GROUP BY дважды")
        group_axes.append(dim.name)

    plain = [c.name for c in select.plain_columns()]
    for name in plain:
        if name not in group_axes:
            raise BindError(
                f"столбец '{name}' присутствует в SELECT, но отсутствует в GROUP BY"
            )

    filters = [m for m in bind_condition(select.where, catalog) if m.axes]
    for j in select.joins:
        for m in bind_condition(j.on, catalog):
            if m.axes:
                filters.append(m)

    aggregates: list[BoundAggregate] = []
    for i, agg in enumerate(select.aggregates()):
        if agg.func not in ("SUM", "AVG", "COUNT", "MIN", "MAX"):
            raise BindError(f"неизвестная агрегатная функция '{agg.func}'")
        if agg.distinct and agg.func != "COUNT":
            raise BindError(f"DISTINCT допустим только в COUNT, не в {agg.func}")
        if agg.distinct:
            # COUNT(DISTINCT d) выразим, но лишь когда различаются значения
            # ИЗМЕРЕНИЯ: ось d есть область различаемых значений. Для DISTINCT
            # по произвольной мере пришлось бы вынести её область значений
            # в отдельную ось, чего модель данных не предусматривает.
            if not isinstance(agg.arg, Column):
                raise BindError(
                    "COUNT(DISTINCT ...) поддерживается только для измерения, "
                    "а не для выражения"
                )
            dim, attr = resolve_dimension(catalog, agg.arg)
            if attr is not None:
                raise BindError(
                    f"COUNT(DISTINCT {attr}) по атрибуту не поддерживается: "
                    f"создайте иерархию {dim.name} -> {attr}"
                )
            if dim.name in group_axes:
                raise BindError(
                    f"измерение '{dim.name}' не может быть одновременно в GROUP BY "
                    "и под DISTINCT"
                )
            aggregates.append(
                BoundAggregate("COUNT_DISTINCT", [Term(1.0, (dim.name,))],
                               _alias_for(select, agg, i), True, agg.window,
                               key=f"__agg{i}")
            )
            continue
        terms = normalize(agg.arg)
        if agg.func in ("MIN", "MAX"):
            if len(terms) != 1 or len(terms[0].measures) != 1 or terms[0].coef != 1:
                raise BindError(
                    f"{agg.func} поддерживается только для одной меры без арифметики: "
                    "это редукция, а не свёртка"
                )
        aggregates.append(
            BoundAggregate(agg.func, terms, _alias_for(select, agg, i), agg.distinct,
                           agg.window, key=f"__agg{i}")
        )

    if not aggregates and not group_axes:
        raise BindError("запрос без агрегатов и без GROUP BY не имеет смысла для гиперкуба")

    return LogicalQuery(select, sources, tuple(group_axes), filters, aggregates, catalog)


def _alias_for(select: Select, agg: Aggregate, index: int) -> str:
    for item in select.items:
        if item.expr is agg and item.alias:
            return item.alias
    return str(agg)
