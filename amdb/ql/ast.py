# -*- coding: utf-8 -*-
"""Абстрактное синтаксическое дерево SQL-подобного языка AMDB."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


# --- выражения -------------------------------------------------------------
class Expr:
    """Базовый узел выражения."""


@dataclass
class Column(Expr):
    """Ссылка на столбец: имя меры, измерения или атрибута (возможно qualified)."""

    name: str
    qualifier: str | None = None

    def __str__(self) -> str:
        return f"{self.qualifier}.{self.name}" if self.qualifier else self.name


@dataclass
class Literal(Expr):
    value: Any

    def __str__(self) -> str:
        return repr(self.value)


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class Star(Expr):
    def __str__(self) -> str:
        return "*"


@dataclass
class Case(Expr):
    """CASE WHEN условие THEN выражение [...] [ELSE выражение] END.

    В алгебре раскрывается через индикаторы: ветвь с условием ф даёт слагаемое
    вида (выражение) * х, где х — индикатор ф, а последующие ветви домножаются
    ещё и на дополнения предыдущих условий. Тем самым разбор случаев не требует
    ни ветвления при исполнении, ни дополнительного прохода по данным — только
    дополнительных сомножителей в том же терме.
    """

    branches: list[tuple["Condition", Expr]] = field(default_factory=list)
    otherwise: Expr | None = None

    def __str__(self) -> str:
        parts = [f"WHEN {c} THEN {e}" for c, e in self.branches]
        if self.otherwise is not None:
            parts.append(f"ELSE {self.otherwise}")
        return "CASE " + " ".join(parts) + " END"


@dataclass
class WindowSpec:
    partition_by: tuple[str, ...] = ()
    order_by: str | None = None
    frame_preceding: int | None = None   # None = UNBOUNDED


@dataclass
class Aggregate(Expr):
    func: str                    # SUM | AVG | COUNT | MIN | MAX
    arg: Expr
    distinct: bool = False
    window: WindowSpec | None = None

    def __str__(self) -> str:
        d = "DISTINCT " if self.distinct else ""
        base = f"{self.func}({d}{self.arg})"
        return base + (" OVER (...)" if self.window else "")


# --- условия ---------------------------------------------------------------
class Condition:
    """Базовый узел условия."""


@dataclass
class Compare(Condition):
    op: str                      # = != < <= > >=
    column: Column
    value: Any


@dataclass
class InList(Condition):
    column: Column
    values: tuple[Any, ...]
    negated: bool = False


@dataclass
class InSubquery(Condition):
    column: Column
    subquery: "Select"
    negated: bool = False


@dataclass
class Between(Condition):
    column: Column
    low: Any
    high: Any
    negated: bool = False


@dataclass
class IsNull(Condition):
    """IS [NOT] NULL: индикатор ординала отсутствующего значения."""

    column: Column
    negated: bool = False


@dataclass
class Logical(Condition):
    op: str                      # AND | OR
    left: Condition
    right: Condition


@dataclass
class Not(Condition):
    inner: Condition


# --- запрос ----------------------------------------------------------------
@dataclass
class SelectItem:
    expr: Expr
    alias: str | None = None

    @property
    def label(self) -> str:
        return self.alias or str(self.expr)


@dataclass
class Join:
    source: str
    on: Condition | None = None


@dataclass
class OrderKey:
    key: str | int
    descending: bool = False


@dataclass
class Select:
    items: list[SelectItem] = field(default_factory=list)
    source: str | None = None
    joins: list[Join] = field(default_factory=list)
    where: Condition | None = None
    group_by: tuple[str, ...] = ()
    having: Condition | None = None
    order_by: tuple[OrderKey, ...] = ()
    limit: int | None = None

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple([s for s in [self.source] if s] + [j.source for j in self.joins])

    def aggregates(self) -> list[Aggregate]:
        found: list[Aggregate] = []

        def walk(e: Expr) -> None:
            if isinstance(e, Aggregate):
                found.append(e)
                walk(e.arg)
            elif isinstance(e, BinOp):
                walk(e.left)
                walk(e.right)

        for item in self.items:
            walk(item.expr)
        return found

    def plain_columns(self) -> list[Column]:
        found: list[Column] = []

        def walk(e: Expr, inside_agg: bool) -> None:
            if isinstance(e, Aggregate):
                walk(e.arg, True)
            elif isinstance(e, BinOp):
                walk(e.left, inside_agg)
                walk(e.right, inside_agg)
            elif isinstance(e, Column) and not inside_agg:
                found.append(e)

        for item in self.items:
            walk(item.expr, False)
        return found
