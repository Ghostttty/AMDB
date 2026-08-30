# -*- coding: utf-8 -*-
"""Построение плана матричных операций из логического запроса.

Главное преобразование техпроекта: соединение по общим измерениям, фильтрация
и агрегация сливаются в один вызов einsum. Оси, попавшие в GROUP BY, остаются
(это λ-индексы); все прочие общие оси суммируются (μ-индексы).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..core.convolve import TooManyIndicesError, build_spec
from ..storage.catalog import Catalog, Cube
from .binder import BindError, BoundAggregate, LogicalQuery, MaskTensor, Term

CUBE = "cube"
INDICATOR = "indicator"
ARRAY = "array"


@dataclass
class Operand:
    """Операнд einsum: куб каталога, его индикатор или готовый массив.

    ``presum`` — оси, по которым операнд сворачивается **до** входа в
    произведение. Это (0, μ)-свёртка с матрицей единиц, и по теореме о
    разложении она результата не меняет, зато резко сокращает работу: einsum
    подбирает порядок между операндами, но собственные «висячие» оси операнда
    сам не сворачивает.
    """

    kind: str
    name: str
    axes: tuple[str, ...]
    array: np.ndarray | None = None
    presum: tuple[str, ...] = ()

    def __repr__(self) -> str:
        tail = f" − {list(self.presum)}" if self.presum else ""
        return f"{self.name}{list(self.axes)}{tail}"


@dataclass
class EinsumStep:
    """Одна элементарная (λ, μ)-свёртка — цепочка операндов в одном einsum."""

    operands: list[Operand]
    output: tuple[str, ...]
    spec: str = ""
    path: Any = None
    #: Цепочка парных свёрток, если план не помещается в 52 индекса einsum.
    chain: list["EinsumStep"] | None = None

    def __post_init__(self) -> None:
        if not self.spec:
            try:
                self.spec = build_spec([o.axes for o in self.operands], self.output)
            except TooManyIndicesError:
                # Спецификация строится позже, после декомпозиции в оптимизаторе.
                self.spec = ""

    @property
    def lam_mu(self) -> tuple[int, int]:
        """(λ, μ): общих осей в выходе и свёрнутых общих осей."""
        counts: dict[str, int] = {}
        for o in self.operands:
            for a in set(o.axes):
                counts[a] = counts.get(a, 0) + 1
        shared = [a for a, c in counts.items() if c > 1]
        lam = sum(1 for a in shared if a in self.output)
        return lam, len(shared) - lam

    def describe(self) -> str:
        if self.chain:
            inner = "\n      ".join(s.describe() for s in self.chain)
            return f"цепочка из {len(self.chain)} свёрток:\n      {inner}"
        lam, mu = self.lam_mu
        ops = ", ".join(repr(o) for o in self.operands)
        return f"einsum('{self.spec}')  ({lam},{mu})-свёртка  [{ops}]"


@dataclass
class BinaryProduct:
    """Один шаг разложения плана в бинарное (λ, μ)-свёрнутое произведение."""

    left: tuple[str, ...]
    right: tuple[str, ...]
    lam_axes: tuple[str, ...]
    mu_axes: tuple[str, ...]
    result: tuple[str, ...]

    @property
    def lam(self) -> int:
        return len(self.lam_axes)

    @property
    def mu(self) -> int:
        return len(self.mu_axes)

    #: Правый операнд — матрица из единиц (суммирование по осям как (0,μ)-свёртка).
    right_is_ones: bool = False
    #: Имена операндов-листьев; None — промежуточный результат.
    left_label: str | None = None
    right_label: str | None = None

    def describe(self) -> str:
        right = (f"𝟙{list(self.right)}" if self.right_is_ones else f"{list(self.right)}")
        return (f"{list(self.left)} ∗ {right} -> {list(self.result)}"
                f"   ({self.lam},{self.mu})-свёртка"
                f"{'; λ: ' + ', '.join(self.lam_axes) if self.lam_axes else ''}"
                f"{'; μ: ' + ', '.join(self.mu_axes) if self.mu_axes else ''}")


def binary_decomposition(step: "EinsumStep") -> list[BinaryProduct]:
    """Разложение einsum-шага в цепочку бинарных (λ, μ)-произведений.

    Многооперандный einsum не является операцией алгебры Соколова: она бинарна.
    Но выбранный оптимизатором порядок свёрток задаёт разложение плана в
    композицию бинарных (λ, μ)-произведений — эта функция его и предъявляет.
    Роли индексов определяются по определению: общая ось, нужная дальше или
    попадающая в результат, — это λ (скоттов индекс); общая ось, исчезающая на
    этом шаге, — μ (кэлиев индекс).
    """
    if step.chain:
        out: list[BinaryProduct] = []
        for sub in step.chain:
            out.extend(binary_decomposition(sub))
        return out

    chain_head: list[BinaryProduct] = []
    labels: list[str | None] = [o.name for o in step.operands]
    for o in step.operands:
        if o.presum:
            full = tuple(o.axes) + tuple(o.presum)
            chain_head.append(
                BinaryProduct(full, tuple(o.presum), (), tuple(o.presum),
                              tuple(o.axes), right_is_ones=True, left_label=o.name))

    operands = [tuple(o.axes) for o in step.operands]
    output = set(step.output)
    if len(operands) == 1:
        # Суммирование по осям — это (0,μ)-свёртка с матрицей из единиц.
        axes = operands[0]
        kept = tuple(a for a in axes if a in output)
        dropped = tuple(a for a in axes if a not in output)
        if not dropped:
            return chain_head
        return chain_head + [
            BinaryProduct(axes, dropped, (), dropped, kept, right_is_ones=True,
                          left_label=labels[0])]

    path = step.path[1:] if step.path and len(step.path) > 1 else [
        tuple(range(len(operands)))]
    chain: list[BinaryProduct] = []
    for group in path:
        picked = [operands[i] for i in sorted(group)]
        picked_labels = [labels[i] for i in sorted(group)]
        for i in sorted(group, reverse=True):
            operands.pop(i)
            labels.pop(i)
        survivors = {a for o in operands for a in o} | output
        acc, acc_label = picked[0], picked_labels[0]
        for nxt, nxt_label in zip(picked[1:], picked_labels[1:]):
            common = [a for a in acc if a in nxt]
            lam_axes = tuple(a for a in common if a in survivors)
            mu_axes = tuple(a for a in common if a not in survivors)
            result = (tuple(a for a in acc if a not in common) + lam_axes
                      + tuple(a for a in nxt if a not in common))
            chain.append(BinaryProduct(acc, nxt, lam_axes, mu_axes, result,
                                       left_label=acc_label, right_label=nxt_label))
            acc, acc_label = result, None
        # Свободные оси в (λ, μ)-произведении по определению сохраняются, поэтому
        # их исчезновение — отдельная операция: (0,μ)-свёртка с матрицей единиц.
        # einsum выполняет её тем же вызовом, но алгебраически это второй шаг.
        dropped = tuple(a for a in acc if a not in survivors)
        if dropped:
            kept = tuple(a for a in acc if a in survivors)
            chain.append(BinaryProduct(acc, dropped, (), dropped, kept,
                                       right_is_ones=True, left_label=acc_label))
            acc, acc_label = kept, None
        operands.append(acc)
        labels.append(acc_label)
    return chain_head + chain


def term_expression(step: "EinsumStep", operand_names: dict | None = None) -> str:
    """Записывает план как терм алгебры: вложенные (λ, μ)-произведения.

    Терм — элемент второй основы двухосновной алгебраической системы: данные
    (гиперкубы) и операции над ними заданы разными сортами, и оптимизация есть
    переписывание терма, сохраняющее интерпретацию.
    """
    chain = binary_decomposition(step)
    if not chain:
        return step.operands[0].name if step.operands else "?"
    expr: dict[tuple[str, ...], str] = {}
    text = ""
    for bp in chain:
        left = bp.left_label or expr.get(bp.left, "?")
        right = ("𝟙" if bp.right_is_ones
                 else (bp.right_label or expr.get(bp.right, "?")))
        text = f"({left} ∗[{bp.lam},{bp.mu}] {right})"
        expr[bp.result] = text
    return text


@dataclass
class ReduceStep:
    """Редукция MIN/MAX — вне einsum: не-линейные агрегаты не свёртываются."""

    cube: str
    masks: list[MaskTensor]
    rollups: list[Operand]
    output: tuple[str, ...]
    how: str

    def describe(self) -> str:
        return (f"reduce({self.how}) куб {self.cube} -> {list(self.output)}"
                f"{' с масками' if self.masks else ''}")


@dataclass
class DistinctStep:
    """COUNT DISTINCT: композиция ∨-свёртки и подсчёта (см. §5 обоснования)."""

    cube: str
    distinct_axis: str
    masks: list[MaskTensor]
    output: tuple[str, ...]

    def describe(self) -> str:
        return (f"count_distinct({self.distinct_axis}) куб {self.cube} -> "
                f"{list(self.output)}   [(∨,∧)-свёртка, затем подсчёт]")


@dataclass
class AggPlan:
    func: str
    alias: str
    key: str
    terms: list[tuple[float, EinsumStep]] = field(default_factory=list)
    denominator: EinsumStep | None = None
    reduce: ReduceStep | None = None
    distinct: "DistinctStep | None" = None
    window: Any = None

    def describe(self) -> str:
        lines = [f"{self.alias}: {self.func}"]
        for coef, step in self.terms:
            lines.append(f"    {coef:+g} × {step.describe()}")
        if self.denominator is not None:
            lines.append(f"    ÷ {self.denominator.describe()}")
        if self.reduce is not None:
            lines.append(f"    {self.reduce.describe()}")
        if self.distinct is not None:
            lines.append(f"    {self.distinct.describe()}")
        if self.window is not None:
            lines.append(f"    окно: {self.window}")
        return "\n".join(lines)


@dataclass
class PhysicalPlan:
    query: LogicalQuery
    group_axes: tuple[str, ...]
    aggregates: list[AggPlan]
    presence: EinsumStep | None

    def explain(self) -> str:
        lines = ["План выполнения:",
                 f"  GROUP BY: {list(self.group_axes) or '(скаляр)'}"]
        for a in self.aggregates:
            lines.append("  " + a.describe().replace("\n", "\n  "))
        if self.presence is not None:
            lines.append(f"  наличие групп: {self.presence.describe()}")
        return "\n".join(lines)


def plan_query(logical: LogicalQuery, catalog: Catalog) -> PhysicalPlan:
    """Логический запрос -> физический план (DAG einsum-операций)."""
    primary = _primary_cube(logical, catalog)
    aggregates: list[AggPlan] = []

    for agg in logical.aggregates:
        plan = AggPlan(agg.func, agg.alias, agg.key, window=agg.window)
        if agg.func in ("MIN", "MAX"):
            plan.reduce = _plan_reduce(agg, logical, catalog)
        elif agg.func == "COUNT_DISTINCT":
            plan.distinct = _plan_distinct(agg, logical, catalog, primary)
        elif agg.func == "COUNT":
            plan.terms = [(1.0, _plan_count(agg, logical, catalog, primary))]
        else:
            for term in agg.terms:
                plan.terms.append(
                    (term.coef, _plan_term(term, logical, catalog, primary))
                )
            if agg.func == "AVG":
                plan.denominator = _plan_count(agg, logical, catalog, primary)
        aggregates.append(plan)

    presence = _plan_count(None, logical, catalog, primary) if logical.group_axes else None
    return PhysicalPlan(logical, logical.group_axes, aggregates, presence)


# --- построение отдельных шагов -------------------------------------------
def _primary_cube(logical: LogicalQuery, catalog: Catalog) -> Cube:
    """Основной куб факта: по нему считаются COUNT(*) и наличие групп."""
    for s in logical.sources:
        if s in catalog.cubes:
            return catalog.cube(s)
    for agg in logical.aggregates:
        for term in agg.terms:
            for m in term.measures:
                return _cube_for(catalog, m, logical.sources)
    raise BindError("не удалось определить основной куб факта для запроса")


def _cube_for(catalog: Catalog, measure: str, sources: Sequence[str]) -> Cube:
    if measure in catalog.cubes:
        return catalog.cube(measure)
    candidates = [s for s in sources if s in catalog.cubes] or None
    try:
        return catalog.cube_for_measure(measure, candidates)
    except KeyError:
        return catalog.cube_for_measure(measure, None)


def _plan_term(term: Term, logical: LogicalQuery, catalog: Catalog,
               primary: Cube) -> EinsumStep:
    if not term.measures:
        # SUM(константа) = константа × число строк; при разборе случаев к этому
        # добавляются индикаторы ветви, поэтому считать надо не все строки, а
        # попавшие под условие.
        step = _plan_count(None, logical, catalog, primary)
        return _with_masks(step, term, catalog)
    operands = [
        Operand(CUBE, c.name, c.axes)
        for c in (_cube_for(catalog, m, logical.sources) for m in term.measures)
    ]
    step = _assemble(operands, logical, catalog)
    return _with_masks(step, term, catalog)


def _with_masks(step: EinsumStep, term: Term, catalog: Catalog) -> EinsumStep:
    """Добавляет к шагу индикаторы ветвей разбора случаев.

    Ветвление в алгебре есть умножение на индикатор: слагаемое, отвечающее
    ветви, домножается на её условие и на дополнения условий предыдущих ветвей.
    Дополнительного прохода по данным это не требует — только дополнительных
    сомножителей в том же стягивании.
    """
    if not term.masks:
        return step
    available = {a for o in step.operands for a in o.axes}
    operands = list(step.operands)
    for i, mask in enumerate(term.masks):
        missing = [a for a in mask.axes if a not in available]
        if missing:
            raise BindError(
                f"условие ветви CASE по {missing} неприменимо: измерения нет "
                f"ни в одном операнде ({sorted(available)})"
            )
        operands.append(Operand(ARRAY, f"case{i}:{'/'.join(mask.axes)}",
                                mask.axes, mask.data))
    return EinsumStep(operands, step.output)


def _counting_operand(cube: Cube, catalog: Catalog) -> Operand:
    """Счётный куб факта, если он загружен, иначе индикатор непустых ячеек."""
    counter = f"{cube.name}__count"
    if counter in catalog.cubes:
        c = catalog.cube(counter)
        return Operand(CUBE, c.name, c.axes)
    return Operand(INDICATOR, cube.name, cube.axes)


def _plan_count(agg: BoundAggregate | None, logical: LogicalQuery, catalog: Catalog,
                primary: Cube) -> EinsumStep:
    """COUNT: свёртка счётного куба, если он загружен, иначе индикаторного.

    Когда меры не названы — это COUNT(*) либо служебный шаг наличия групп, —
    берутся счётные кубы **всех** фактов запроса, а не одного основного.
    Иначе запрос по двум фактам с группировкой по измерению второго из них
    отвергался бы: у основного куба такой оси нет. Произведение счётных кубов
    и есть число пар соединения, то есть ровно семантика COUNT(*) по
    соединению; для шага наличия групп существенно лишь то, что оно отлично
    от нуля тогда и только тогда, когда факт есть в каждом из фактов.
    """
    measures = agg.terms[0].measures if (agg and agg.terms) else ()
    if measures:
        # Знаменатель AVG по соединению — число пар, а не число строк одного
        # факта, поэтому берутся кубы всех мер терма. Повторы имён отбрасываются:
        # у AVG(q * q) знаменатель тот же, что у AVG(q).
        cubes = [_cube_for(catalog, m, logical.sources) for m in measures]
    else:
        cubes = [catalog.cube(s) for s in logical.sources if s in catalog.cubes]
        if not cubes:
            cubes = [primary]
    seen: set[str] = set()
    unique = [c for c in cubes if not (c.name in seen or seen.add(c.name))]
    operands = [_counting_operand(c, catalog) for c in unique]
    return _assemble(operands, logical, catalog)


def _plan_reduce(agg: BoundAggregate, logical: LogicalQuery,
                 catalog: Catalog) -> ReduceStep:
    measure = agg.terms[0].measures[0]
    cube = _cube_for(catalog, measure, logical.sources)
    rollups, available = _rollups_for(list(cube.axes), logical, catalog)
    missing = [a for a in logical.group_axes if a not in available]
    if missing:
        raise BindError(
            f"{agg.func}: измерения {missing} недостижимы из осей куба "
            f"'{cube.name}' {list(cube.axes)}"
        )
    for m in logical.filters:
        for a in m.axes:
            if a not in cube.axes:
                raise BindError(
                    f"{agg.func}: фильтр по '{a}' неприменим — измерение отсутствует "
                    f"в кубе '{cube.name}'"
                )
    return ReduceStep(cube.name, list(logical.filters), rollups,
                      logical.group_axes, agg.func.lower())


def _plan_distinct(agg: BoundAggregate, logical: LogicalQuery, catalog: Catalog,
                   primary: Cube) -> "DistinctStep":
    """COUNT DISTINCT по измерению: ось различаемых значений должна быть осью куба."""
    axis = agg.terms[0].measures[0]
    cube = primary
    if axis not in cube.axes:
        for c in catalog.cubes.values():
            if axis in c.axes and not c.name.endswith("__count"):
                cube = c
                break
    if axis not in cube.axes:
        raise BindError(
            f"COUNT(DISTINCT {axis}): измерение не является осью ни одного куба запроса"
        )
    missing = [a for a in logical.group_axes if a not in cube.axes]
    if missing:
        raise BindError(
            f"COUNT(DISTINCT {axis}): измерения {missing} недостижимы из осей куба "
            f"'{cube.name}' {list(cube.axes)}; иерархии в этом агрегате не поддержаны"
        )
    for m in logical.filters:
        for a in m.axes:
            if a not in cube.axes:
                raise BindError(
                    f"COUNT(DISTINCT {axis}): фильтр по '{a}' неприменим — измерения "
                    f"нет в кубе '{cube.name}'"
                )
    return DistinctStep(cube.name, axis, list(logical.filters), logical.group_axes)


def _assemble(operands: list[Operand], logical: LogicalQuery,
              catalog: Catalog) -> EinsumStep:
    """Дополняет операнды матрицами иерархий и масками фильтров."""
    available = [a for o in operands for a in o.axes]
    needed = list(logical.group_axes) + [a for m in logical.filters for a in m.axes]
    rollups, available = _rollups_for(available, logical, catalog, needed)
    operands = operands + rollups

    for m in logical.filters:
        missing = [a for a in m.axes if a not in available]
        if missing:
            raise BindError(
                f"фильтр по {missing} неприменим: измерения нет ни в одном операнде "
                f"({sorted(set(available))})"
            )
        operands.append(Operand(ARRAY, f"mask:{'/'.join(m.axes)}", m.axes, m.data))

    missing = [a for a in logical.group_axes if a not in available]
    if missing:
        raise BindError(
            f"GROUP BY по {missing} невозможен: измерения нет ни в одном кубе запроса "
            f"и нет иерархии, ведущей к нему"
        )
    return EinsumStep(operands, logical.group_axes)


def _rollups_for(available: list[str], logical: LogicalQuery, catalog: Catalog,
                 needed: Iterable[str] | None = None) -> tuple[list[Operand], list[str]]:
    """Добавляет матрицы перехода по иерархиям для недостающих измерений."""
    available = list(available)
    rollups: list[Operand] = []
    wanted = list(needed if needed is not None else logical.group_axes)
    for axis in wanted:
        if axis in available:
            continue
        for child in list(available):
            m = catalog.rollup_matrix(child, axis)
            if m is not None:
                rollups.append(Operand(ARRAY, f"rollup:{child}->{axis}", (child, axis), m))
                available.append(axis)
                break
    return rollups, available
