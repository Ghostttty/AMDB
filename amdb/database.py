# -*- coding: utf-8 -*-
"""Высокоуровневый фасад AMDB: загрузка данных и выполнение запросов."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .exec.engine import Engine, NumpyEngine, warn_if_reference_blas
from .exec.executor import Executor
from .exec.result import ResultSet
from .ql.binder import bind
from .ql.optimizer import PlanCache, optimize_plan
from .ql.parser import parse
from .ql.planner import (
    PhysicalPlan,
    binary_decomposition,
    plan_query,
    term_expression,
)
from .security.rls import apply_rls
from .storage.catalog import Catalog, Cube, Grant, Role
from .storage.dimension import Dimension, Hierarchy
from .storage.loader import add_hierarchy, load_dimension_table, load_fact, read_csv, read_sql


class Database:
    """Алгебраическая машина баз данных: гиперкубы + SQL-подобный язык."""

    def __init__(self, catalog: Catalog | None = None, engine: Engine | None = None):
        self.catalog = catalog or Catalog()
        self.engine = engine or NumpyEngine()
        self.plan_cache = PlanCache()
        self._executor = Executor(self.catalog, self.engine)

    # -- жизненный цикл -----------------------------------------------------
    @classmethod
    def open(cls, path: str | Path, engine: Engine | None = None) -> "Database":
        return cls(Catalog.load(path), engine)

    def save(self, path: str | Path) -> Path:
        return self.catalog.save(path)

    def use_engine(self, engine: Engine) -> "Database":
        self.engine = engine
        self._executor = Executor(self.catalog, engine)
        return self

    # -- загрузка данных ----------------------------------------------------
    def load_frame(self, frame: Any, dimensions: Sequence[str], measure: str,
                   name: str | None = None, **kwargs: Any) -> Cube:
        cube = load_fact(self.catalog, frame, dimensions, measure, name, **kwargs)
        self.plan_cache.clear()
        return cube

    def load_csv(self, path: str | Path, dimensions: Sequence[str], measure: str,
                 name: str | None = None, read_kwargs: dict[str, Any] | None = None,
                 **kwargs: Any) -> Cube:
        frame = read_csv(path, **(read_kwargs or {}))
        return self.load_frame(frame, dimensions, measure, name, **kwargs)

    def load_sql(self, connection: Any, query: str, dimensions: Sequence[str],
                 measure: str, name: str | None = None, **kwargs: Any) -> Cube:
        return self.load_frame(read_sql(query, connection), dimensions, measure,
                               name, **kwargs)

    def load_dimension(self, frame: Any, key: str, name: str | None = None,
                       attributes: Sequence[str] = (), measures: Sequence[str] = (),
                       **kwargs: Any) -> Dimension:
        dim = load_dimension_table(self.catalog, frame, key, name, attributes,
                                   measures, **kwargs)
        self.plan_cache.clear()
        return dim

    def add_hierarchy(self, child: str, parent: str,
                      mapping: Sequence[Any] | dict[Any, Any], **kwargs: Any) -> Hierarchy:
        h = add_hierarchy(self.catalog, child, parent, mapping, **kwargs)
        self.plan_cache.clear()
        return h

    def add_cube(self, cube: Cube) -> Cube:
        self.plan_cache.clear()
        return self.catalog.add_cube(cube)

    def grant(self, role_name: str, axis: str, allowed: Iterable[Any] | None = None,
              can_project: bool = True, permissive: bool = True) -> Role:
        """Выдаёт роли право на измерение, опционально ограничивая значения."""
        role = self.catalog.roles.get(role_name) or self.catalog.add_role(
            Role(role_name, {}, permissive))
        mask = None
        if allowed is not None:
            mask = self.catalog.dimension(axis).mask_of(list(allowed))
        role.grants[axis] = Grant(mask, can_project)
        self.plan_cache.clear()
        return role

    # -- запросы ------------------------------------------------------------
    def compile(self, sql: str, role: str | None = None,
                use_cache: bool = True) -> PhysicalPlan:
        """Транслирует запрос в план матричных операций."""
        key = PlanCache.key(sql, self.catalog, role)
        if use_cache:
            cached = self.plan_cache.get(key)
            if cached is not None:
                return cached
        ast = parse(sql)
        logical = bind(ast, self.catalog)
        plan = plan_query(logical, self.catalog)
        if role is not None:
            apply_rls(plan, self.catalog, self.catalog.role(role))
        optimize_plan(plan, self.catalog)
        return self.plan_cache.put(key, plan) if use_cache else plan

    def sql(self, query: str, role: str | None = None) -> ResultSet:
        """Выполняет запрос и возвращает таблицу результата."""
        t0 = time.perf_counter()
        plan = self.compile(query, role)
        t_compile = time.perf_counter() - t0
        result = self._executor.run(plan)
        result.stats["compile_seconds"] = t_compile
        result.stats["plan_cache"] = {"hits": self.plan_cache.hits,
                                      "misses": self.plan_cache.misses}
        return result

    def explain(self, query: str, role: str | None = None) -> str:
        """Показывает план: цепочку einsum-свёрток с ролями индексов."""
        plan = self.compile(query, role, use_cache=False)
        costs = optimize_plan(plan, self.catalog)
        lines = [f"Запрос: {' '.join(query.split())}", "",
                 plan.query.describe(), "", plan.explain(), "", "Оценки:"]
        for name, cost in costs.items():
            lines.append(f"  {name}: {cost!r}")
        return "\n".join(lines)

    def sokolov(self, query: str, role: str | None = None) -> str:
        """Разложение запроса в цепочку бинарных (λ, μ)-произведений Соколова.

        Показывает, что план — не «просто einsum», а композиция операций
        алгебры многомерных матриц, с явными ролями индексов на каждом шаге.
        """
        plan = self.compile(query, role, use_cache=False)
        lines = [f"Запрос: {' '.join(query.split())}", ""]
        for agg in plan.aggregates:
            lines.append(f"{agg.alias}: {agg.func}")
            if agg.reduce is not None:
                lines.append("    вне (+,·): редукция "
                             f"{agg.reduce.how.upper()} по осям "
                             f"{[a for a in self.cube(agg.reduce.cube).axes if a not in agg.reduce.output]}"
                             "   [выразимо над (max,·) при неотрицательных мерах]")
            if agg.distinct is not None:
                d = agg.distinct
                collapse = [a for a in self.cube(d.cube).axes
                            if a != d.distinct_axis and a not in d.output]
                lines.append(f"    1. (0,{len(collapse)})-свёртка индикатора над (∨,∧) "
                             f"по осям {collapse}")
                lines.append(f"    2. (0,1)-свёртка с 𝟙 над (+,·) по оси "
                             f"'{d.distinct_axis}' — подсчёт")
            for coef, step in agg.terms:
                lines.append(f"    слагаемое с коэффициентом {coef:+g}:")
                for i, bp in enumerate(binary_decomposition(step), 1):
                    lines.append(f"      {i}. {bp.describe()}")
            if agg.denominator is not None:
                lines.append("    знаменатель (COUNT):")
                for i, bp in enumerate(binary_decomposition(agg.denominator), 1):
                    lines.append(f"      {i}. {bp.describe()}")
        return "\n".join(lines)

    def term(self, query: str, role: str | None = None) -> str:
        """Запрос как терм алгебры: вложенные (λ, μ)-произведения.

        Терм — элемент второй основы двухосновной алгебраической системы:
        данные (гиперкубы) и операции над ними разнесены по сортам, а
        оптимизация есть переписывание терма с сохранением интерпретации.
        """
        plan = self.compile(query, role, use_cache=False)
        lines = []
        for agg in plan.aggregates:
            if agg.reduce is not None:
                axes = [a for a in self.cube(agg.reduce.cube).axes
                        if a not in agg.reduce.output]
                lines.append(
                    f"{agg.alias} = {agg.reduce.how}_{{{', '.join(axes)}}}"
                    f"({agg.reduce.cube})   [редукция вне (+,·); см. полукольца]")
                continue
            if agg.distinct is not None:
                d = agg.distinct
                collapse = [a for a in self.cube(d.cube).axes
                            if a != d.distinct_axis and a not in d.output]
                lines.append(
                    f"{agg.alias} = (∨_{{{', '.join(collapse)}}}[{d.cube} ≠ 0]) "
                    f"∗[0,1] 𝟙[{d.distinct_axis}]   [(∨,∧), затем (+,·)]")
                continue
            parts = []
            for coef, step in agg.terms:
                body = term_expression(step)
                parts.append(body if coef == 1 else f"{coef:g}·{body}")
            body = " + ".join(parts)
            if agg.denominator is not None:
                numerator = body if len(parts) == 1 else f"({body})"
                body = f"{numerator} / {term_expression(agg.denominator)}"
            lines.append(f"{agg.alias} = {body}")
        return "\n".join(lines)

    def einsum_of(self, query: str) -> list[str]:
        """einsum-строки, в которые транслируется запрос."""
        plan = self.compile(query, use_cache=False)
        specs = []
        for agg in plan.aggregates:
            specs.extend(step.spec for _, step in agg.terms)
            if agg.denominator is not None:
                specs.append(agg.denominator.spec)
        return specs

    # -- интроспекция -------------------------------------------------------
    @property
    def cubes(self) -> dict[str, Cube]:
        return self.catalog.cubes

    @property
    def dimensions(self) -> dict[str, Dimension]:
        return self.catalog.dimensions

    def cube(self, name: str) -> Cube:
        return self.catalog.cube(name)

    def stats(self, cube: str) -> dict[str, Any]:
        return self.catalog.stats(cube)

    def summary(self) -> str:
        warn = warn_if_reference_blas()
        head = f"AMDB, движок {self.engine.name}"
        tail = f"\nВНИМАНИЕ: {warn}" if warn else ""
        return f"{head}\n{self.catalog.summary()}{tail}"

    def __repr__(self) -> str:
        return (f"Database(кубов={len(self.catalog.cubes)}, "
                f"измерений={len(self.catalog.dimensions)}, движок={self.engine.name})")
