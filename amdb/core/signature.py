# -*- coding: utf-8 -*-
"""Сигнатура двухосновной алгебраической системы AMDB.

Концепция алгебраической машины баз данных требует, чтобы данные и операции
над ними были заданы как алгебраическая система, а не подразумевались кодом.
Этот модуль объявляет систему явно:

    E = (S, T; Ω)

    S — основа данных: многомерные матрицы над полукольцом, с именованными
        осями и словарями измерений;
    T — основа операций: термы над сигнатурой Ω, то есть планы запросов;
    Ω — сигнатура: символы операций с указанием арности и сортов;
    интерпретация — отображение T × S* -> S, реализуемое исполнителем.

Каждая аксиома сопровождается ссылкой на тест, который её проверяет: система
задана декларативно, но её выполнение не постулируется, а верифицируется.

    >>> from amdb.core.signature import SYSTEM
    >>> print(SYSTEM.describe())
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sort:
    """Основа (сорт) алгебраической системы."""

    name: str
    description: str
    carrier: str          # что реализует носитель сорта


@dataclass(frozen=True)
class OperationSymbol:
    """Символ операции сигнатуры Ω."""

    name: str
    arity: str            # сигнатура вида "S × S -> S"
    description: str
    implementation: str
    in_algebra: bool = True   # принадлежит ли операция алгебре Соколова над (+,·)


@dataclass(frozen=True)
class Axiom:
    """Аксиома системы с указанием проверяющего её теста."""

    name: str
    statement: str
    test: str
    conditions: str = ""


@dataclass(frozen=True)
class AlgebraicSystem:
    """Двухосновная алгебраическая система E = (S, T; Ω)."""

    sorts: tuple[Sort, ...]
    operations: tuple[OperationSymbol, ...]
    axioms: tuple[Axiom, ...]
    outside: tuple[OperationSymbol, ...] = field(default=())

    def describe(self) -> str:
        lines = ["Двухосновная алгебраическая система AMDB: E = (S, T; Ω)", ""]
        lines.append("Основы:")
        for s in self.sorts:
            lines.append(f"  {s.name}: {s.description}")
            lines.append(f"      носитель: {s.carrier}")
        lines.append("")
        lines.append("Сигнатура Ω (операции алгебры):")
        for o in self.operations:
            lines.append(f"  {o.name:<22} {o.arity:<24} {o.description}")
            lines.append(f"  {'':22} реализация: {o.implementation}")
        if self.outside:
            lines.append("")
            lines.append("Вне сигнатуры (не выражаются через (+, ·)-свёртку):")
            for o in self.outside:
                lines.append(f"  {o.name:<22} {o.arity:<24} {o.description}")
        lines.append("")
        lines.append("Аксиомы (каждая проверяется автоматическим тестом):")
        for a in self.axioms:
            cond = f"  [{a.conditions}]" if a.conditions else ""
            lines.append(f"  {a.name}: {a.statement}{cond}")
            lines.append(f"      проверка: {a.test}")
        return "\n".join(lines)


DATA = Sort(
    "S (данные)",
    "многомерные матрицы над полукольцом с именованными осями",
    "amdb.core.MultidimensionalMatrix, amdb.core.COOCube",
)
OPERATIONS = Sort(
    "T (операции)",
    "термы над Ω — планы запросов, доступные до исполнения",
    "amdb.ql.PhysicalPlan, EinsumStep, BinaryProduct",
)

OMEGA = (
    OperationSymbol(
        "conv[λ,μ]", "S × S -> S",
        "(λ, μ)-свёрнутое произведение Соколова; λ — скоттовы индексы "
        "(сохраняются), μ — кэлиевы (свёртываются)",
        "amdb.core.convolve, convolve_named"),
    OperationSymbol(
        "add", "S × S -> S",
        "поэлементное сложение с выравниванием осей",
        "MultidimensionalMatrix.__add__"),
    OperationSymbol(
        "scale", "K × S -> S",
        "умножение на скаляр полукольца",
        "MultidimensionalMatrix.__mul__"),
    OperationSymbol(
        "transpose", "S × Perm -> S",
        "перестановка индексов",
        "MultidimensionalMatrix.transpose"),
    OperationSymbol(
        "project", "S × Axes -> S",
        "проекция: (0, μ)-свёртка с матрицей единиц",
        "MultidimensionalMatrix.project"),
    OperationSymbol(
        "slice", "S × Index -> S",
        "срез: фиксация значений индексов",
        "MultidimensionalMatrix.slice"),
    OperationSymbol(
        "intconv", "S × (i, j) -> S",
        "внутренняя свёртка по паре собственных индексов (обобщение следа)",
        "amdb.core.internal_convolution"),
    OperationSymbol(
        "unit", "Shape -> S",
        "единичная матрица произведения: E[s, c, m] = δ(c, m)",
        "amdb.core.unit_matrix"),
    OperationSymbol(
        "zero", "Shape -> S",
        "нулевая матрица — нейтральный элемент сложения",
        "numpy.zeros"),
)

OUTSIDE = (
    OperationSymbol(
        "reduce[min|max]", "S × Axes -> S",
        "редукция; алгебраически — conv[λ,μ] над полукольцом (max, ·) / (min, ·)",
        "amdb.core.reduce_axes, amdb.core.convolve_semiring", in_algebra=False),
    OperationSymbol(
        "order / limit / having", "Table -> Table",
        "операции над результирующей таблицей, а не над гиперкубами",
        "amdb.exec.Executor", in_algebra=False),
    OperationSymbol(
        "count_distinct", "S × Axis × Axes -> S",
        "композиция двух свёрток: (0,μ) над (∨,∧), затем (0,1) над (+,·); "
        "различаются значения измерения, не произвольной меры",
        "amdb.core.count_distinct", in_algebra=False),
)

AXIOMS = (
    Axiom("A1. Замкнутость S",
          "результат любой операции Ω снова принадлежит S",
          "test_algebraic_machine.py::test_data_sort_is_closed_under_signature"),
    Axiom("A2. Арифметика рангов",
          "rank(A conv[λ,μ] B) = rank A + rank B − λ − 2μ",
          "test_sokolov_algebra.py::test_rank_arithmetic"),
    Axiom("A3. Билинейность",
          "conv[λ,μ] дистрибутивна относительно add и однородна по scale",
          "test_sokolov_algebra.py::test_distributive_over_addition, "
          "test_homogeneous_in_both_arguments"),
    Axiom("A4. Ассоциативность",
          "(A conv B) conv C = A conv (B conv C)",
          "test_sokolov_algebra.py::test_associative_when_lambda_zero, "
          "test_associative_when_lambda_axis_shared_by_all",
          "при сохранении ролей индексов; λ-ось должна оставаться λ-осью"),
    Axiom("A5. Некоммутативность",
          "conv[λ,μ] в общем случае не коммутативна",
          "test_sokolov_algebra.py::test_not_commutative"),
    Axiom("A6. Единица",
          "A conv[λ,μ] unit = A",
          "test_sokolov_algebra.py::test_right_unit_for_mu_one, "
          "test_right_unit_for_mu_two"),
    Axiom("A7. Нуль",
          "A conv[λ,μ] zero = zero;  A add zero = A",
          "test_sokolov_algebra.py::test_convolution_with_zero_gives_zero, "
          "test_zero_matrix_is_additive_identity"),
    Axiom("A8. Транспонирование",
          "(A conv[λ,μ] B)^T = B^T conv[λ,μ] A^T",
          "test_sokolov_algebra.py::test_transposition_law"),
    Axiom("A9. Вырождение",
          "conv[0,0] — тензорное произведение; conv[0,1] на ранге 2 — умножение "
          "матриц; conv[0,μ] — tensordot",
          "test_sokolov_algebra.py::test_reduces_to_outer_product, "
          "test_reduces_to_matrix_multiplication, test_reduces_to_tensordot"),
    Axiom("A10. Терм и интерпретация",
          "план запроса есть терм над Ω; его интерпретация операциями ядра "
          "совпадает с исполнением движком",
          "test_algebraic_machine.py::test_interpretation_of_term_equals_engine_execution"),
    Axiom("A11. Корректность переписывания",
          "любой порядок применения операций терма даёт один результат — "
          "оптимизация семантику сохраняет",
          "test_algebraic_machine.py::test_every_contraction_order_gives_the_same_result"),
    Axiom("A12'. COUNT DISTINCT над булевым полукольцом",
          "COUNT(DISTINCT d) = подсчёт по оси d от ∨-свёртки индикатора",
          "test_queries.py::test_count_distinct_over_dimension, "
          "test_count_distinct_with_filter",
          "различаются значения измерения; для произвольной меры не определено"),
    Axiom("A12. Права доступа как операция",
          "ограничение доступа есть добавление операнда в терм и коммутирует "
          "с оптимизацией",
          "test_algebraic_machine.py::test_access_control_is_an_algebraic_rewriting, "
          "test_access_control_commutes_with_optimization"),
)

SYSTEM = AlgebraicSystem((DATA, OPERATIONS), OMEGA, AXIOMS, OUTSIDE)
