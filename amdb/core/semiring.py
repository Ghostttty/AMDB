# -*- coding: utf-8 -*-
"""(λ, μ)-свёрнутое произведение над произвольным полукольцом.

Алгебра Соколова строится над кольцом (R, +, ·): в определении свёрнутого
произведения сложение агрегирует, умножение комбинирует. Если заменить эту
пару операций другой парой, образующей полукольцо, определение сохраняет
форму, а операция меняет смысл:

    (+, ·)    — исходное произведение Соколова, агрегат SUM;
    (∨, ∧)    — булево полукольцо, основа агрегата COUNT DISTINCT;
    (max, ·)  — агрегат MAX;
    (min, ·)  — агрегат MIN;
    (max, +)  — тропическое полукольцо, «критический путь»;
    (min, +)  — тропическое полукольцо, кратчайшие пути.

Это даёт формально корректное место в алгебре тем агрегатам, которые через
обычное (+, ·)-произведение не выражаются: MIN и MAX оказываются не
исключением из модели, а тем же (λ, μ)-произведением над другим полукольцом.

Цена: einsum и BLAS работают только с (+, ·). Остальные полукольца считаются
широковещательным произведением с последующей редукцией, что требует
материализации промежуточного массива размера |L|·|S|·|C|·|M|. Поэтому
реализация проверяет бюджет памяти до аллокации.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from .convolve import build_einsum, convolve


class Semiring(NamedTuple):
    """Пара операций: агрегирующая (аналог +) и комбинирующая (аналог ·).

    ``nonnegative_only`` отмечает полукольца, законы которых выполняются лишь
    на неотрицательных значениях. Это не педантизм: для (max, ·) на
    знакопеременных данных нарушается дистрибутивность —

        max(2, 3) · (−1) = −3,  а  max(2·(−1), 3·(−1)) = −2,

    а вместе с ней и ассоциативность свёрнутого произведения. Значит,
    переписывание плана (смена порядка свёрток) перестаёт быть корректным —
    рушится главное свойство алгебраической машины.
    """

    name: str
    aggregate: Callable[..., np.ndarray]
    combine: Callable[[np.ndarray, np.ndarray], np.ndarray]
    identity: float
    nonnegative_only: bool = False

    def __repr__(self) -> str:
        return f"Semiring({self.name})"


SUM_PROD = Semiring("sum-prod", np.sum, np.multiply, 0.0)
#: Булево полукольцо ({0,1}, ∨, ∧). Даёт агрегату COUNT DISTINCT место в модели:
#: свёртка по ∨ схлопывает лишние оси, сохраняя лишь факт присутствия значения.
BOOL_OR_AND = Semiring("bool-or-and", np.max, np.minimum, 0.0)
MAX_PROD = Semiring("max-prod", np.max, np.multiply, -np.inf, nonnegative_only=True)
MIN_PROD = Semiring("min-prod", np.min, np.multiply, np.inf, nonnegative_only=True)
MAX_PLUS = Semiring("max-plus", np.max, np.add, -np.inf)
MIN_PLUS = Semiring("min-plus", np.min, np.add, np.inf)

SEMIRINGS = {s.name: s for s in (SUM_PROD, BOOL_OR_AND, MAX_PROD, MIN_PROD,
                                 MAX_PLUS, MIN_PLUS)}

#: Бюджет на промежуточный массив широковещательного произведения.
MAX_INTERMEDIATE_CELLS = 1 << 27      # ~1 ГиБ во float64


def convolve_semiring(
    A: np.ndarray,
    B: np.ndarray,
    lam: int,
    mu: int,
    semiring: Semiring | str = SUM_PROD,
    max_cells: int = MAX_INTERMEDIATE_CELLS,
    check_domain: bool = True,
) -> np.ndarray:
    """(λ, μ)-свёрнутое произведение над заданным полукольцом.

    Раскладка осей та же, что у :func:`amdb.core.convolve`:
    ``A: [L | S | C]``, ``B: [S | C | M]`` -> ``[L | S | M]``.

    Для полукольца (+, ·) вызов эквивалентен обычному ``convolve`` и
    перенаправляется на einsum: незачем терять BLAS там, где он применим.
    """
    if isinstance(semiring, str):
        try:
            semiring = SEMIRINGS[semiring]
        except KeyError:
            raise ValueError(
                f"неизвестное полукольцо '{semiring}'; доступны: {sorted(SEMIRINGS)}"
            ) from None
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    build_einsum(A.ndim, B.ndim, lam, mu)          # проверка согласованности рангов
    if check_domain and semiring.nonnegative_only:
        check_domain_of(semiring, A, B)
    if semiring is SUM_PROD:
        return convolve(A, B, lam, mu)

    k = A.ndim - lam - mu
    v = B.ndim - lam - mu
    if tuple(A.shape[k:]) != tuple(B.shape[: lam + mu]):
        raise ValueError(
            f"несогласованные длины общих осей: A{tuple(A.shape[k:])} и "
            f"B{tuple(B.shape[: lam + mu])}"
        )

    shape_l, shape_sc = A.shape[:k], A.shape[k:]
    shape_m = B.shape[lam + mu:]
    cells = int(np.prod(shape_l + shape_sc + shape_m)) if (shape_l + shape_sc + shape_m) else 1
    if cells > max_cells:
        raise MemoryError(
            f"полукольцо {semiring.name} требует материализации "
            f"{cells:,} ячеек (~{cells * 8 / 2**30:.1f} ГиБ): в отличие от (+, ·) "
            "оно не сводится к BLAS. Уменьшите размерность или считайте по частям."
        )

    # A -> [L, S, C, 1...1(M)],  B -> [1...1(L), S, C, M]
    A_exp = A.reshape(A.shape + (1,) * v)
    B_exp = B.reshape((1,) * k + B.shape)
    combined = semiring.combine(A_exp, B_exp)
    if mu:
        axes = tuple(range(k + lam, k + lam + mu))
        combined = semiring.aggregate(combined, axis=axes)
    return combined


def count_distinct(
    cube: np.ndarray,
    distinct_axis: int,
    keep_axes: tuple[int, ...],
) -> np.ndarray:
    """COUNT DISTINCT по измерению как композиция двух свёрнутых произведений.

    Вопреки распространённому мнению, этот агрегат выражается в модели —
    но не над числовым полукольцом, а над булевым. Схема из двух шагов:

    1. (0, μ)-свёртка индикатора над полукольцом (∨, ∧) по осям, которые не
       входят ни в группировку, ни в ось различаемых значений. Она схлопывает
       кратность, сохраняя лишь факт присутствия значения;
    2. (0, 1)-свёртка полученного индикатора с матрицей единиц над обычным
       (+, ·) по оси различаемых значений — то есть подсчёт.

    Ограничение существенно: различаются значения **измерения**, то есть
    координаты по оси. Для DISTINCT по произвольной мере потребовалось бы
    вынести область её значений в отдельную ось.
    """
    cube = np.asarray(cube)
    keep = tuple(keep_axes)
    if distinct_axis in keep:
        raise ValueError("ось различаемых значений не может входить в группировку")
    present = cube != 0                      # булев индикатор: 1 байт на ячейку

    collapse = tuple(i for i in range(cube.ndim)
                     if i != distinct_axis and i not in keep)
    if collapse:
        # Шаг 1: ∨-свёртка по «лишним» осям (полукольцо (∨, ∧)).
        #
        # Свёртка с матрицей единиц над (∨, ∧) поэлементно совпадает с прямой
        # ∨-редукцией по тем же осям, но общий путь полукольца материализует
        # произведение операндов целиком — лишняя копия размером с куб.
        # Здесь берётся редукция: результат тот же, промежуточного массива нет,
        # и индикатор до самой редукции остаётся булевым.
        present = np.max(present, axis=collapse)
        remaining = [i for i in range(cube.ndim) if i not in collapse]
        distinct_pos = remaining.index(distinct_axis)
        keep_pos = [remaining.index(k) for k in keep]
    else:
        distinct_pos = distinct_axis
        keep_pos = [k for k in keep]

    # Шаг 2: подсчёт — (0,1)-свёртка с матрицей единиц над (+, ·). einsum
    # сворачивает без промежуточного массива; в float64 переводится уже
    # свёрнутый индикатор, а не исходный куб.
    order = keep_pos + [distinct_pos]
    present = np.transpose(present, order).astype(np.float64)
    return convolve_semiring(present, np.ones(present.shape[-1]), lam=0, mu=1,
                             semiring=SUM_PROD)


def check_domain_of(semiring: Semiring, *operands: np.ndarray) -> None:
    """Проверяет, что данные лежат в области, где полукольцо действительно полукольцо.

    Вычислить свёртку можно и на знакопеременных данных — она определена
    поэлементно. Но алгебраические законы (дистрибутивность, а с ней и
    ассоциативность) там не выполняются, и оптимизатор, меняющий порядок
    свёрток, начнёт менять ответ. Поэтому проверка включена по умолчанию.
    """
    for i, x in enumerate(operands):
        arr = np.asarray(x)
        if arr.size and np.nanmin(arr) < 0:
            raise ValueError(
                f"полукольцо {semiring.name} образует полукольцо только на "
                f"неотрицательных значениях, а операнд {i} содержит отрицательные "
                f"(минимум {float(np.nanmin(arr)):g}). На знакопеременных данных "
                "нарушается дистрибутивность, и порядок свёрток начинает влиять "
                "на результат. Используйте тропическое полукольцо (min, +) / "
                "(max, +) либо передайте check_domain=False, приняв, что "
                "алгебраические гарантии не действуют."
            )


def aggregate_over_axes(
    A: np.ndarray, axes: tuple[int, ...], semiring: Semiring | str = MAX_PROD,
    check_domain: bool = True,
) -> np.ndarray:
    """Агрегация по осям как (0, μ)-произведение с единицей полукольца.

    Показывает, что «редукция MAX по осям» — частный случай свёрнутого
    произведения над (max, ·): вторым операндом выступает матрица из единиц.
    """
    if isinstance(semiring, str):
        semiring = SEMIRINGS[semiring]
    moved = np.moveaxis(A, axes, range(A.ndim - len(axes), A.ndim))
    ones = np.ones(tuple(moved.shape[A.ndim - len(axes):]), dtype=np.float64)
    return convolve_semiring(moved, ones, lam=0, mu=len(axes), semiring=semiring,
                             check_domain=check_domain)
