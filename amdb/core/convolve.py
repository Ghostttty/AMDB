# -*- coding: utf-8 -*-
"""(λ, μ)-свёрнутое произведение многомерных матриц через соглашение Эйнштейна.

Реализует определение Н.П. Соколова:

    c[l_1..l_k, s_1..s_λ, m_1..m_v]
        = Σ по (c_1..c_μ)  a[l_1..l_k, s_1..s_λ, c_1..c_μ]
                         · b[s_1..s_λ, c_1..c_μ, m_1..m_v]

Общие для обоих сомножителей индексы носят устоявшиеся имена (Мунерман В.И.,
Мунерман Д.В., 2022):

    s_1..s_λ — **скоттовы** индексы: общие и сохраняемые в результате;
    c_1..c_μ — **кэлиевы** индексы: общие и свёртываемые.

Именно наличие скоттовых индексов отличает эту операцию от умножения обычных
матриц и от тензорной свёртки: при λ = 0 она вырождается в них.

Ключевая идея (см. docs/): операция естественно выражается одним вызовом
``numpy.einsum``, поэтому вся тяжёлая арифметика уходит в BLAS.
"""
from __future__ import annotations

import string
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np

#: Алфавит индексов einsum. NumPy принимает a-z и A-Z — ровно 52 символа.
#: Списочная форма einsum лимит не снимает: subscript must be in range [0, 52).
ALPHABET: str = string.ascii_lowercase + string.ascii_uppercase

#: Максимальное число различных индексов в одной элементарной операции einsum.
MAX_SUBSCRIPTS: int = len(ALPHABET)

#: Ниже этого суммарного числа ячеек поиск порядка свёрток стоит дороже самой
#: свёртки. Порог намеренно низкий: выше него einsum уходит в BLAS, и отказ от
#: оптимизации обошёлся бы дороже (проверено bench/bench_convolve.py).
SMALL_INPUT_CELLS: int = 2_000

#: С этого размера свёртка при λ >= 1 идёт пакетным matmul вместо einsum.
#: Ниже порога перестановки осей стоят дороже выигрыша от BLAS.
BATCH_MATMUL_CELLS: int = 4_096


class TooManyIndicesError(ValueError):
    """Плану требуется больше 52 различных индексов — нужна декомпозиция."""


def symbols_for(names: Sequence[str]) -> dict[str, str]:
    """Сопоставляет именам измерений буквы einsum в порядке первого появления."""
    uniq: list[str] = []
    for n in names:
        if n not in uniq:
            uniq.append(n)
    if len(uniq) > MAX_SUBSCRIPTS:
        raise TooManyIndicesError(
            f"{len(uniq)} различных индексов > {MAX_SUBSCRIPTS}; "
            "план должен быть разбит на цепочку парных свёрток"
        )
    return {name: ch for name, ch in zip(uniq, ALPHABET)}


def build_spec(subscripts: Iterable[Sequence[str]], output: Sequence[str]) -> str:
    """Строит einsum-строку по именованным осям операндов и результата.

    >>> build_spec([("c", "p", "d"), ("p",), ("d", "m")], ("c", "m"))
    'abc,b,cd->ad'
    """
    subs = [tuple(s) for s in subscripts]
    sym = symbols_for([a for s in subs for a in s] + list(output))
    lhs = ",".join("".join(sym[a] for a in s) for s in subs)
    return f"{lhs}->{''.join(sym[a] for a in output)}"


@lru_cache(maxsize=512)
def build_einsum(rank_a: int, rank_b: int, lam: int, mu: int) -> str:
    """einsum-строка для (lam, mu)-свёрнутого произведения матриц данных рангов.

    Каноническая раскладка осей (Соколов; та же, что у Мунермана, 2022):
        A: [l_1..l_k | s_1..s_lam | c_1..c_mu],   k = rank_a - lam - mu
        B: [s_1..s_lam | c_1..c_mu | m_1..m_v],   v = rank_b - lam - mu
        C: [l_1..l_k | s_1..s_lam | m_1..m_v]
    где s — скоттовы (сохраняемые) индексы, c — кэлиевы (свёртываемые).

    >>> build_einsum(6, 9, lam=2, mu=3)
    'abcdef,bcdefghij->abcghij'
    >>> build_einsum(2, 2, lam=0, mu=1)
    'ab,bc->ac'
    """
    if lam < 0 or mu < 0:
        raise ValueError("lam и mu не могут быть отрицательными")
    k, v = rank_a - lam - mu, rank_b - lam - mu
    if k < 0:
        raise ValueError(f"ранг A ({rank_a}) меньше lam + mu ({lam + mu})")
    if v < 0:
        raise ValueError(f"ранг B ({rank_b}) меньше lam + mu ({lam + mu})")
    total = k + lam + mu + v
    if total > MAX_SUBSCRIPTS:
        raise TooManyIndicesError(
            f"{total} индексов > {MAX_SUBSCRIPTS}; требуется декомпозиция плана"
        )
    it = iter(ALPHABET)
    L = [next(it) for _ in range(k)]
    S = [next(it) for _ in range(lam)]
    C = [next(it) for _ in range(mu)]
    M = [next(it) for _ in range(v)]
    return f"{''.join(L + S + C)},{''.join(S + C + M)}->{''.join(L + S + M)}"


def convolve(
    A: np.ndarray,
    B: np.ndarray,
    lam: int,
    mu: int,
    optimize: bool | str | list = True,
) -> np.ndarray:
    """(lam, mu)-свёрнутое произведение двух многомерных матриц.

    Оси должны быть предварительно приведены к канонической раскладке
    (см. :func:`build_einsum`); для работы с именованными осями используйте
    :func:`amdb.core.mdm.convolve_named`.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    spec = build_einsum(A.ndim, B.ndim, lam, mu)
    shared = A.shape[A.ndim - lam - mu :]
    expected = B.shape[: lam + mu]
    if tuple(shared) != tuple(expected):
        raise ValueError(
            f"несогласованные длины общих осей: A{tuple(shared)} и B{tuple(expected)}"
        )
    if lam and mu and A.size + B.size >= BATCH_MATMUL_CELLS:
        out = _batched_matmul(A, B, lam, mu)
        if out is not None:
            return out
    if optimize is True:
        # optimize=True пересчитывает порядок свёрток при каждом вызове — на
        # операндах в сотни килобайт это стоит дороже самой свёртки. Для малых
        # операндов оптимизация не нужна вовсе, для остальных путь кэшируется.
        optimize = (False if A.size + B.size < SMALL_INPUT_CELLS
                    else cached_path(spec, A.shape, B.shape))
    return np.einsum(spec, A, B, optimize=optimize)


def _batched_matmul(A: np.ndarray, B: np.ndarray, lam: int, mu: int):
    """Свёрнутое произведение при λ >= 1 как пакетный BLAS-gemm.

    ``numpy.einsum`` не умеет отправлять в BLAS свёртку, у которой есть индекс,
    общий обоим сомножителям и сохраняемый в результате: ``einsum_path`` сводит
    операции к ``tensordot``, а у того пакетного измерения нет. Поэтому при
    λ >= 1 einsum уходит в собственный цикл на C и теряет BLAS — на замерах
    в 8-27 раз (bench/bench_convolve.py).

    Обход подсказан самой алгеброй. По утверждению Мунермана и Мунермана (2022),
    при λ >= 1 результат составлен из сечений ориентации (s_1..s_λ), каждое из
    которых есть (0, μ)-произведение соответствующих сечений сомножителей.
    А набор одинаковых (0, μ)-произведений над сечениями — это ровно то, что
    ``numpy.matmul`` исполняет пакетным gemm:

        A[L | S | C] -> (|S|, |L|, |C|),   B[S | C | M] -> (|S|, |C|, |M|)
        matmul -> (|S|, |L|, |M|)      ->  C[L | S | M]

    Возвращает None, если свести к трёхмерному виду не удалось (несмежные оси
    после перестановки), — тогда вызывающий код остаётся на einsum.
    """
    k = A.ndim - lam - mu
    v = B.ndim - lam - mu
    shape_l, shape_s = A.shape[:k], A.shape[k:k + lam]
    shape_c, shape_m = A.shape[k + lam:], B.shape[lam + mu:]
    n_l = int(np.prod(shape_l)) if shape_l else 1
    n_s = int(np.prod(shape_s)) if shape_s else 1
    n_c = int(np.prod(shape_c)) if shape_c else 1
    n_m = int(np.prod(shape_m)) if shape_m else 1
    if not (n_s and n_c and n_l and n_m):
        return None
    try:
        a3 = np.transpose(A, tuple(range(k, k + lam)) + tuple(range(k))
                          + tuple(range(k + lam, A.ndim))).reshape(n_s, n_l, n_c)
        b3 = np.asarray(B).reshape(n_s, n_c, n_m)
    except ValueError:                       # pragma: no cover — форма не сводится
        return None
    out = np.matmul(a3, b3).reshape(shape_s + shape_l + shape_m)
    return np.transpose(out, tuple(range(lam, lam + k)) + tuple(range(lam))
                        + tuple(range(lam + k, out.ndim)))


@lru_cache(maxsize=1024)
def cached_path(spec: str, shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> list:
    """Порядок свёрток для пары форм. Считается один раз на форму, не на вызов."""
    zero = np.zeros((), dtype=np.float32)
    stubs = (np.broadcast_to(zero, shape_a), np.broadcast_to(zero, shape_b))
    return np.einsum_path(spec, *stubs, optimize="optimal")[0]


def unit_matrix(
    mu_shape: Sequence[int],
    lam_shape: Sequence[int] = (),
    dtype: Any = np.float64,
) -> np.ndarray:
    """Единичный элемент (λ, μ)-свёрнутого произведения.

    Возвращает матрицу ``E`` формы ``lam_shape + mu_shape + mu_shape`` такую,
    что ``convolve(A, E, lam, mu) == A`` для любой согласованной A:

        E[s_1..s_λ, c_1..c_μ, m_1..m_μ] = δ(c_1, m_1) · … · δ(c_μ, m_μ)

    По λ-индексам единица не зависит — они лишь «переносятся» операцией.
    При μ = 1 и λ = 0 это обычная единичная матрица.
    """
    mu_shape = tuple(int(n) for n in mu_shape)
    lam_shape = tuple(int(n) for n in lam_shape)
    if not mu_shape:
        raise ValueError("единичный элемент определён при μ >= 1")
    eye = np.eye(mu_shape[0], dtype=dtype)
    for n in mu_shape[1:]:
        eye = np.multiply.outer(eye, np.eye(n, dtype=dtype))
    # Сейчас порядок осей — (c1,m1,c2,m2,…); приводим к (c1..cμ, m1..mμ).
    order = list(range(0, 2 * len(mu_shape), 2)) + list(range(1, 2 * len(mu_shape), 2))
    eye = np.transpose(eye, order)
    if not lam_shape:
        return np.ascontiguousarray(eye)
    return np.ascontiguousarray(np.broadcast_to(eye, lam_shape + eye.shape))


def convolve_naive(A: np.ndarray, B: np.ndarray, lam: int, mu: int) -> np.ndarray:
    """Прямая реализация по определению — эталон для тестов, не для продакшена."""
    A = np.asarray(A)
    B = np.asarray(B)
    k = A.ndim - lam - mu
    v = B.ndim - lam - mu
    Ls, Ss = A.shape[:k], A.shape[k : k + lam]
    Cs, Ms = A.shape[k + lam :], B.shape[lam + mu :]
    out = np.zeros(Ls + Ss + Ms, dtype=np.result_type(A.dtype, B.dtype))
    for li in np.ndindex(Ls):
        for si in np.ndindex(Ss):
            acc = np.zeros(Ms, dtype=out.dtype)
            for ci in np.ndindex(Cs):
                acc += A[li + si + ci] * B[si + ci]
            out[li + si] = acc
    return out


def batched_from_spec(spec: str, A: np.ndarray, B: np.ndarray,
                      min_cells: int = BATCH_MATMUL_CELLS):
    """Сводит парную einsum-спецификацию к пакетному gemm, если это возможно.

    Ядро умеет исполнять (λ, μ)-произведение пакетным матричным умножением
    (см. :func:`_batched_matmul`), но требует канонической раскладки осей.
    Исполнитель запросов оперирует спецификациями с произвольным порядком
    индексов, и без этой функции быстрый путь до него не доходил.

    Роли индексов восстанавливаются из самой спецификации: общие и остающиеся
    в результате — скоттовы, общие и исчезающие — кэлиевы, прочие свободны.
    Возвращает None, если спецификация к произведению не сводится: есть
    повторы индекса внутри операнда (диагональ), приватная свёртываемая ось
    или индекс результата, отсутствующий в обоих операндах.
    """
    if "->" not in spec:
        return None
    lhs, out = spec.split("->")
    parts = lhs.split(",")
    if len(parts) != 2:
        return None
    sa, sb = parts
    if len(set(sa)) != len(sa) or len(set(sb)) != len(sb) or len(set(out)) != len(out):
        return None            # диагональ или повтор в результате
    a_idx, b_idx, o_idx = set(sa), set(sb), set(out)
    shared = a_idx & b_idx
    S = [x for x in out if x in shared]
    C = [x for x in sa if x in shared and x not in o_idx]
    L = [x for x in out if x in a_idx and x not in b_idx]
    M = [x for x in out if x in b_idx and x not in a_idx]
    lam, mu = len(S), len(C)
    if lam < 1 or mu < 1:
        return None            # быстрый путь определён лишь при λ >= 1, μ >= 1
    if set(L + S + C) != a_idx or set(S + C + M) != b_idx or set(L + S + M) != o_idx:
        return None            # приватная свёртываемая ось либо лишний индекс
    if A.size + B.size < min_cells:
        return None            # перестановки осей дороже выигрыша

    a_perm = [sa.index(x) for x in L + S + C]
    b_perm = [sb.index(x) for x in S + C + M]
    result = _batched_matmul(np.transpose(A, a_perm), np.transpose(B, b_perm), lam, mu)
    if result is None:
        return None
    produced = L + S + M
    if produced == list(out):
        return result
    return np.transpose(result, [produced.index(x) for x in out])
