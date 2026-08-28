# -*- coding: utf-8 -*-
"""Проверка эталонной реализации (lambda, mu)-свёрнутого произведения."""
from __future__ import annotations
import string, time
import numpy as np

_ALPHABET = string.ascii_lowercase + string.ascii_uppercase  # 52 индекса

class IndexAllocator:
    def __init__(self):
        self._n = 0
    def new(self) -> str:
        if self._n >= len(_ALPHABET):
            raise ValueError("превышен лимит 52 различных индексов einsum")
        ch = _ALPHABET[self._n]; self._n += 1
        return ch

def build_einsum(rank_a: int, rank_b: int, lam: int, mu: int) -> str:
    """Строит einsum-строку для (lam, mu)-свёрнутого произведения.

    Раскладка Соколова:
      A: [l_1..l_k | s_1..s_lam | c_1..c_mu],  k = rank_a - lam - mu
      B: [s_1..s_lam | c_1..c_mu | m_1..m_v],  v = rank_b - lam - mu
      C: [l_1..l_k | s_1..s_lam | m_1..m_v]
    """
    k = rank_a - lam - mu
    v = rank_b - lam - mu
    if k < 0 or v < 0:
        raise ValueError(f"недопустимые ранги: k={k}, v={v}")
    a = IndexAllocator()
    L = [a.new() for _ in range(k)]
    S = [a.new() for _ in range(lam)]
    C = [a.new() for _ in range(mu)]
    M = [a.new() for _ in range(v)]
    lhs_a = "".join(L + S + C)
    lhs_b = "".join(S + C + M)
    out   = "".join(L + S + M)
    return f"{lhs_a},{lhs_b}->{out}"

def convolve(A: np.ndarray, B: np.ndarray, lam: int, mu: int, optimize=True) -> np.ndarray:
    spec = build_einsum(A.ndim, B.ndim, lam, mu)
    return np.einsum(spec, A, B, optimize=optimize)

def naive_convolve_2_3(A, B):
    """Прямая проверка примера из статьи: A ранга 6, B ранга 9, (2,3)-свёртка."""
    k, s1, s2, c1, c2, c3 = A.shape
    _, _, _, _, _, m1, m2, m3, m4 = B.shape
    Cm = np.zeros((k, s1, s2, m1, m2, m3, m4))
    for i in range(k):
        for x in range(s1):
            for y in range(s2):
                Cm[i, x, y] = np.tensordot(A[i, x, y], B[x, y], axes=([0,1,2],[0,1,2]))
    return Cm

if __name__ == "__main__":
    print("einsum (2,3), rank6 x rank9 :", build_einsum(6, 9, 2, 3))
    print("einsum (0,1), rank2 x rank2 :", build_einsum(2, 2, 0, 1))
    print("einsum (1,1), rank3 x rank3 :", build_einsum(3, 3, 1, 1))

    rng = np.random.default_rng(42)
    A = rng.random((2, 3, 2, 3, 2, 2))
    B = rng.random((3, 2, 3, 2, 2, 2, 2, 2, 2))
    C1 = convolve(A, B, lam=2, mu=3)
    C2 = naive_convolve_2_3(A, B)
    print("shape", C1.shape, "== ", C2.shape, " max err:", np.abs(C1 - C2).max())

    # Бенчмарк (1,1)-свёртки 100x100x100
    A = rng.random((100, 100, 100)); B = rng.random((100, 100, 100))
    t0 = time.perf_counter(); C = convolve(A, B, lam=1, mu=1); t1 = time.perf_counter()
    print("3D 100^3 (1,1)-свёртка:", round((t1-t0)*1e3, 2), "мс, shape", C.shape)
