# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, time

rng = np.random.default_rng(1)
N = 500_000
df = pd.DataFrame({
    "customer": rng.integers(0, 100, N),
    "product":  rng.integers(0, 100, N),
    "date":     rng.integers(0, 100, N),
    "qty":      rng.random(N).astype(np.float32),
})

class Dimension:
    """Словарь измерения: значение <-> порядковый индекс оси."""
    def __init__(self, name, values):
        self.name = name
        self.values = np.asarray(values)
        self._pos = {v: i for i, v in enumerate(self.values.tolist())}
    def __len__(self): return len(self.values)
    def encode(self, series): return series.map(self._pos).to_numpy(np.int64)

def load_fact(df, dim_cols, measure, agg="sum"):
    dims = [Dimension(c, np.sort(df[c].unique())) for c in dim_cols]
    shape = tuple(len(d) for d in dims)
    idx = np.ravel_multi_index([d.encode(df[d.name]) for d in dims], shape)
    flat = np.bincount(idx, weights=df[measure].to_numpy(np.float64),
                       minlength=int(np.prod(shape)))
    return flat.reshape(shape).astype(np.float32), dims

t0 = time.perf_counter(); cube, dims = load_fact(df, ["customer","product","date"], "qty"); t1 = time.perf_counter()
nnz = np.count_nonzero(cube)
print(f"загрузка {N} строк -> гиперкуб {cube.shape}: {round((t1-t0)*1e3,1)} мс")
print(f"заполненность: {nnz/cube.size:.3%}, память плотно: {cube.nbytes/2**20:.2f} МиБ")
print("контроль суммы:", np.isclose(cube.sum(), df.qty.sum(), rtol=1e-4))
