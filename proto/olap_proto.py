# -*- coding: utf-8 -*-
import time
import numpy as np

rng = np.random.default_rng(7)
NC, NP, ND = 100, 100, 100          # Customer, Product, Date
NM = 12                             # Month

sales_qty = rng.random((NC, NP, ND)).astype(np.float32)   # факт
price     = rng.random(NP).astype(np.float32)             # измерение-справочник
date2month = np.zeros((ND, NM), dtype=np.float32)         # матрица иерархии
date2month[np.arange(ND), rng.integers(0, NM, ND)] = 1.0

# SELECT Customer, Month, SUM(qty*price) ... GROUP BY Customer, Month
t0 = time.perf_counter()
res = np.einsum('cpd,p,dm->cm', sales_qty, price, date2month, optimize=True)
t1 = time.perf_counter()
print("наивный порядок:", round((t1-t0)*1e3, 3), "мс, shape", res.shape)

path, info = np.einsum_path('cpd,p,dm->cm', sales_qty, price, date2month, optimize='optimal')
print(path)
print(info.split('\n')[2].strip(), '|', info.split('\n')[3].strip())

# Контрольная проверка через явные циклы (по подвыборке)
ref = np.zeros((NC, NM), dtype=np.float64)
for c in range(NC):
    for p in range(NP):
        for d in range(ND):
            ref[c, np.argmax(date2month[d])] += sales_qty[c,p,d]*price[p]
print("max rel err:", np.abs(ref-res).max()/np.abs(ref).max())

# фильтр WHERE region -> срез, затем свёртка
mask = np.zeros(NC, dtype=np.float32); mask[rng.integers(0,NC,20)] = 1
t0=time.perf_counter()
res2 = np.einsum('c,cpd,p,dm->cm', mask, sales_qty, price, date2month, optimize=True)
t1=time.perf_counter()
print("с фильтром:", round((t1-t0)*1e3,3), "мс")
