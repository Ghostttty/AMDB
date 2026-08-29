# -*- coding: utf-8 -*-
"""B12: четыре исполнителя на одном наборе запросов.

Расширяет `bench_clickhouse.py` столбцом ускорителя, чтобы числа с GPU можно
было положить рядом с уже снятыми на CPU. Набор запросов, данные, порядок
прогонов и построчная сверка — те же самые, поэтому столбцы AMDB (CPU),
DuckDB и ClickHouse должны совпасть с выводом `bench_clickhouse.py` на той же
машине. Если не совпали — сравнивать с GPU нечего, и об этом стенд сообщит.

Столбцов ускорителя два, и разница между ними существенна:

* **GPU** — обычный режим: гиперкуб живёт в оперативной памяти, операнды
  переносятся на карту при каждом запросе;
* **GPU резид.** — куб загружен на карту заранее и остаётся там между
  запросами. Это рабочий режим витрины; в нём переносится только результат.

Второй режим требует, чтобы куб помещался в память карты, и стенд это
проверяет, а не предполагает.

    python bench/bench_engines.py --rows 5000000
    python bench/bench_engines.py --rows 500000 --no-clickhouse

Без CUDA стенд отработает, оставив столбцы ускорителя пустыми: остальные три
колонки от этого не зависят.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb.exec.engine import NumpyEngine, TorchEngine, gpu_available, gpu_info
from amdb.exec.parallel import ParallelEngine
from bench.workload import (
    QUERIES,
    build_amdb,
    build_duckdb,
    make_workload,
    normalize,
    to_clickhouse,
)


def timeit(fn, repeat: int) -> tuple[float, object]:
    out = fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def ch_rows(session, sql: str) -> list[tuple]:
    import io

    import pandas as pd

    text = str(session.query(sql, "CSV"))
    if not text.strip():
        return []
    return list(pd.read_csv(io.StringIO(text), header=None)
                .itertuples(index=False, name=None))


def cube_bytes(db) -> int:
    return sum(db.stats(name)["bytes_in_memory"] for name in db.cubes)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--products", type=int, default=100)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                   help="точность вычислений на ускорителе")
    p.add_argument("--no-clickhouse", action="store_true")
    p.add_argument("--parallel", action="store_true",
                   help="добавить столбец разбиения по скоттовым сечениям")
    p.add_argument("--workers", type=int, default=0,
                   help="процессов для параллельного столбца; 0 — по числу ядер")
    a = p.parse_args()

    work = make_workload(a.rows, a.customers, a.products, a.days, a.seed)
    db = build_amdb(work)
    con = build_duckdb(work)

    session = None
    floor = 0.0
    if not a.no_clickhouse:
        try:
            from bench.workload import build_clickhouse, clickhouse_overhead

            session = build_clickhouse(work)
            floor = clickhouse_overhead(session)
        except ImportError:
            print("chdb не установлен — столбец ClickHouse пропущен.\n")

    gpu = gpu_res = None
    if gpu_available():
        import torch

        dtype = torch.float64 if a.dtype == "float64" else torch.float32
        gpu = TorchEngine(dtype=dtype)
        gpu_res = TorchEngine(dtype=dtype, resident=True)
        info = gpu_info()
        need = cube_bytes(db) * (8 if a.dtype == "float64" else 4) / 8
        print(f"GPU: {info['name']}, {info['memory_gib']} ГиБ; точность {a.dtype}; "
              f"кубы занимают {need / 2**20:.0f} МиБ из "
              f"{info['memory_gib'] * 1024:.0f} МиБ памяти карты")
        if need > info["memory_gib"] * 2**30 * 0.8:
            print("ВНИМАНИЕ: кубы не поместятся на карту — резидентный режим "
                  "работать не будет.")
    else:
        print("CUDA недоступна — столбцы ускорителя останутся пустыми.")

    print(f"{a.rows:,} строк фактов, гиперкуб "
          f"{a.customers}×{a.products}×{a.days}\n".replace(",", " "))

    par = None
    if a.parallel:
        par = ParallelEngine(workers=a.workers or None, min_cells=0)
        print(f"параллельный столбец: {par.workers} процессов, "
              f"по одному потоку BLAS в каждом")

    head = (f"{'запрос':<38}{'AMDB CPU':>10}{'по сечен.':>11}{'GPU':>9}"
            f"{'GPU резид.':>12}{'DuckDB':>9}{'ClickHouse':>12}{'сверка':>9}")
    print(head)
    print("-" * len(head))

    cpu_engine = NumpyEngine()
    failures = 0
    for q in QUERIES:
        db.use_engine(cpu_engine)
        t_cpu, res_a = timeit(lambda: db.sql(q.amdb).to_pandas(), a.repeat)
        t_duck, res_d = timeit(lambda: con.execute(q.duck).fetchdf(), a.repeat)

        ref = normalize(list(res_d.itertuples(index=False, name=None)), len(q.keys))
        got = normalize(list(res_a.itertuples(index=False, name=None)), len(q.keys))

        def same(rows):
            return len(rows) == len(ref) and all(
                g[:-1] == r[:-1]
                and abs(g[-1] - r[-1]) <= q.tolerance * max(abs(r[-1]), 1.0)
                for g, r in zip(rows, ref))

        ok = same(got)

        cell_par = "          —"
        if par is not None:
            db.use_engine(par)
            t_par, res_p = timeit(lambda: db.sql(q.amdb).to_pandas(), a.repeat)
            ok = ok and same(normalize(
                list(res_p.itertuples(index=False, name=None)), len(q.keys)))
            cell_par = f"{t_par * 1e3:10.2f}"
            db.use_engine(cpu_engine)

        cell_gpu = cell_res = "        —"
        if gpu is not None:
            db.use_engine(gpu)
            t_gpu, res_g = timeit(lambda: db.sql(q.amdb).to_pandas(), a.repeat)
            ok = ok and same(normalize(
                list(res_g.itertuples(index=False, name=None)), len(q.keys)))
            cell_gpu = f"{t_gpu * 1e3:8.2f}"
            # Резидентный режим: операнды закрепляются на карте и между
            # запросами не переносятся. Разница с предыдущим столбцом и есть
            # цена переноса через шину.
            db.use_engine(gpu_res)
            t_res, res_r = timeit(lambda: db.sql(q.amdb).to_pandas(), a.repeat)
            ok = ok and same(normalize(
                list(res_r.itertuples(index=False, name=None)), len(q.keys)))
            cell_res = f"{t_res * 1e3:11.2f}"
            db.use_engine(cpu_engine)

        cell_ch = "          —"
        if session is not None:
            ch_sql = to_clickhouse(q.duck)
            t_ch, _ = timeit(lambda: session.query(ch_sql, "Arrow"), a.repeat)
            ok = ok and same(normalize(ch_rows(session, ch_sql), len(q.keys)))
            cell_ch = f"{max(t_ch - floor, 0.0) * 1e3:11.2f}"

        failures += not ok
        print(f"{q.name:<38}{t_cpu * 1e3:>10.2f}{cell_par}{cell_gpu}{cell_res}"
              f"{t_duck * 1e3:>9.2f}{cell_ch}"
              f"{'совпало' if ok else 'РАСХОЖД.':>9}")

    print()
    if failures:
        print(f"ВНИМАНИЕ: расхождение результатов на {failures} запросах — "
              "замер недействителен.")
        return 2
    if par is not None:
        par.close()
    print("Все результаты совпали во всех исполнителях.")
    print("Столбцы AMDB CPU, DuckDB и ClickHouse должны совпасть с выводом")
    print("bench_clickhouse.py на этой же машине — это и есть точка сверки.")
    if gpu_res is not None:
        print(f"На карте закреплено {gpu_res.resident_bytes() / 2**20:.0f} МиБ "
              "операндов.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
