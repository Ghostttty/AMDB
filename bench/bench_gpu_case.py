# -*- coding: utf-8 -*-
"""B14: стоит ли расширяться на GPU — развёрнутый ответ на одном стенде.

Рассчитан на Ubuntu Linux: там доступны и встраиваемый ClickHouse (chdb), и
сборки PyTorch с CUDA, поэтому все четыре исполнителя сравниваются в одном
процессе и в одной среде.

Одной таблицы для ответа недостаточно, поэтому стенд состоит из четырёх частей,
каждая из которых закрывает свой вопрос.

**Часть 1, ядро.** С какого размера куба ускоритель обгоняет процессор — отдельно
с переносом операндов через шину и отдельно в резидентном режиме, когда куб
закреплён на карте. Здесь же измеряется цена двойной точности: у потребительских
карт она выполняется на малой доле скорости одинарной, и замер во float64 на них
покажет свойства карты, а не подхода.

**Часть 2, звёздная схема.** Тот же набор из одиннадцати запросов, что в
остальных стендах, но развёрнутый по размеру куба. Отвечает на вопрос, включится
ли ускоритель на обычной аналитической нагрузке и даст ли он там что-нибудь.

**Часть 3, произведение куба на куб.** Соединение двух таблиц фактов по общему
измерению, остающемуся в группировке: λ >= 1, μ >= 1. Это случай, где матричная
модель работает своей главной операцией, а реляционным движкам приходится
выполнять соединение. Если преимущество ускорителя есть где-то, то здесь.

**Часть 4, вывод.** Считается из измеренного, а не вписан заранее.

Результаты сверяются построчно во всех исполнителях; расхождение считается
провалом замера.

    python bench/bench_gpu_case.py
    python bench/bench_gpu_case.py --dtype float32 --sides 100,150,200,250
    python bench/bench_gpu_case.py --quick

Без CUDA стенд отработает целиком, оставив столбцы ускорителя пустыми: части 2 и
3 остаются осмысленными и без него.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amdb import Database
from amdb.core import build_einsum
from amdb.exec.engine import (
    GPU_MIN_FLOPS,
    GPU_MIN_INTENSITY,
    NumpyEngine,
    TorchEngine,
    blas_info,
    gpu_available,
    gpu_info,
    spec_cost,
)
from bench.workload import (
    QUERIES,
    build_amdb,
    build_duckdb,
    make_workload,
    normalize,
    to_clickhouse,
)

SEPARATOR = "=" * 96


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


def same_as(reference, rows, keys: int, tolerance: float) -> bool:
    got = normalize(rows, keys)
    return len(got) == len(reference) and all(
        g[:-1] == r[:-1] and abs(g[-1] - r[-1]) <= tolerance * max(abs(r[-1]), 1.0)
        for g, r in zip(got, reference))


# --- часть 0: среда ---------------------------------------------------------
def report_environment(dtype_name: str) -> dict:
    print(SEPARATOR)
    print("СРЕДА")
    print(SEPARATOR)
    info = blas_info()
    print(f"ядер: {os.cpu_count()}; NumPy {info['numpy']}, "
          f"BLAS {info.get('blas')} {info.get('blas_version', '')}")
    try:
        import duckdb
        print(f"DuckDB {duckdb.__version__}", end="")
    except ImportError:
        print("DuckDB не установлен", end="")
    try:
        import chdb
        print(f"; ClickHouse через chdb {chdb.__version__}")
    except ImportError:
        print("; chdb не установлен — столбец ClickHouse будет пропущен")

    if not gpu_available():
        print("CUDA недоступна: столбцы ускорителя останутся пустыми.\n")
        return {}

    gpu = gpu_info()
    print(f"GPU: {gpu['name']}, {gpu['memory_gib']} ГиБ, способность "
          f"{gpu['capability']}; torch {gpu['torch']}, CUDA {gpu['cuda']}")

    import torch

    size = 2048
    ratios = {}
    for name, dt in (("float32", torch.float32), ("float64", torch.float64)):
        x = torch.randn(size, size, device="cuda", dtype=dt)
        torch.matmul(x, x); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            torch.matmul(x, x)
        torch.cuda.synchronize()
        ratios[name] = (time.perf_counter() - t0) / 3
        del x
    torch.cuda.empty_cache()
    penalty = ratios["float64"] / ratios["float32"]
    gflops = 2 * size ** 3 / ratios["float64"] / 1e9
    verdict = ("пригодна для двойной точности" if penalty <= 4 else
               "двойная точность урезана — снимайте и float32" if penalty <= 16 else
               "потребительская карта: во float64 замер покажет её ограничение")
    print(f"двойная точность медленнее одинарной в {penalty:.1f} раза "
          f"({gflops:.0f} Гфлопс) — {verdict}")
    print(f"замеры ведутся в {dtype_name}\n")
    return {"penalty": penalty, "memory_gib": gpu["memory_gib"], "name": gpu["name"]}


# --- часть 1: ядро -----------------------------------------------------------
def part_kernel(sides, repeat: int, dtype_name: str) -> list[dict]:
    print(SEPARATOR)
    print("ЧАСТЬ 1. Ядро: (1,1)-свёртка куба на куб")
    print(SEPARATOR)
    cpu = NumpyEngine()
    spec = build_einsum(3, 3, 1, 1)
    rows = []
    if not gpu_available():
        print("пропущено: нет ускорителя\n")
        return rows

    import torch

    torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32
    np_dtype = np.float64 if dtype_name == "float64" else np.float32
    gpu = TorchEngine(dtype=torch_dtype)

    head = (f"{'сторона':<9}{'ячеек':>13}{'МиБ':>8}{'CPU':>10}{'GPU+перенос':>13}"
            f"{'ускор.':>9}{'GPU резид.':>12}{'ускор.':>9}{'сверка':>9}")
    print(head)
    print("-" * len(head))
    for side in sides:
        rng = np.random.default_rng(0)
        A = rng.random((side, side, side)).astype(np_dtype)
        B = rng.random((side, side, side)).astype(np_dtype)
        reference = cpu.einsum(spec, A, B)
        t_cpu, _ = timeit(lambda: cpu.einsum(spec, A, B), repeat)
        t_move, _ = timeit(lambda: gpu.einsum(spec, A, B), repeat)

        da, db_ = gpu.upload(A), gpu.upload(B)

        def resident():
            out = gpu.download(gpu.einsum_device(spec, da, db_))
            gpu.synchronize()
            return out

        got = resident()
        t_res, _ = timeit(resident, repeat)
        tol = 1e-10 if dtype_name == "float64" else 1e-4
        ok = np.allclose(got, reference, rtol=tol,
                         atol=tol * max(float(np.abs(reference).max()), 1.0))
        del da, db_
        torch.cuda.empty_cache()

        rows.append({"side": side, "cpu": t_cpu, "move": t_move, "resident": t_res,
                     "ok": ok})
        print(f"{side:<9}{side ** 3:>13,}{A.nbytes / 2**20:>8.0f}"
              f"{t_cpu * 1e3:>9.1f}м{t_move * 1e3:>12.1f}м{t_cpu / t_move:>8.2f}×"
              f"{t_res * 1e3:>11.1f}м{t_cpu / t_res:>8.2f}×"
              f"{'совпало' if ok else 'РАСХОЖД.':>9}".replace(",", " "))
    print()
    return rows


# --- часть 2: звёздная схема --------------------------------------------------
def part_star(sides, rows_count: int, repeat: int, dtype_name: str,
              use_clickhouse: bool) -> list[dict]:
    print(SEPARATOR)
    print("ЧАСТЬ 2. Звёздная схема: одиннадцать запросов, развёртка по размеру куба")
    print(SEPARATOR)
    results = []
    for side in sides:
        work = make_workload(rows_count, side, side, side, 42)
        db = build_amdb(work)
        con = build_duckdb(work)
        session = None
        floor = 0.0
        if use_clickhouse:
            try:
                from bench.workload import build_clickhouse, clickhouse_overhead
                session = build_clickhouse(work)
                floor = clickhouse_overhead(session)
            except ImportError:
                session = None

        engines = {"cpu": NumpyEngine()}
        if gpu_available():
            import torch
            dt = torch.float64 if dtype_name == "float64" else torch.float32
            engines["gpu"] = TorchEngine(dtype=dt)
            engines["gpu_res"] = TorchEngine(dtype=dt, resident=True)

        totals = {k: 0.0 for k in ("cpu", "gpu", "gpu_res", "duck", "ch")}
        failures = 0
        for q in QUERIES:
            db.use_engine(engines["cpu"])
            t_cpu, res_a = timeit(lambda: db.sql(q.amdb).to_pandas(), repeat)
            t_duck, res_d = timeit(lambda: con.execute(q.duck).fetchdf(), repeat)
            reference = normalize(
                list(res_d.itertuples(index=False, name=None)), len(q.keys))
            ok = same_as(reference, list(res_a.itertuples(index=False, name=None)),
                         len(q.keys), q.tolerance)
            totals["cpu"] += t_cpu
            totals["duck"] += t_duck

            for key in ("gpu", "gpu_res"):
                if key in engines:
                    db.use_engine(engines[key])
                    t, res = timeit(lambda: db.sql(q.amdb).to_pandas(), repeat)
                    ok &= same_as(reference,
                                  list(res.itertuples(index=False, name=None)),
                                  len(q.keys), q.tolerance)
                    totals[key] += t
            db.use_engine(engines["cpu"])

            if session is not None:
                sql = to_clickhouse(q.duck)
                t_ch, _ = timeit(lambda: session.query(sql, "Arrow"), repeat)
                ok &= same_as(reference, ch_rows(session, sql), len(q.keys),
                              q.tolerance)
                totals["ch"] += max(t_ch - floor, 0.0)
            failures += not ok

        fill = db.stats("sales")["fill_factor"]
        results.append({"side": side, "fill": fill, "failures": failures, **totals})
        if "gpu_res" in engines:
            engines["gpu_res"].clear_resident()

    head = (f"{'куб':<14}{'заполн.':>9}{'фактов/яч':>11}{'AMDB CPU':>11}"
            f"{'AMDB GPU':>11}{'GPU резид.':>12}{'DuckDB':>10}{'ClickHouse':>12}"
            f"{'сверка':>9}")
    print(head)
    print("-" * len(head))
    for r in results:
        per_cell = rows_count / (r["side"] ** 3)
        cells = [f"{r['gpu'] * 1e3:10.1f}" if r["gpu"] else "         —",
                 f"{r['gpu_res'] * 1e3:11.1f}" if r["gpu_res"] else "          —",
                 f"{r['ch'] * 1e3:11.1f}" if r["ch"] else "          —"]
        print(f"{str(r['side']) + '³':<14}{r['fill']:>8.1%}{per_cell:>11.2f}"
              f"{r['cpu'] * 1e3:>11.1f}{cells[0]}{cells[1]}"
              f"{r['duck'] * 1e3:>10.1f}{cells[2]}"
              f"{'совпало' if not r['failures'] else 'РАСХОЖД.':>9}")
    print("суммарное время одиннадцати запросов, мс\n")
    return results


# --- часть 3: произведение куба на куб ----------------------------------------
def build_two_fact(side: int, rows_count: int, seed: int = 7):
    """Две таблицы фактов с общим измерением: продажи и цены по (товар, дата)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    sales = pd.DataFrame({
        "customer": rng.integers(0, side, rows_count),
        "product": rng.integers(0, side, rows_count),
        "date": rng.integers(0, side, rows_count),
        "quantity": rng.random(rows_count) * 10.0,
    })
    grid = np.indices((side, side)).reshape(2, -1)
    price = pd.DataFrame({"product": grid[0], "date": grid[1],
                          "price": rng.random(side * side) * 100.0})
    return sales, price


TWO_FACT_QUERIES = [
    ("λ=1, μ=1: группировка по клиенту и товару",
     "SELECT customer, product, SUM(sales.quantity * price.price) AS v "
     "FROM sales JOIN price GROUP BY customer, product",
     "SELECT s.customer, s.product, SUM(s.quantity * p.price) AS v "
     "FROM sales s JOIN price p ON s.product = p.product AND s.date = p.date "
     "GROUP BY s.customer, s.product", 2),
    ("λ=1, μ=2: группировка по товару",
     "SELECT product, SUM(sales.quantity * price.price) AS v "
     "FROM sales JOIN price GROUP BY product",
     "SELECT s.product, SUM(s.quantity * p.price) AS v "
     "FROM sales s JOIN price p ON s.product = p.product AND s.date = p.date "
     "GROUP BY s.product", 1),
    ("λ=2, μ=0: рост ранга, группировка по трём осям",
     "SELECT customer, product, date, SUM(sales.quantity * price.price) AS v "
     "FROM sales JOIN price GROUP BY customer, product, date",
     "SELECT s.customer, s.product, s.date, SUM(s.quantity * p.price) AS v "
     "FROM sales s JOIN price p ON s.product = p.product AND s.date = p.date "
     "GROUP BY s.customer, s.product, s.date", 3),
]


def part_two_fact(sides, rows_count: int, repeat: int, dtype_name: str,
                  use_clickhouse: bool) -> list[dict]:
    print(SEPARATOR)
    print("ЧАСТЬ 3. Произведение куба на куб: соединение двух таблиц фактов")
    print(SEPARATOR)
    print("Обе колоночные СУБД получают таблицы гранулярности гиперкуба —")
    print("тот же предпосчёт, что и у матричной модели, иначе сравнение")
    print("измеряло бы предварительную агрегацию, а не исполнение.\n")

    import duckdb

    results = []
    for side in sides:
        sales, price = build_two_fact(side, rows_count)
        db = Database()
        db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
        db.load_frame(price, ["product", "date"], "price", "price")

        con = duckdb.connect()
        con.register("_s", sales)
        con.register("_p", price)
        # Гранулярность куба: факт свёрнут до (клиент, товар, дата).
        con.execute("CREATE TABLE sales AS SELECT customer, product, date, "
                    "SUM(quantity) AS quantity FROM _s GROUP BY 1, 2, 3")
        con.execute("CREATE TABLE price AS SELECT * FROM _p")
        con.unregister("_s"); con.unregister("_p")

        session = None
        floor = 0.0
        if use_clickhouse:
            try:
                from chdb import session as chs

                from bench.workload import NOT_NULL, clickhouse_overhead
                tmp = tempfile.mkdtemp(prefix="amdb-gpu-case-")
                agg = con.execute("SELECT * FROM sales").fetchdf()
                agg.to_parquet(os.path.join(tmp, "sales.parquet"), index=False)
                price.to_parquet(os.path.join(tmp, "price.parquet"), index=False)
                session = chs.Session()
                for name in ("sales", "price"):
                    session.query(f"DROP TABLE IF EXISTS {name}")
                    session.query(
                        f"CREATE TABLE {name} ENGINE=Memory AS SELECT * FROM "
                        f"file('{os.path.join(tmp, name + '.parquet')}', Parquet) "
                        f"{NOT_NULL}")
                floor = clickhouse_overhead(session)
            except ImportError:
                session = None

        engines = {"cpu": NumpyEngine()}
        if gpu_available():
            import torch
            dt = torch.float64 if dtype_name == "float64" else torch.float32
            engines["gpu"] = TorchEngine(dtype=dt)
            engines["gpu_res"] = TorchEngine(dtype=dt, resident=True)

        for label, amdb_sql, duck_sql, keys in TWO_FACT_QUERIES:
            db.use_engine(engines["cpu"])
            t_cpu, res_a = timeit(lambda: db.sql(amdb_sql).to_pandas(), repeat)
            t_duck, res_d = timeit(lambda: con.execute(duck_sql).fetchdf(), repeat)
            reference = normalize(
                list(res_d.itertuples(index=False, name=None)), keys)
            ok = same_as(reference, list(res_a.itertuples(index=False, name=None)),
                         keys, 1e-6)

            times = {"cpu": t_cpu, "duck": t_duck, "gpu": 0.0, "gpu_res": 0.0,
                     "ch": 0.0}
            for key in ("gpu", "gpu_res"):
                if key in engines:
                    db.use_engine(engines[key])
                    t, res = timeit(lambda: db.sql(amdb_sql).to_pandas(), repeat)
                    ok &= same_as(reference,
                                  list(res.itertuples(index=False, name=None)),
                                  keys, 1e-6)
                    times[key] = t
            db.use_engine(engines["cpu"])

            if session is not None:
                t_ch, _ = timeit(lambda: session.query(duck_sql, "Arrow"), repeat)
                ok &= same_as(reference, ch_rows(session, duck_sql), keys, 1e-6)
                times["ch"] = max(t_ch - floor, 0.0)

            results.append({"side": side, "label": label, "ok": ok,
                            "spec": db.explain(amdb_sql), **times})

        if "gpu_res" in engines:
            engines["gpu_res"].clear_resident()

    head = (f"{'куб':<7}{'запрос':<44}{'AMDB CPU':>10}{'AMDB GPU':>10}"
            f"{'GPU резид.':>12}{'DuckDB':>10}{'ClickHouse':>12}{'сверка':>9}")
    print(head)
    print("-" * len(head))
    for r in results:
        cells = [f"{r['gpu'] * 1e3:9.1f}" if r["gpu"] else "        —",
                 f"{r['gpu_res'] * 1e3:11.1f}" if r["gpu_res"] else "          —",
                 f"{r['ch'] * 1e3:11.1f}" if r["ch"] else "          —"]
        print(f"{str(r['side']) + '³':<7}{r['label']:<44}{r['cpu'] * 1e3:>10.1f}"
              f"{cells[0]}{cells[1]}{r['duck'] * 1e3:>10.1f}{cells[2]}"
              f"{'совпало' if r['ok'] else 'РАСХОЖД.':>9}")
    print()
    return results


# --- часть 4: вывод -------------------------------------------------------------
def verdict(kernel, star, two_fact, env) -> None:
    print(SEPARATOR)
    print("ЧАСТЬ 4. Вывод")
    print(SEPARATOR)

    if not gpu_available():
        print("Ускорителя не было, поэтому вывод о нём сделать нельзя.")
        print("Части 2 и 3 дают сравнение процессорного пути с DuckDB и ClickHouse.")
        return

    wins = [r for r in kernel if r["cpu"] / r["resident"] > 1.0]
    if wins:
        first = min(wins, key=lambda r: r["side"])
        best = max(kernel, key=lambda r: r["cpu"] / r["resident"])
        print(f"1. На ядре ускоритель обгоняет процессор начиная с куба "
              f"{first['side']}³ в резидентном режиме; лучшее отношение "
              f"{best['cpu'] / best['resident']:.1f}× при {best['side']}³.")
        moved = max(kernel, key=lambda r: r["cpu"] / r["move"])
        print(f"   С переносом операндов лучшее отношение "
              f"{moved['cpu'] / moved['move']:.1f}× — разница между этими двумя "
              "числами и есть цена шины.")
    else:
        print("1. На ядре ускоритель не обогнал процессор ни на одном размере: "
              "либо кубы малы, либо двойная точность на этой карте урезана.")

    star_gpu = [r for r in star if r["gpu"]]
    if star_gpu:
        best = max(star_gpu, key=lambda r: r["cpu"] / max(r["gpu_res"], 1e-12))
        print(f"2. На звёздной схеме лучшее, что дал ускоритель, — "
              f"{best['cpu'] / max(best['gpu_res'], 1e-12):.2f}× при кубе "
              f"{best['side']}³ (резидентно).")
    else:
        print("2. На звёздной схеме ускоритель не включался.")

    tf_gpu = [r for r in two_fact if r["gpu_res"]]
    if tf_gpu:
        best = max(tf_gpu, key=lambda r: r["cpu"] / max(r["gpu_res"], 1e-12))
        ratio = best["cpu"] / max(best["gpu_res"], 1e-12)
        versus = best["duck"] / max(best["gpu_res"], 1e-12)
        print(f"3. На произведении куба на куб лучшее отношение к процессору "
              f"{ratio:.2f}×, к DuckDB {versus:.1f}× (куб {best['side']}³, "
              f"{best['label']}).")
        print()
        if ratio > 3 and versus > 10:
            print("ВЫВОД: расширение на GPU оправдано. Ускоритель даёт кратный")
            print("выигрыш над процессорным путём именно там, где матричная модель")
            print("работает своей главной операцией, и отрыв от реляционных")
            print("движков там же наибольший.")
        elif ratio > 1.5:
            print("ВЫВОД: выигрыш есть, но умеренный. Прежде вложений стоит")
            print("проверить, встречается ли произведение куба на куб в реальной")
            print("нагрузке достаточно часто, чтобы окупить перенос данных.")
        else:
            print("ВЫВОД: расширение на GPU себя не окупает на этой карте и этих")
            print("размерах. Процессорный путь с многопоточным BLAS не уступает,")
            print("а перенос через шину съедает то, что даёт счёт.")
    else:
        print("3. На произведении куба на куб ускоритель не включался.")

    if env.get("penalty", 0) > 16:
        print()
        print("ОГОВОРКА: двойная точность на этой карте урезана более чем "
              "шестнадцатикратно.")
        print("Повторите с --dtype float32, иначе вывод описывает карту, а не подход.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sides", default="100,150,200,250",
                   help="стороны кубов через запятую")
    p.add_argument("--kernel-sides", default="100,200,300,400",
                   help="стороны для части 1 (ядро)")
    p.add_argument("--rows", type=int, default=5_000_000)
    p.add_argument("--two-fact-rows", type=int, default=2_000_000)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--no-clickhouse", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="сокращённый прогон для проверки работоспособности")
    a = p.parse_args()

    if a.quick:
        a.sides, a.kernel_sides = "60,100", "100,150"
        a.rows, a.two_fact_rows, a.repeat = 500_000, 300_000, 2

    sides = [int(x) for x in a.sides.split(",")]
    kernel_sides = [int(x) for x in a.kernel_sides.split(",")]
    use_ch = not a.no_clickhouse

    env = report_environment(a.dtype)
    print(f"порог автовыбора движка: операций > {GPU_MIN_FLOPS:.0e}, "
          f"интенсивность > {GPU_MIN_INTENSITY}")
    probe = np.zeros((sides[0],) * 3)
    flops, moved = spec_cost(build_einsum(3, 3, 1, 1), probe, probe)
    print(f"для куба {sides[0]}³ × {sides[0]}³: {flops:.1e} операций, "
          f"интенсивность {flops / moved:.1f}\n")

    kernel = part_kernel(kernel_sides, a.repeat, a.dtype)
    star = part_star(sides, a.rows, a.repeat, a.dtype, use_ch)
    two_fact = part_two_fact(sides, a.two_fact_rows, a.repeat, a.dtype, use_ch)
    verdict(kernel, star, two_fact, env)

    bad = (sum(1 for r in kernel if not r["ok"])
           + sum(r["failures"] for r in star)
           + sum(1 for r in two_fact if not r["ok"]))
    if bad:
        print(f"\nВНИМАНИЕ: расхождение результатов в {bad} случаях — "
              "замер недействителен.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
