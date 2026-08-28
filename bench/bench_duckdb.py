# -*- coding: utf-8 -*-
"""Сравнение AMDB и DuckDB на одной базе данных.

Стенд формирует звёздную схему, загружает её в обе системы, выполняет один и
тот же набор запросов и сверяет результаты. Замер без сверки бессмыслен:
быстрый неправильный ответ ничего не стоит, поэтому расхождение в результате
здесь считается провалом запроса, а не примечанием.

    python bench/bench_duckdb.py --rows 500000
    python bench/bench_duckdb.py --rows 2000000 --skew 1.2 --threads 8

Что важно понимать про честность сравнения:

* AMDB строит гиперкуб **один раз**, DuckDB работает по сырым таблицам при
  каждом запросе. Поэтому в отчёте есть столбец окупаемости: со скольких
  запросов к одному набору данных подход себя оправдывает;
* обе системы многопоточные (DuckDB — свой пул, AMDB — потоки BLAS);
* размеры измерений задают число ячеек куба, и при их росте AMDB упирается
  в память раньше, чем DuckDB, — это видно по строке заполненности.
"""
from __future__ import annotations

import argparse
import time

try:  # запуск из копии репозитория, без установки пакета
    import amdb  # noqa: F401
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from amdb.exec.engine import blas_info
from bench.workload import QUERIES, build_amdb, build_duckdb, make_workload, normalize


def timeit(fn, repeat: int) -> tuple[float, object]:
    out = fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:,.2f}".replace(",", " ")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", type=int, default=500_000)
    p.add_argument("--customers", type=int, default=100)
    p.add_argument("--products", type=int, default=100)
    p.add_argument("--days", type=int, default=100)
    p.add_argument("--skew", type=float, default=0.0,
                   help="показатель степенного закона (0 = равномерно)")
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--threads", type=int, default=None,
                   help="ограничить число потоков DuckDB")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fetch", choices=["df", "tuples"], default="df",
                   help="формат выдачи результата: кадр данных или кортежи Python")
    a = p.parse_args()

    try:
        import duckdb
    except ImportError:
        print("DuckDB не установлен. Установите: pip install duckdb")
        return 1

    info = blas_info()
    print(f"NumPy {info['numpy']}, BLAS {info.get('blas')}; DuckDB {duckdb.__version__}")
    print(f"Формат выдачи: {'кадр данных' if a.fetch == 'df' else 'кортежи Python'} "
          "(на широких результатах выбор формата решает исход сравнения — см. --fetch)")
    print(f"Данные: {a.rows:,} строк фактов, гиперкуб "
          f"{a.customers}×{a.products}×{a.days}"
          f"{f', перекос {a.skew}' if a.skew else ''}\n".replace(",", " "))

    work = make_workload(a.rows, a.customers, a.products, a.days, a.seed, a.skew)

    # --- построение баз -----------------------------------------------------
    t0 = time.perf_counter()
    db = build_amdb(work)
    t_amdb_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    con = build_duckdb(work, a.threads)
    con.execute("SELECT COUNT(*) FROM sales").fetchall()      # прогрев
    t_duck_load = time.perf_counter() - t0

    stats = db.stats("sales")
    total_mib = sum(db.stats(n)["bytes_in_memory"] for n in db.cubes) / 2**20
    print("Построение баз:")
    print(f"  AMDB   — гиперкуб за {fmt_ms(t_amdb_load)} мс; "
          f"{stats['total_cells']:,} ячеек, заполнено {stats['fill_factor']:.2%}, "
          f"{total_mib:.1f} МиБ во всех кубах".replace(",", " "))
    print(f"  DuckDB — таблицы за {fmt_ms(t_duck_load)} мс "
          "(факты остаются сырыми строками)\n")

    # --- запросы ------------------------------------------------------------
    header = (f"{'запрос':<38}{'AMDB, мс':>11}{'DuckDB, мс':>12}"
              f"{'отношение':>11}{'сверка':>9}{'окупаемость':>14}")
    print(header)
    print("-" * len(header))

    wins = losses = mismatches = 0
    rows_report = []
    for q in QUERIES:
        try:
            if a.fetch == "df":
                t_amdb, res_amdb = timeit(lambda: db.sql(q.amdb).to_pandas(), a.repeat)
                got = normalize(list(res_amdb.itertuples(index=False, name=None)),
                                len(q.keys))
            else:
                t_amdb, res_amdb = timeit(lambda: db.sql(q.amdb).rows, a.repeat)
                got = normalize([tuple(r) for r in res_amdb], len(q.keys))
        except Exception as e:
            print(f"{q.name:<38}{'ошибка AMDB':>11}   {type(e).__name__}: {e}")
            mismatches += 1
            continue
        try:
            if a.fetch == "df":
                t_duck, res_duck = timeit(lambda: con.execute(q.duck).fetchdf(), a.repeat)
                ref = normalize(list(res_duck.itertuples(index=False, name=None)),
                                len(q.keys))
            else:
                t_duck, res_duck = timeit(lambda: con.execute(q.duck).fetchall(), a.repeat)
                ref = normalize(res_duck, len(q.keys))
        except Exception as e:
            print(f"{q.name:<38}{'ошибка DuckDB':>11}   {type(e).__name__}: {e}")
            mismatches += 1
            continue

        ok = len(got) == len(ref) and all(
            g[:-1] == r[:-1] and abs(g[-1] - r[-1]) <= q.tolerance * max(abs(r[-1]), 1.0)
            for g, r in zip(got, ref)
        )
        ratio = t_duck / t_amdb
        if ok:
            wins += ratio > 1.05
            losses += ratio < 0.95
        else:
            mismatches += 1

        # Со скольких запросов окупается разовое построение гиперкуба
        saved = t_duck - t_amdb
        payback = (f"{int(np.ceil(t_amdb_load / saved)):,} зпр".replace(",", " ")
                   if saved > 0 else "не окупается")

        print(f"{q.name:<38}{fmt_ms(t_amdb):>11}{fmt_ms(t_duck):>12}"
              f"{ratio:>10.2f}×{'совпало' if ok else 'РАСХОЖД.':>9}{payback:>14}")
        rows_report.append((q, ratio, ok, len(got)))

    # --- итоги --------------------------------------------------------------
    print()
    print(f"Итог: AMDB быстрее на {wins} запросах, DuckDB — на {losses}, "
          f"паритет на {len(QUERIES) - wins - losses - mismatches}.")
    if mismatches:
        print(f"ВНИМАНИЕ: расхождений или ошибок — {mismatches}. "
              "Замеры при расхождении недействительны.")
    else:
        print("Все результаты совпали с DuckDB — замеры корректны.")

    def num(n: int) -> str:
        return f"{n:,}".replace(",", " ")

    print("\nПояснения к отдельным запросам:")
    for q, ratio, ok, nrows in rows_report:
        if q.expect and ok:
            print(f"  {q.name} — {ratio:.2f}×, строк {num(nrows)}: {q.expect}")
    if losses:
        print("\nГде AMDB уступает:")
        for q, ratio, ok, nrows in rows_report:
            if ok and ratio < 0.95:
                print(f"  {q.name} — {ratio:.2f}×, результат {num(nrows)} строк")

    # Зависимость преимущества от ширины выдачи — главное, что видно в наборе
    valid = [r for r in rows_report if r[2]]
    if valid:
        best = max(valid, key=lambda r: r[1])
        widest = max(valid, key=lambda r: r[3])
        print(f"\nЛучшее отношение: {best[1]:.1f}× — {best[0].name}, "
              f"{num(best[3])} строк.")
        print(f"Самая широкая выдача: {widest[1]:.1f}× — {num(widest[3])} строк.")
        print("Свёртка дешевеет с каждым свёрнутым измерением, а сборка "
              "результата — нет; отсюда и разрыв.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
