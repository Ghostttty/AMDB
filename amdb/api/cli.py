# -*- coding: utf-8 -*-
"""Командный интерфейс AMDB: загрузка данных, запросы, EXPLAIN, REPL."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ..database import Database
from ..exec.engine import blas_info, gpu_available, warn_if_reference_blas
from ..ql.binder import BindError
from ..ql.lexer import QuerySyntaxError
from ..security.rls import AccessDenied


def _open(path: str | None) -> Database:
    if path is None:
        raise SystemExit("укажите каталог базы: --db ПУТЬ")
    p = Path(path)
    if not (p / "catalog.sqlite").exists():
        raise SystemExit(f"в '{path}' нет каталога AMDB (файл catalog.sqlite)")
    return Database.open(p)


def cmd_load(args: argparse.Namespace) -> int:
    path = Path(args.db)
    db = Database.open(path) if (path / "catalog.sqlite").exists() else Database()
    if args.dimension:
        dim = db.load_dimension(
            _read(args.csv), args.key, args.name,
            attributes=args.attribute or (), measures=args.measure or ())
        print(f"загружено измерение {dim.name}: {len(dim)} значений, "
              f"атрибуты {sorted(dim.attributes)}")
    else:
        if not args.dimensions or not args.value:
            raise SystemExit("для факта укажите --dimensions и --value")
        cube = db.load_csv(args.csv, args.dimensions, args.value, args.name,
                           ordered_dims=args.ordered or ())
        s = db.stats(cube.name)
        print(f"загружен куб {cube.name}: оси {list(cube.axes)}, форма {cube.shape}, "
              f"заполненность {s['fill_factor']:.3%}, представление {cube.layout}")
    db.save(path)
    print(f"каталог сохранён в {path}")
    return 0


def _read(path: str):
    from ..storage.loader import read_csv

    return read_csv(path)


def cmd_hierarchy(args: argparse.Namespace) -> int:
    """Регистрирует иерархию измерений: ROLLUP становится (0,1)-свёрткой."""
    path = Path(args.db)
    db = _open(args.db)
    frame = _read(args.csv)
    keys = list(frame[args.key]) if not hasattr(frame, "columns") else \
        list(frame[args.key])
    parents = list(frame[args.column])
    child = db.catalog.dimension(args.child)
    unknown = [k for k in keys if k not in child]
    if unknown:
        raise SystemExit(
            f"в измерении '{args.child}' нет значений {unknown[:5]}"
            f"{' и ещё ' + str(len(unknown) - 5) if len(unknown) > 5 else ''}"
        )
    mapping = dict(zip(keys, parents))
    missing = [v for v in child.labels() if v not in mapping]
    if missing:
        raise SystemExit(
            f"для {len(missing)} значений '{args.child}' не задан родитель, "
            f"например {missing[:5]}"
        )
    h = db.add_hierarchy(args.child, args.parent, mapping)
    db.save(path)
    print(f"иерархия {h.child.name} -> {h.parent.name}: "
          f"{len(h.child)} значений сворачиваются в {len(h.parent)}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    db = _open(args.db)
    sql = args.sql or sys.stdin.read()
    result = db.sql(sql, role=args.role)
    if args.format == "csv":
        print(",".join(result.columns))
        for row in result:
            print(",".join("" if v is None else str(v) for v in row))
    elif args.format == "json":
        import json

        print(json.dumps(result.to_dicts(), ensure_ascii=False, indent=2, default=str))
    else:
        print(result.to_text(max_rows=args.limit))
    if args.timing:
        st = result.stats
        print(f"\nтрансляция {st['compile_seconds'] * 1e3:.2f} мс, "
              f"вычисление {st['compute_seconds'] * 1e3:.2f} мс, "
              f"движок {st['engine']}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    db = _open(args.db)
    print(db.explain(args.sql or sys.stdin.read(), role=args.role))
    return 0


def cmd_sokolov(args: argparse.Namespace) -> int:
    """Раскладывает план в цепочку бинарных (λ, μ)-произведений Соколова."""
    db = _open(args.db)
    print(db.sokolov(args.sql or sys.stdin.read(), role=args.role))
    return 0


def cmd_signature(args: argparse.Namespace) -> int:
    """Печатает сигнатуру двухосновной алгебраической системы."""
    from ..core.signature import SYSTEM

    print(SYSTEM.describe())
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    db = _open(args.db)
    print(db.summary())
    print()
    for name in db.cubes:
        s = db.stats(name)
        print(f"{name}: {s['nnz']:,} непустых из {s['total_cells']:,} "
              f"({s['fill_factor']:.3%}), {s['bytes_in_memory'] / 2**20:.2f} МиБ, "
              f"представление {s['layout']} (рекомендуется {s['recommended_layout']})")
    print()
    info = blas_info()
    print(f"NumPy {info['numpy']}, BLAS {info.get('blas')} {info.get('blas_version', '')}")
    print(f"GPU доступен: {'да' if gpu_available() else 'нет'}")
    warn = warn_if_reference_blas()
    if warn:
        print(f"ВНИМАНИЕ: {warn}")
    return 0


def cmd_shell(args: argparse.Namespace) -> int:
    db = _open(args.db)
    print(db.summary())
    print("\nВводите запросы. \\e ЗАПРОС — план, \\s ЗАПРОС — разложение по Соколову,"
          "\n\\i — сведения, \\q — выход.\n")
    while True:
        try:
            line = input("amdb> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("\\q", "quit", "exit"):
            return 0
        try:
            if line == "\\i":
                print(db.summary())
            elif line.startswith("\\e "):
                print(db.explain(line[3:]))
            elif line.startswith("\\s "):
                print(db.sokolov(line[3:]))
            else:
                res = db.sql(line, role=args.role)
                print(res.to_text())
                print(f"({res.stats['compute_seconds'] * 1e3:.2f} мс)")
        except (QuerySyntaxError, BindError, AccessDenied, KeyError) as e:
            print(f"ошибка: {e}")
        except Exception as e:  # pragma: no cover
            print(f"{type(e).__name__}: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="amdb", description="Алгебраическая машина баз данных (AMDB)")
    p.add_argument("--db", help="каталог базы данных")
    sub = p.add_subparsers(dest="command", required=True)

    lo = sub.add_parser("load", help="загрузить CSV в гиперкуб или измерение")
    lo.add_argument("csv")
    lo.add_argument("--dimensions", nargs="+", help="столбцы-измерения факта")
    lo.add_argument("--value", help="столбец-мера факта")
    lo.add_argument("--name", help="имя куба или измерения")
    lo.add_argument("--ordered", nargs="*", help="упорядоченные измерения")
    lo.add_argument("--dimension", action="store_true", help="загрузить как справочник")
    lo.add_argument("--key", help="ключевой столбец справочника")
    lo.add_argument("--attribute", nargs="*", help="атрибуты справочника")
    lo.add_argument("--measure", nargs="*", help="меры справочника (кубы ранга 1)")
    lo.set_defaults(func=cmd_load)

    h = sub.add_parser("hierarchy", help="зарегистрировать иерархию измерений")
    h.add_argument("csv", help="таблица соответствия child -> parent")
    h.add_argument("--child", required=True, help="дочернее измерение (например, date)")
    h.add_argument("--parent", required=True, help="родительское измерение (month)")
    h.add_argument("--key", required=True, help="столбец со значениями child")
    h.add_argument("--column", required=True, help="столбец со значениями parent")
    h.set_defaults(func=cmd_hierarchy)

    q = sub.add_parser("query", help="выполнить запрос")
    q.add_argument("sql", nargs="?")
    q.add_argument("--role")
    q.add_argument("--format", choices=["text", "csv", "json"], default="text")
    q.add_argument("--limit", type=int, default=50)
    q.add_argument("--timing", action="store_true")
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("explain", help="показать план запроса (цепочку einsum)")
    e.add_argument("sql", nargs="?")
    e.add_argument("--role")
    e.set_defaults(func=cmd_explain)

    sk = sub.add_parser("sokolov",
                        help="разложить план в цепочку (λ, μ)-произведений Соколова")
    sk.add_argument("sql", nargs="?")
    sk.add_argument("--role")
    sk.set_defaults(func=cmd_sokolov)

    sg = sub.add_parser("signature",
                        help="сигнатура алгебраической системы: сорта, операции, аксиомы")
    sg.set_defaults(func=cmd_signature)

    i = sub.add_parser("info", help="сведения о базе и окружении")
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("shell", help="интерактивный режим")
    s.add_argument("--role")
    s.set_defaults(func=cmd_shell)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (QuerySyntaxError, BindError, AccessDenied) as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 2
    except (KeyError, ValueError, MemoryError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
