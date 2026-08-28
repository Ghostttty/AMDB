# -*- coding: utf-8 -*-
"""Результат запроса: таблица со столбцами-метками и столбцами-значениями.

Результат хранится **по столбцам**, а строки собираются лениво. Причина
измеримая: на выдаче в 400 тыс. строк сборка кортежей Python занимает больше
времени, чем сама свёртка, и потребителю, которому нужен кадр pandas или
массивы, платить за неё незачем.
"""
from __future__ import annotations

from typing import Any, Iterator, Sequence


class ResultSet:
    """Таблица результата.

    Хранит либо столбцы (``column_values``), либо готовые строки, либо и то,
    и другое. Строки материализуются по первому обращению к ``rows``.
    """

    def __init__(self, columns: Sequence[str],
                 rows: Sequence[tuple] | None = None,
                 stats: dict[str, Any] | None = None,
                 column_values: Sequence[Sequence[Any]] | None = None):
        self.columns = list(columns)
        self._rows: list[tuple] | None = list(rows) if rows is not None else None
        self.column_values = list(column_values) if column_values else None
        self.stats: dict[str, Any] = stats or {}
        if self._rows is None and self.column_values is None:
            self._rows = []

    # -- доступ -------------------------------------------------------------
    @property
    def rows(self) -> list[tuple]:
        """Строки результата. Собираются при первом обращении."""
        if self._rows is None:
            self._rows = list(zip(*self.column_values)) if self.column_values else []
        return self._rows

    @property
    def materialized(self) -> bool:
        """Собраны ли уже строки — полезно при замерах."""
        return self._rows is not None

    def __len__(self) -> int:
        if self._rows is not None:
            return len(self._rows)
        return len(self.column_values[0]) if self.column_values else 0

    def __iter__(self) -> Iterator[tuple]:
        return iter(self.rows)

    def __getitem__(self, i: int) -> tuple:
        return self.rows[i]

    def column(self, name: str) -> list[Any]:
        """Один столбец. Без сборки строк, если результат хранится по столбцам."""
        i = self.columns.index(name)
        if self.column_values is not None:
            col = self.column_values[i]
            return col.tolist() if hasattr(col, "tolist") else list(col)
        return [r[i] for r in self.rows]

    # -- преобразования -----------------------------------------------------
    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(zip(self.columns, r)) for r in self.rows]

    def to_pandas(self):
        """Кадр pandas. Строится из столбцов — минуя кортежи Python."""
        import pandas as pd

        if self.column_values is not None:
            return pd.DataFrame(dict(zip(self.columns, self.column_values)))
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_text(self, max_rows: int = 50, float_fmt: str = "{:.6g}") -> str:
        """Выравненная таблица для терминала."""
        def fmt(v: Any) -> str:
            if isinstance(v, float):
                return float_fmt.format(v)
            return "" if v is None else str(v)

        shown = self.rows[:max_rows]
        table = [self.columns] + [[fmt(v) for v in r] for r in shown]
        widths = [max(len(str(row[i])) for row in table) for i in range(len(self.columns))]
        sep = "-+-".join("-" * w for w in widths)
        lines = [" | ".join(str(c).ljust(w) for c, w in zip(self.columns, widths)), sep]
        for r in table[1:]:
            lines.append(" | ".join(v.rjust(w) if _numeric(v) else v.ljust(w)
                                    for v, w in zip(r, widths)))
        if len(self) > max_rows:
            lines.append(f"... ещё {len(self) - max_rows} строк")
        lines.append(f"({len(self)} строк)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ResultSet({len(self)} строк, столбцы={self.columns})"


def _numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
