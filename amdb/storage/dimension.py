# -*- coding: utf-8 -*-
"""Измерения, их словари, атрибуты и иерархии.

Словарь измерения — append-only: позиция значения в словаре есть индекс вдоль
оси гиперкуба, поэтому переупорядочивание сделало бы ранее сохранённые кубы
некорректными (риск R9 техпроекта). Новые значения получают следующие ординалы.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ..core.ops import rollup_matrix


class Dimension:
    """Измерение: упорядоченный словарь значений + атрибуты."""

    def __init__(self, name: str, values: Iterable[Any] = (), ordered: bool = False):
        self.name = name
        self.ordered = ordered
        self._values: list[Any] = []
        self._pos: dict[Any, int] = {}
        self.attributes: dict[str, np.ndarray] = {}
        #: Кэш для быстрого кодирования; сбрасывается при дозагрузке.
        self._table: tuple[np.ndarray | None, np.ndarray | None] | None = None
        self.extend(values)

    # -- словарь ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Dimension({self.name!r}, cardinality={len(self)})"

    def __contains__(self, value: Any) -> bool:
        return self._norm(value) in self._pos

    @property
    def values(self) -> np.ndarray:
        return np.array(self._values, dtype=object if self._is_text else None)

    @property
    def _is_text(self) -> bool:
        return bool(self._values) and isinstance(self._values[0], str)

    @property
    def comparable(self) -> bool:
        """Допустимы ли диапазонные условия.

        Числовые измерения упорядочены по своей природе, поэтому требовать
        явного ordered=True для них — лишнее трение. Для категориальных
        измерений диапазон бессмыслен, и запрос должен упасть с внятной ошибкой.
        """
        return self.ordered or (
            bool(self._values) and all(isinstance(v, (int, float)) for v in self._values)
        )

    @staticmethod
    def _norm(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.str_,)):
            return str(value)
        return value

    def extend(self, values: Iterable[Any]) -> list[int]:
        """Добавляет новые значения в конец словаря; возвращает их ординалы."""
        added = []
        for v in values:
            v = self._norm(v)
            if v not in self._pos:
                self._pos[v] = len(self._values)
                self._values.append(v)
                added.append(self._pos[v])
        if added:
            self._table = None       # словарь изменился — кэш поиска устарел
            for name, arr in list(self.attributes.items()):
                pad = np.full(len(self) - len(arr), None, dtype=object)
                self.attributes[name] = np.concatenate([arr.astype(object), pad])
        return added

    def ordinal(self, value: Any) -> int:
        v = self._norm(value)
        try:
            return self._pos[v]
        except KeyError:
            raise KeyError(
                f"значение {value!r} отсутствует в измерении '{self.name}'"
            ) from None

    def label(self, ordinal: int) -> Any:
        return self._values[int(ordinal)]

    def labels(self, ordinals: Sequence[int] | None = None) -> list[Any]:
        if ordinals is None:
            return list(self._values)
        return [self._values[int(o)] for o in ordinals]

    def _search_table(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Отсортированный массив значений и их ординалы — для быстрого поиска.

        Строится один раз и сбрасывается при дозагрузке. Если значения
        разнородны и в массив не укладываются, возвращается None, и кодирование
        идёт медленным путём.
        """
        if self._table is None:
            try:
                arr = np.asarray(self._values)
            except Exception:  # pragma: no cover — разнородные значения
                self._table = (None, None)
            else:
                if arr.dtype == object or arr.ndim != 1:
                    self._table = (None, None)
                else:
                    order = np.argsort(arr, kind="stable")
                    self._table = (arr[order], order.astype(np.int64))
        return None if self._table[0] is None else self._table

    def encode(self, values: Sequence[Any]) -> np.ndarray:
        """Кодирование последовательности значений в ординалы.

        Быстрый путь — двоичный поиск по отсортированному словарю: на загрузке
        крупного факта это подавляющая часть работы, и пословный цикл на Python
        обходился на два порядка дороже самого построения гиперкуба.
        """
        table = self._search_table()
        if table is not None:
            sorted_values, ordinals = table
            arr = np.asarray(values)
            if arr.dtype != object and arr.dtype.kind == sorted_values.dtype.kind:
                pos = np.searchsorted(sorted_values, arr)
                np.clip(pos, 0, len(sorted_values) - 1, out=pos)
                bad = sorted_values[pos] != arr
                if bad.any():
                    missing = arr[bad][0]
                    raise KeyError(
                        f"значение {missing!r} отсутствует в измерении '{self.name}'"
                    )
                return ordinals[pos]
        return np.fromiter(
            (self._pos[self._norm(v)] for v in values), dtype=np.int64, count=len(values)
        )

    # -- атрибуты -----------------------------------------------------------
    def set_attribute(self, name: str, values: Sequence[Any]) -> None:
        """Атрибут значения измерения (например, region у customer)."""
        arr = np.asarray(values, dtype=object)
        if arr.size != len(self):
            raise ValueError(
                f"атрибут '{name}': {arr.size} значений при мощности {len(self)}"
            )
        self.attributes[name] = arr

    def attribute_mask(self, attr: str, predicate) -> np.ndarray:
        """0/1-маска по предикату над атрибутом."""
        if attr not in self.attributes:
            raise KeyError(f"измерение '{self.name}' не имеет атрибута '{attr}'")
        return np.array(
            [1.0 if predicate(v) else 0.0 for v in self.attributes[attr]],
            dtype=np.float32,
        )

    # -- маски --------------------------------------------------------------
    def mask_all(self) -> np.ndarray:
        return np.ones(len(self), dtype=np.float32)

    def mask_of(self, values: Iterable[Any]) -> np.ndarray:
        m = np.zeros(len(self), dtype=np.float32)
        for v in values:
            m[self.ordinal(v)] = 1.0
        return m

    def mask_range(self, low: Any = None, high: Any = None,
                   inclusive: tuple[bool, bool] = (True, True)) -> np.ndarray:
        """Диапазонная маска. Требует упорядоченного измерения."""
        if not self.comparable:
            raise ValueError(
                f"измерение '{self.name}' категориальное: диапазонные условия "
                "к нему неприменимы; используйте IN (...) или объявите измерение "
                "упорядоченным (ordered_dims при загрузке)"
            )
        vals = self._values
        m = np.zeros(len(self), dtype=np.float32)
        for i, v in enumerate(vals):
            ok = True
            if low is not None:
                ok &= (v >= low) if inclusive[0] else (v > low)
            if high is not None:
                ok &= (v <= high) if inclusive[1] else (v < high)
            m[i] = 1.0 if ok else 0.0
        return m


class Hierarchy:
    """Иерархия измерений: child -> parent (date -> month -> year).

    Хранится как разреженная матрица перехода [child × parent], благодаря чему
    ROLLUP становится обычной (0,1)-свёрткой, а не отдельным кодом агрегации.
    """

    def __init__(self, name: str, child: Dimension, parent: Dimension,
                 mapping: Sequence[Any]):
        if len(mapping) != len(child):
            raise ValueError(
                f"иерархия '{name}': {len(mapping)} отображений при мощности "
                f"дочернего измерения {len(child)}"
            )
        self.name = name
        self.child = child
        self.parent = parent
        parent.extend(mapping)
        self.child_ordinals = parent.encode(list(mapping))

    def __repr__(self) -> str:
        return f"Hierarchy({self.child.name} -> {self.parent.name})"

    def matrix(self, dtype=np.float32) -> np.ndarray:
        return rollup_matrix(self.child_ordinals, len(self.parent), dtype=dtype)
