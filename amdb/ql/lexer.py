# -*- coding: utf-8 -*-
"""Лексический анализатор языка запросов AMDB.

Идентификаторы допускают кириллицу: имена измерений и мер в прикладных схемах
чаще русские, и заставлять транслитерировать их — лишнее трение.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KEYWORDS = {
    "SELECT", "FROM", "JOIN", "INNER", "LEFT", "ON", "WHERE", "GROUP", "BY",
    "HAVING", "ORDER", "LIMIT", "AS", "AND", "OR", "NOT", "IN", "BETWEEN",
    "ASC", "DESC", "DISTINCT", "OVER", "PARTITION", "ROWS", "RANGE", "BETWEEN",
    "PRECEDING", "CURRENT", "ROW", "UNBOUNDED", "FOLLOWING", "NULL", "IS",
    "CASE", "WHEN", "THEN", "ELSE", "END",
}

AGGREGATES = {"SUM", "AVG", "COUNT", "MIN", "MAX"}

_TWO_CHAR = {"<=", ">=", "<>", "!="}
_ONE_CHAR = set("=<>+-*/(),.;")


class QuerySyntaxError(SyntaxError):
    """Ошибка разбора запроса с указанием позиции."""

    def __init__(self, message: str, text: str, pos: int):
        line = text.count("\n", 0, pos) + 1
        col = pos - (text.rfind("\n", 0, pos) + 1) + 1
        snippet = text.splitlines()[line - 1] if text.splitlines() else text
        super().__init__(
            f"{message} (строка {line}, позиция {col})\n  {snippet}\n  {' ' * (col - 1)}^"
        )
        self.position = pos


@dataclass
class Token:
    kind: str          # KEYWORD | NAME | NUMBER | STRING | OP | EOF
    value: Any
    pos: int

    def __str__(self) -> str:
        return f"{self.kind}({self.value!r})"


def _is_name_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_name_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "-" and text.startswith("--", i):          # комментарий до конца строки
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if _is_name_start(ch):
            j = i + 1
            while j < n and _is_name_char(text[j]):
                j += 1
            word = text[i:j]
            upper = word.upper()
            kind = "KEYWORD" if upper in KEYWORDS or upper in AGGREGATES else "NAME"
            tokens.append(Token(kind, upper if kind == "KEYWORD" else word, i))
            i = j
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            seen_dot = False
            while j < n and (text[j].isdigit() or (text[j] == "." and not seen_dot)):
                seen_dot |= text[j] == "."
                j += 1
            raw = text[i:j]
            tokens.append(Token("NUMBER", float(raw) if seen_dot else int(raw), i))
            i = j
            continue
        if ch in "'\"":
            quote = ch
            j = i + 1
            buf = []
            while j < n:
                if text[j] == quote:
                    if j + 1 < n and text[j + 1] == quote:   # экранирование удвоением
                        buf.append(quote)
                        j += 2
                        continue
                    break
                buf.append(text[j])
                j += 1
            if j >= n:
                raise QuerySyntaxError("незакрытая строковая константа", text, i)
            kind = "STRING" if quote == "'" else "NAME"
            tokens.append(Token(kind, "".join(buf), i))
            i = j + 1
            continue
        if text[i:i + 2] in _TWO_CHAR:
            op = text[i:i + 2]
            tokens.append(Token("OP", "!=" if op in ("<>", "!=") else op, i))
            i += 2
            continue
        if ch in _ONE_CHAR:
            tokens.append(Token("OP", ch, i))
            i += 1
            continue
        raise QuerySyntaxError(f"недопустимый символ {ch!r}", text, i)
    tokens.append(Token("EOF", None, n))
    return tokens
