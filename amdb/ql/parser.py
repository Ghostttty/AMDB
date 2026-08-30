# -*- coding: utf-8 -*-
"""Парсер запросов AMDB: рекурсивный спуск, без внешних зависимостей.

Поддерживается подмножество SQL, покрывающее ТЗ 3.2.2: SELECT/JOIN/WHERE/
GROUP BY/HAVING/ORDER BY/LIMIT, вложенные запросы в IN и оконные функции.
"""
from __future__ import annotations

from typing import Any

from .ast import (
    Aggregate,
    Between,
    BinOp,
    Case,
    Column,
    Compare,
    Condition,
    Expr,
    InList,
    IsNull,
    InSubquery,
    Join,
    Literal,
    Logical,
    Not,
    OrderKey,
    Select,
    SelectItem,
    Star,
    WindowSpec,
)
from .lexer import AGGREGATES, QuerySyntaxError, Token, tokenize


class Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0

    # -- служебное ----------------------------------------------------------
    @property
    def cur(self) -> Token:
        return self.tokens[self.i]

    def at(self, kind: str, value: Any = None) -> bool:
        t = self.cur
        return t.kind == kind and (value is None or t.value == value)

    def at_keyword(self, *words: str) -> bool:
        return self.cur.kind == "KEYWORD" and self.cur.value in words

    def take(self) -> Token:
        t = self.cur
        self.i += 1
        return t

    def expect(self, kind: str, value: Any = None) -> Token:
        if not self.at(kind, value):
            want = value or kind
            raise QuerySyntaxError(
                f"ожидалось {want!r}, получено {self.cur.value!r}", self.text, self.cur.pos
            )
        return self.take()

    def accept(self, kind: str, value: Any = None) -> Token | None:
        return self.take() if self.at(kind, value) else None

    # -- запрос -------------------------------------------------------------
    def parse(self) -> Select:
        q = self.parse_select()
        if self.cur.kind != "EOF":
            if self.at("OP", ";"):
                self.take()
            if self.cur.kind != "EOF":
                raise QuerySyntaxError(
                    f"лишние символы после запроса: {self.cur.value!r}",
                    self.text, self.cur.pos)
        return q

    def parse_select(self) -> Select:
        self.expect("KEYWORD", "SELECT")
        q = Select()
        # SELECT DISTINCT a, b — то же, что GROUP BY a, b без агрегатов:
        # различные сочетания значений измерений и есть непустые группы.
        distinct = bool(self.accept("KEYWORD", "DISTINCT"))
        q.items = self.parse_select_list()
        if self.accept("KEYWORD", "FROM"):
            q.source = self.parse_name()
            while self.at_keyword("JOIN", "INNER", "LEFT"):
                if self.at_keyword("INNER", "LEFT"):
                    self.take()
                self.expect("KEYWORD", "JOIN")
                src = self.parse_name()
                on = self.parse_condition() if self.accept("KEYWORD", "ON") else None
                q.joins.append(Join(src, on))
        if self.accept("KEYWORD", "WHERE"):
            q.where = self.parse_condition()
        if self.at_keyword("GROUP"):
            self.take()
            self.expect("KEYWORD", "BY")
            cols = [str(self.parse_column())]
            while self.accept("OP", ","):
                cols.append(str(self.parse_column()))
            q.group_by = tuple(cols)
        if self.accept("KEYWORD", "HAVING"):
            q.having = self.parse_condition()
        if self.at_keyword("ORDER"):
            self.take()
            self.expect("KEYWORD", "BY")
            keys = [self.parse_order_key()]
            while self.accept("OP", ","):
                keys.append(self.parse_order_key())
            q.order_by = tuple(keys)
        if self.accept("KEYWORD", "LIMIT"):
            q.limit = int(self.expect("NUMBER").value)
        if distinct:
            if q.group_by:
                raise QuerySyntaxError(
                    "DISTINCT вместе с GROUP BY не поддержан: это одно и то же",
                    self.text, self.cur.pos)
            if q.aggregates():
                raise QuerySyntaxError(
                    "DISTINCT с агрегатом не поддержан", self.text, self.cur.pos)
            q.group_by = tuple(str(c) for c in q.plain_columns())
        return q

    def parse_order_key(self) -> OrderKey:
        if self.at("NUMBER"):
            key: Any = int(self.take().value)
        else:
            key = str(self.parse_column())
        desc = False
        if self.at_keyword("ASC", "DESC"):
            desc = self.take().value == "DESC"
        return OrderKey(key, desc)

    def parse_select_list(self) -> list[SelectItem]:
        items = [self.parse_select_item()]
        while self.accept("OP", ","):
            items.append(self.parse_select_item())
        return items

    def parse_select_item(self) -> SelectItem:
        if self.at("OP", "*"):
            self.take()
            return SelectItem(Star())
        expr = self.parse_expr()
        alias = None
        if self.accept("KEYWORD", "AS"):
            alias = self.parse_name()
        elif self.at("NAME"):
            alias = self.parse_name()
        return SelectItem(expr, alias)

    def parse_name(self) -> str:
        """Имя столбца или источника.

        Ключевые слова именами быть не могут: иначе `SELECT FROM f` разобрался бы
        как выборка столбца с именем from. Имя, совпадающее с ключевым словом,
        экранируется двойными кавычками.
        """
        t = self.cur
        if t.kind == "NAME":
            return str(self.take().value)
        raise QuerySyntaxError(f"ожидалось имя, получено {t.value!r}", self.text, t.pos)

    def parse_column(self) -> Column:
        first = self.parse_name()
        if self.at("OP", "."):
            self.take()
            return Column(self.parse_name(), first)
        return Column(first)

    # -- выражения ----------------------------------------------------------
    def parse_expr(self) -> Expr:
        return self.parse_additive()

    def parse_additive(self) -> Expr:
        left = self.parse_multiplicative()
        while self.at("OP", "+") or self.at("OP", "-"):
            op = self.take().value
            left = BinOp(op, left, self.parse_multiplicative())
        return left

    def parse_multiplicative(self) -> Expr:
        left = self.parse_unary()
        while self.at("OP", "*") or self.at("OP", "/"):
            op = self.take().value
            left = BinOp(op, left, self.parse_unary())
        return left

    def parse_unary(self) -> Expr:
        if self.at("OP", "-"):
            self.take()
            return BinOp("-", Literal(0), self.parse_unary())
        if self.at("OP", "+"):
            self.take()
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        t = self.cur
        if t.kind == "NUMBER":
            return Literal(self.take().value)
        if t.kind == "STRING":
            return Literal(self.take().value)
        if t.kind == "OP" and t.value == "(":
            self.take()
            e = self.parse_expr()
            self.expect("OP", ")")
            return e
        if t.kind == "KEYWORD" and t.value in AGGREGATES:
            return self.parse_aggregate()
        if t.kind == "KEYWORD" and t.value == "CASE":
            return self.parse_case()
        if t.kind == "NAME":
            return self.parse_column()
        raise QuerySyntaxError(f"неожиданный токен {t.value!r}", self.text, t.pos)

    def parse_case(self) -> Case:
        """CASE WHEN условие THEN выражение [WHEN ...] [ELSE выражение] END."""
        self.expect("KEYWORD", "CASE")
        branches: list[tuple[Condition, Expr]] = []
        while self.at_keyword("WHEN"):
            self.take()
            condition = self.parse_condition()
            self.expect("KEYWORD", "THEN")
            branches.append((condition, self.parse_expr()))
        if not branches:
            raise QuerySyntaxError("в CASE нет ни одной ветви WHEN", self.text,
                                   self.cur.pos)
        otherwise = None
        if self.at_keyword("ELSE"):
            self.take()
            otherwise = self.parse_expr()
        self.expect("KEYWORD", "END")
        return Case(branches, otherwise)

    def parse_aggregate(self) -> Aggregate:
        func = str(self.take().value)
        self.expect("OP", "(")
        distinct = bool(self.accept("KEYWORD", "DISTINCT"))
        if self.at("OP", "*"):
            self.take()
            arg: Expr = Star()
        else:
            arg = self.parse_expr()
        self.expect("OP", ")")
        window = self.parse_window() if self.at_keyword("OVER") else None
        return Aggregate(func, arg, distinct, window)

    def parse_window(self) -> WindowSpec:
        self.expect("KEYWORD", "OVER")
        self.expect("OP", "(")
        spec = WindowSpec()
        if self.at_keyword("PARTITION"):
            self.take()
            self.expect("KEYWORD", "BY")
            cols = [str(self.parse_column())]
            while self.accept("OP", ","):
                cols.append(str(self.parse_column()))
            spec.partition_by = tuple(cols)
        if self.at_keyword("ORDER"):
            self.take()
            self.expect("KEYWORD", "BY")
            spec.order_by = str(self.parse_column())
            if self.at_keyword("ASC", "DESC"):
                self.take()
        if self.at_keyword("ROWS", "RANGE"):
            self.take()
            spec.frame_preceding = self.parse_frame()
        self.expect("OP", ")")
        return spec

    def parse_frame(self) -> int | None:
        """BETWEEN {UNBOUNDED | n} PRECEDING AND CURRENT ROW -> ширина окна."""
        self.expect("KEYWORD", "BETWEEN")
        if self.at_keyword("UNBOUNDED"):
            self.take()
            width = None
        else:
            width = int(self.expect("NUMBER").value) + 1
        self.expect("KEYWORD", "PRECEDING")
        self.expect("KEYWORD", "AND")
        if self.at_keyword("CURRENT"):
            self.take()
            self.expect("KEYWORD", "ROW")
        else:
            raise QuerySyntaxError(
                "поддерживается только рамка ... AND CURRENT ROW",
                self.text, self.cur.pos)
        return width

    # -- условия ------------------------------------------------------------
    def parse_condition(self) -> Condition:
        return self.parse_or()

    def parse_or(self) -> Condition:
        left = self.parse_and()
        while self.at_keyword("OR"):
            self.take()
            left = Logical("OR", left, self.parse_and())
        return left

    def parse_and(self) -> Condition:
        left = self.parse_not()
        while self.at_keyword("AND"):
            self.take()
            left = Logical("AND", left, self.parse_not())
        return left

    def parse_not(self) -> Condition:
        if self.at_keyword("NOT"):
            self.take()
            return Not(self.parse_not())
        return self.parse_predicate()

    def parse_predicate(self) -> Condition:
        if self.at("OP", "("):
            save = self.i
            self.take()
            try:
                inner = self.parse_condition()
                self.expect("OP", ")")
                return inner
            except QuerySyntaxError:
                self.i = save
        col = self.parse_column()
        if self.at_keyword("IS"):
            self.take()
            is_negated = bool(self.accept("KEYWORD", "NOT"))
            self.expect("KEYWORD", "NULL")
            return IsNull(col, is_negated)
        negated = bool(self.accept("KEYWORD", "NOT"))
        if self.at_keyword("IN"):
            self.take()
            self.expect("OP", "(")
            if self.at_keyword("SELECT"):
                sub = self.parse_select()
                self.expect("OP", ")")
                return InSubquery(col, sub, negated)
            values = [self.parse_value()]
            while self.accept("OP", ","):
                values.append(self.parse_value())
            self.expect("OP", ")")
            return InList(col, tuple(values), negated)
        if self.at_keyword("BETWEEN"):
            self.take()
            low = self.parse_value()
            self.expect("KEYWORD", "AND")
            high = self.parse_value()
            return Between(col, low, high, negated)
        if negated:
            raise QuerySyntaxError("после NOT ожидалось IN или BETWEEN",
                                   self.text, self.cur.pos)
        op = self.expect("OP").value
        if op not in ("=", "!=", "<", "<=", ">", ">="):
            raise QuerySyntaxError(f"недопустимый оператор сравнения {op!r}",
                                   self.text, self.cur.pos)
        return Compare(op, col, self.parse_value())

    def parse_value(self) -> Any:
        t = self.cur
        if t.kind in ("NUMBER", "STRING"):
            return self.take().value
        if t.kind == "NAME":
            # Квалифицированное имя — ссылка на столбец (условие соединения),
            # простое имя — константа-значение измерения.
            if self.tokens[self.i + 1].kind == "OP" and self.tokens[self.i + 1].value == ".":
                return self.parse_column()
            return self.take().value
        raise QuerySyntaxError(f"ожидалась константа, получено {t.value!r}",
                               self.text, t.pos)


def parse(text: str) -> Select:
    """Разбирает текст запроса в AST."""
    return Parser(text).parse()
