# -*- coding: utf-8 -*-
"""REST API поверх AMDB (FastAPI — необязательная зависимость).

    uvicorn amdb.api.rest:app --factory  # или create_app(db)
"""
from __future__ import annotations

import os
from typing import Any

from ..database import Database
from ..ql.binder import BindError
from ..ql.lexer import QuerySyntaxError
from ..security.rls import AccessDenied


def create_app(db: Database | None = None):
    """Создаёт FastAPI-приложение поверх открытой базы."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from pydantic import BaseModel
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "REST API требует fastapi и pydantic: pip install 'amdb[api]'"
        ) from e

    if db is None:
        path = os.environ.get("AMDB_PATH")
        if not path:
            raise RuntimeError("укажите базу через AMDB_PATH или аргумент create_app(db)")
        db = Database.open(path)

    app = FastAPI(title="AMDB", description="Алгебраическая машина баз данных",
                  version="0.1.0")

    class Query(BaseModel):
        sql: str
        role: str | None = None
        limit: int | None = None

    def _role(x_amdb_role: str | None = Header(default=None)) -> str | None:
        """Роль берётся из заголовка, а не из тела запроса: тело подделать проще."""
        return x_amdb_role

    @app.get("/catalog")
    def catalog() -> dict[str, Any]:
        return {
            "dimensions": {
                name: {"cardinality": len(d), "ordered": d.ordered,
                       "attributes": sorted(d.attributes)}
                for name, d in db.dimensions.items()
            },
            "cubes": {name: db.stats(name) for name in db.cubes},
            "hierarchies": [
                {"child": h.child.name, "parent": h.parent.name, "name": h.name}
                for h in db.catalog.hierarchies
            ],
        }

    @app.get("/cube/{name}")
    def cube(name: str) -> dict[str, Any]:
        try:
            return db.stats(name)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.post("/query")
    def query(q: Query, role: str | None = Depends(_role)) -> dict[str, Any]:
        try:
            result = db.sql(q.sql, role=role or q.role)
        except AccessDenied as e:
            raise HTTPException(403, str(e))
        except (QuerySyntaxError, BindError) as e:
            raise HTTPException(400, str(e))
        except KeyError as e:
            raise HTTPException(404, str(e))
        rows = result.rows[: q.limit] if q.limit else result.rows
        return {"columns": result.columns,
                "rows": [list(r) for r in rows],
                "row_count": len(result.rows),
                "stats": {k: v for k, v in result.stats.items() if k != "plan_cache"}}

    @app.post("/explain")
    def explain(q: Query, role: str | None = Depends(_role)) -> dict[str, str]:
        try:
            return {"plan": db.explain(q.sql, role=role or q.role)}
        except (QuerySyntaxError, BindError) as e:
            raise HTTPException(400, str(e))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "cubes": len(db.cubes), "engine": db.engine.name}

    return app


app = create_app  # для uvicorn --factory
