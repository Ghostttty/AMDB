# -*- coding: utf-8 -*-
"""AMDB — алгебраическая машина баз данных.

СУБД на алгебре многомерных матриц Н.П. Соколова: SQL-подобные запросы
транслируются в цепочки (λ, μ)-свёрнутых произведений и исполняются одним
вызовом ``einsum`` поверх оптимизированного BLAS.

    >>> from amdb import Database
    >>> db = Database()
    >>> db.load_frame(sales, ["customer", "product", "date"], "quantity", "sales")
    >>> db.sql("SELECT customer, SUM(quantity) FROM sales GROUP BY customer")
"""
from .core.convolve import build_einsum, convolve
from .core.mdm import MultidimensionalMatrix, convolve_named
from .core.sparse import COOCube
from .database import Database
from .exec.result import ResultSet
from .storage.catalog import Catalog, Cube
from .storage.dimension import Dimension, Hierarchy

__version__ = "0.1.0"

__all__ = [
    "COOCube",
    "Catalog",
    "Cube",
    "Database",
    "Dimension",
    "Hierarchy",
    "MultidimensionalMatrix",
    "ResultSet",
    "__version__",
    "build_einsum",
    "convolve",
    "convolve_named",
]
