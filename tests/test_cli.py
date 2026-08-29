# -*- coding: utf-8 -*-
"""Командный интерфейс: загрузка, запрос, EXPLAIN, сведения."""
import pytest

from amdb.api.cli import main

pd = pytest.importorskip("pandas")


@pytest.fixture
def loaded(tmp_path, frames):
    sales, products, customers = frames
    sales.to_csv(tmp_path / "sales.csv", index=False)
    products.to_csv(tmp_path / "products.csv", index=False)
    base = tmp_path / "base"
    assert main(["--db", str(base), "load", str(tmp_path / "sales.csv"),
                 "--dimensions", "customer", "product", "date",
                 "--value", "quantity", "--name", "sales"]) == 0
    assert main(["--db", str(base), "load", str(tmp_path / "products.csv"),
                 "--dimension", "--key", "product",
                 "--attribute", "category", "--measure", "price"]) == 0
    return base


def test_load_query_and_formats(loaded, capsys):
    assert main(["--db", str(loaded), "query",
                 "SELECT customer, SUM(quantity) AS q FROM sales GROUP BY customer",
                 "--format", "csv"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0] == "customer,q"
    assert len(out) == 13     # заголовок + 12 клиентов


def test_hierarchy_enables_rollup(tmp_path, loaded, capsys):
    dates = pd.DataFrame({"date": range(12), "month": [f"M{d // 3}" for d in range(12)]})
    dates.to_csv(tmp_path / "dates.csv", index=False)
    assert main(["--db", str(loaded), "hierarchy", str(tmp_path / "dates.csv"),
                 "--child", "date", "--parent", "month",
                 "--key", "date", "--column", "month"]) == 0
    assert "12 значений сворачиваются в 4" in capsys.readouterr().out
    assert main(["--db", str(loaded), "query",
                 "SELECT month, SUM(quantity) AS q FROM sales GROUP BY month",
                 "--format", "csv"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 5


def test_hierarchy_rejects_incomplete_mapping(tmp_path, loaded):
    partial = pd.DataFrame({"date": [0, 1], "month": ["M0", "M0"]})
    partial.to_csv(tmp_path / "partial.csv", index=False)
    with pytest.raises(SystemExit, match="не задан родитель"):
        main(["--db", str(loaded), "hierarchy", str(tmp_path / "partial.csv"),
              "--child", "date", "--parent", "month",
              "--key", "date", "--column", "month"])


def test_query_json(loaded, capsys):
    import json

    assert main(["--db", str(loaded), "query",
                 "SELECT SUM(quantity) AS total FROM sales", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["total"] > 0


def test_explain_shows_einsum(loaded, capsys):
    assert main(["--db", str(loaded), "explain",
                 "SELECT customer, SUM(quantity * price) FROM sales "
                 "JOIN product ON sales.product = product.product "
                 "GROUP BY customer"]) == 0
    assert "einsum(" in capsys.readouterr().out


def test_sokolov_shows_binary_products(loaded, capsys):
    assert main(["--db", str(loaded), "sokolov",
                 "SELECT customer, SUM(quantity * price) FROM sales "
                 "JOIN product ON sales.product = product.product "
                 "GROUP BY customer"]) == 0
    out = capsys.readouterr().out
    assert "-свёртка" in out and "∗" in out
    assert "μ: product" in out


def test_info_reports_layout_and_blas(loaded, capsys):
    assert main(["--db", str(loaded), "info"]) == 0
    out = capsys.readouterr().out
    assert "sales" in out and "NumPy" in out


def test_syntax_error_exit_code(loaded, capsys):
    assert main(["--db", str(loaded), "query", "SELECT FROM"]) == 2


def test_bind_error_exit_code(loaded, capsys):
    assert main(["--db", str(loaded), "query",
                 "SELECT SUM(quantity) FROM отсутствует GROUP BY customer"]) == 2


def test_missing_database(tmp_path):
    with pytest.raises(SystemExit):
        main(["--db", str(tmp_path / "нет"), "query", "SELECT SUM(q) FROM f"])


def test_doctor_runs_without_a_database():
    """Самопроверка должна работать на чистом клоне, до всякой загрузки данных."""
    from amdb.api.cli import main

    assert main(["doctor"]) == 0


def test_doctor_reports_missing_optional_packages(capsys, monkeypatch):
    """Отсутствие необязательного пакета — предупреждение с командой установки,
    а не отказ: на арендованной машине важно узнать это за секунды."""
    import amdb.api.doctor as doctor

    real = doctor.__dict__["_check_optional"]
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    real()
    out = capsys.readouterr().out
    assert "duckdb не установлен" in out
    assert "pip install" in out


def test_doctor_dispatch_threshold_is_shown(capsys):
    """Порог автовыбора должен быть виден: без него непонятно, почему на
    небольшом кубе столбцы ускорителя пусты."""
    from amdb.api.doctor import _check_dispatch

    _check_dispatch()
    out = capsys.readouterr().out
    assert "Порог автовыбора движка" in out
    assert "ускоритель" in out and "процессор" in out
