# -*- coding: utf-8 -*-
"""Скрипты примеров и бенчмарков должны работать из копии репозитория.

Регрессионный тест на конкретную ошибку: скрипты импортировали `amdb`, не
добавляя корень репозитория в путь поиска модулей, и падали с
ModuleNotFoundError у всякого, кто не установил пакет через pip. При запуске
с переменной PYTHONPATH ошибка не проявлялась, поэтому здесь окружение
намеренно очищается.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Запускает скрипт в отдельном процессе без PYTHONPATH и без текущего каталога."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    # -I изолирует интерпретатор: не добавляет каталог скрипта и игнорирует
    # пользовательские настройки. Скрипт обязан позаботиться о пути сам.
    return subprocess.run(
        [sys.executable, str(ROOT / script), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(cwd or Path.cwd()), timeout=600,
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    proc = run("examples/generate.py", "--out", str(out), "--customers", "20",
               "--products", "20", "--days", "30", "--rows", "5000")
    assert proc.returncode == 0, proc.stderr
    assert (out / "sales.csv").exists()
    return out


def test_generate_runs_standalone(dataset):
    assert (dataset / "products.csv").exists()
    assert (dataset / "dates.csv").exists()


def test_demo_runs_standalone(dataset):
    """Главный случай: `python examples/demo.py` без установки пакета."""
    proc = run("examples/demo.py", "--data", str(dataset))
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "ModuleNotFoundError" not in proc.stderr
    assert "einsum" in proc.stdout


@pytest.mark.parametrize("script,args", [
    ("bench/bench_convolve.py", ["--only", "b1", "--repeat", "1"]),
    ("bench/bench_gpu.py", []),
    ("bench/bench_cardinality.py", ["--rows", "50000", "--cards", "50,200",
                                    "--products", "10", "--months", "4"]),
])
def test_bench_scripts_run_standalone(script, args):
    proc = run(script, *args)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "ModuleNotFoundError" not in proc.stderr


def test_bench_olap_runs_standalone():
    proc = run("bench/bench_olap.py", "--rows", "20000", "--side", "30",
               "--repeat", "1", "--only", "b3")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "einsum" in proc.stdout


def test_cli_runs_as_module():
    """`python -m amdb signature` не требует ни установки, ни PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "-m", "amdb", "signature"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "E = (S, T; Ω)" in proc.stdout


def test_appendix_example_reproduces_plan_from_article():
    """Приложение к статье обещает воспроизводимость — она и проверяется.

    Сверяется не только запуск, но и сама einsum-спецификация: приложение
    приводит её дословно, и расхождение означало бы, что напечатанное в статье
    больше не соответствует системе.
    """
    proc = run("examples/appendix.py")
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "einsum('abc,b,cd,b->ad')" in proc.stdout
    assert "(0,2)-свёртка" in proc.stdout
    assert "строк в результате: 1200" in proc.stdout
