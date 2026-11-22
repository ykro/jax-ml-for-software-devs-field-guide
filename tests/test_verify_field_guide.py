"""The Field Guide check skips the units that are not cloned and says what to clone."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def index_copy(tmp_path: Path) -> Path:
    repo = tmp_path / "jax-ml-for-software-devs-field-guide"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "verify_field_guide.py", repo / "scripts")
    shutil.copytree(ROOT / "diagrams", repo / "diagrams")
    return repo


def verify(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "scripts/verify_field_guide.py"], cwd=repo,
                          capture_output=True, text=True, timeout=60, check=False)


def fake_unit(parent: Path, slug: str, judge: str) -> None:
    unit = parent / slug
    (unit / "benchmarks").mkdir(parents=True)
    (unit / "diagrams").mkdir()
    (unit / judge).write_text("")
    (unit / "benchmarks" / "latest_results.json").write_text("{}")
    (unit / "benchmarks" / "terminal_output.txt").write_text("")
    for i in range(3):
        for ext in ("html", "svg", "png"):
            (unit / "diagrams" / f"{i}.{ext}").write_text("")


def test_nothing_cloned_says_what_to_clone(tmp_path):
    run = verify(index_copy(tmp_path))
    assert run.returncode == 1
    assert "0 checked, 5 skipped." in run.stdout
    assert run.stdout.count("skipped, not cloned") == 5
    assert "SOME CHECKS FAILED." not in run.stdout
    assert ("git clone https://github.com/ykro/jax-for-software-devs.git "
            "../jax-for-software-devs") in run.stdout


def test_one_unit_cloned_passes_and_the_rest_are_skipped(tmp_path):
    repo = index_copy(tmp_path)
    fake_unit(tmp_path, "jax-for-software-devs", "verify_fundamentals.py")
    run = verify(repo)
    assert run.returncode == 0, run.stdout
    assert "1 checked, 4 skipped." in run.stdout
    assert "ALL CHECKED UNITS PASSED." in run.stdout


def test_a_cloned_unit_without_its_report_fails(tmp_path):
    repo = index_copy(tmp_path)
    fake_unit(tmp_path, "jax-for-software-devs", "verify_fundamentals.py")
    (tmp_path / "jax-for-software-devs" / "benchmarks" / "latest_results.json").unlink()
    run = verify(repo)
    assert run.returncode == 1
    assert "missing benchmarks/latest_results.json" in run.stdout
    assert "SOME CHECKS FAILED." in run.stdout
