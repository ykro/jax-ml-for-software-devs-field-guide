#!/usr/bin/env python3
"""Checks the JAX & ML for Software Devs Field Guide and the unit repos cloned next to it.

The index must have its three diagrams (HTML, SVG, PNG). Each unit cloned next to this repo
must have its judge (verify_*.py), its benchmark report (benchmarks/latest_results.json and
terminal_output.txt) and three diagrams in each format. Units you have not cloned are
reported as skipped and do not count as a failure.

    git clone https://github.com/ykro/jax-for-software-devs.git ../jax-for-software-devs
    python3 scripts/verify_field_guide.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
INDEX_SLUG = "jax-ml-for-software-devs-field-guide"
UNITS: list[tuple[str, str, str]] = [
    ("1", "jax-for-software-devs", "verify_fundamentals.py"),
    ("2", "ml-from-first-principles-in-jax", "verify_ml_principles.py"),
    ("3", "embeddings-and-semantic-search-in-jax", "verify_embeddings.py"),
    ("4", "foundational-models-anatomy-in-jax", "verify_anatomy.py"),
    ("5", "adapting-gemma-for-your-app-in-jax", "verify_adaptation.py"),
]
FULL_SET = {"html": 3, "svg": 3, "png": 3}


def count_diagrams(repo: pathlib.Path) -> dict[str, int]:
    folder = repo / "diagrams"
    return {ext: len(list(folder.glob(f"*.{ext}"))) for ext in FULL_SET}


def check_unit(repo: pathlib.Path, judge: str) -> tuple[list[str], dict[str, object]]:
    """Returns what is missing in a cloned unit, plus the details saved in the summary."""
    results = repo / "benchmarks" / "latest_results.json"
    diagrams = count_diagrams(repo)
    missing = []
    if not (repo / judge).exists():
        missing.append(judge)
    if not results.exists():
        missing.append("benchmarks/latest_results.json")
    if not (repo / "benchmarks" / "terminal_output.txt").exists():
        missing.append("benchmarks/terminal_output.txt")
    if diagrams != FULL_SET:
        missing.append("diagrams")
    status: dict[str, object] = {"judge_script": judge, "diagrams": diagrams}
    if results.exists():
        try:
            status["metrics"] = json.loads(results.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            missing.append("valid latest_results.json")
            status["error"] = str(exc)
    return missing, status


def main() -> int:
    index_diagrams = count_diagrams(ROOT)
    index_ok = index_diagrams == FULL_SET
    rows: list[tuple[str, str, str, str | None]] = []
    projects: dict[str, object] = {}
    for code, slug, judge in UNITS:
        repo = PARENT / slug
        if not repo.exists():
            rows.append((code, slug, "skipped, not cloned", None))
            projects[slug] = {"cloned": False}
            continue
        missing, status = check_unit(repo, judge)
        verdict = "PASS" if not missing else "FAIL"
        detail = "judge, report, 3 diagrams" if not missing else "missing " + ", ".join(missing)
        rows.append((code, slug, detail, verdict))
        projects[slug] = {"cloned": True, "passed": not missing, **status}

    checked = [r for r in rows if r[3]]
    all_passed = index_ok and bool(checked) and all(r[3] == "PASS" for r in checked)

    bar = "=" * 98
    index_row = ("0", INDEX_SLUG, "3 diagrams" if index_ok else "missing diagrams",
                 "PASS" if index_ok else "FAIL")
    lines = [bar, "JAX & ML for Software Devs Field Guide · repo check (CPU, no GPU, no network)",
             bar]
    for code, slug, detail, verdict in [index_row, *rows]:
        lines.append(f"{code:3} | {slug:42} | {detail:34} | {verdict or '-'}")
    lines.append("-" * 98)
    lines.append(f"{len(checked)} checked, {len(rows) - len(checked)} skipped.")
    if not checked:
        first = UNITS[0][1]
        lines.append("No unit of this path is cloned next to this repo, so there is nothing to check.")
        lines.append("Clone at least the first one:")
        lines.append(f"  git clone https://github.com/ykro/{first}.git ../{first}")
    elif all_passed:
        lines.append("ALL CHECKED UNITS PASSED.")
    else:
        lines.append("SOME CHECKS FAILED.")
    report = "\n".join(lines) + "\n"

    summary = {
        "index_repo": {"slug": INDEX_SLUG, "diagrams": index_diagrams, "passed": index_ok},
        "checked": len(checked),
        "skipped": len(rows) - len(checked),
        "projects": projects,
        "all_passed": all_passed,
    }
    bench = ROOT / "benchmarks"
    bench.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (bench / "field_guide_summary.json").write_text(payload, encoding="utf-8")
    (bench / "latest_results.json").write_text(payload, encoding="utf-8")
    (bench / "terminal_output.txt").write_text(report, encoding="utf-8")

    print(report, end="")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
