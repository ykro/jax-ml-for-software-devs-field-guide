#!/usr/bin/env python3
"""Validador maestro de la Serie B: De Software Dev a ML y Foundational Models con JAX.

Consolida las métricas deterministas (Outside Judges) de los 5 repositorios prácticos
de la Serie B (`B1` a `B5`), verifica la integridad de los 18 diagramas (HTML, SVG, PNG)
en los 6 repositorios de la Serie B (`B0` + `B1..B5`) y audita que todos los reportes
de benchmark estén completos.
"""

from __future__ import annotations

import json
import pathlib
import sys

SERIES_ROOT = pathlib.Path(__file__).resolve().parents[2]
INDEX_SLUG = "jax-ml-for-software-devs-field-guide"
SERIE_B_PROJECTS: list[tuple[str, str]] = [
    ("jax-for-software-devs", "verify_fundamentals.py"),
    ("ml-from-first-principles-in-jax", "verify_first_principles.py"),
    ("embeddings-and-semantic-search-in-jax", "verify_embeddings.py"),
    ("foundational-models-anatomy-in-jax", "verify_anatomy.py"),
    ("adapting-gemma-for-your-app-in-jax", "verify_adaptation.py"),
]


def _count_diagrams(repo_dir: pathlib.Path) -> dict[str, int]:
    diag_dir = repo_dir / "diagrams"
    if not diag_dir.exists():
        return {"html": 0, "svg": 0, "png": 0}
    return {
        "html": len(list(diag_dir.glob("*.html"))),
        "svg": len(list(diag_dir.glob("*.svg"))),
        "png": len(list(diag_dir.glob("*.png"))),
    }


def _find_judge_script(repo_dir: pathlib.Path, preferred: str) -> str:
    if (repo_dir / preferred).exists():
        return preferred
    candidates = sorted(repo_dir.glob("verify_*.py"))
    if candidates:
        return candidates[0].name
    return preferred


def main() -> int:
    index_dir = pathlib.Path(__file__).resolve().parents[1]
    index_diagrams = _count_diagrams(index_dir)
    all_passed = (
        index_diagrams["html"] == 3
        and index_diagrams["svg"] == 3
        and index_diagrams["png"] == 3
    )

    projects_status: dict[str, object] = {}
    total_html = index_diagrams["html"]
    total_svg = index_diagrams["svg"]
    total_png = index_diagrams["png"]

    lines = [
        "======================================================================",
        "OUTSIDE JUDGE: Serie B Field Guide — Master Series Verification",
        "======================================================================",
        f"[Repo B0] {INDEX_SLUG}: diagrams={index_diagrams}",
    ]

    for idx, (slug, default_judge) in enumerate(SERIE_B_PROJECTS, start=1):
        repo_dir = SERIES_ROOT / slug
        judge_script = _find_judge_script(repo_dir, default_judge)
        results_file = repo_dir / "benchmarks" / "latest_results.json"
        terminal_file = repo_dir / "benchmarks" / "terminal_output.txt"
        diag_counts = _count_diagrams(repo_dir)
        total_html += diag_counts["html"]
        total_svg += diag_counts["svg"]
        total_png += diag_counts["png"]

        repo_ok = (
            repo_dir.exists()
            and (repo_dir / judge_script).exists()
            and results_file.exists()
            and terminal_file.exists()
            and diag_counts["html"] == 3
            and diag_counts["svg"] == 3
            and diag_counts["png"] == 3
        )
        status: dict[str, object] = {
            "exists": repo_dir.exists(),
            "judge_script": judge_script,
            "judge_script_exists": (repo_dir / judge_script).exists(),
            "results_file": results_file.exists(),
            "terminal_output_file": terminal_file.exists(),
            "diagrams": diag_counts,
        }
        if results_file.exists():
            try:
                metrics = json.loads(results_file.read_text(encoding="utf-8"))
                status["metrics"] = metrics
            except Exception as exc:  # noqa: BLE001
                status["error"] = str(exc)
                repo_ok = False
        else:
            repo_ok = False

        status["passed"] = repo_ok
        if not repo_ok:
            all_passed = False
        projects_status[slug] = status
        lines.append(
            f"[Repo B{idx}] {slug}: passed={repo_ok} | diagrams={diag_counts} | "
            f"results={results_file.exists()} | terminal_log={terminal_file.exists()}"
        )

    expected_total = 3 * (1 + len(SERIE_B_PROJECTS))
    if total_html != expected_total or total_svg != expected_total or total_png != expected_total:
        all_passed = False

    summary: dict[str, object] = {
        "index_repo": {
            "slug": INDEX_SLUG,
            "diagrams": index_diagrams,
        },
        "total_diagrams": {
            "html": total_html,
            "svg": total_svg,
            "png": total_png,
            "expected": expected_total,
        },
        "projects": projects_status,
        "all_passed": all_passed,
    }

    lines.append("----------------------------------------------------------------------")
    lines.append(
        f"OVERALL SERIE B STATUS: {'PASS' if all_passed else 'PARTIAL'} "
        f"(html={total_html}/{expected_total}, svg={total_svg}/{expected_total}, "
        f"png={total_png}/{expected_total})"
    )
    lines.append("======================================================================")
    terminal_report = "\n".join(lines) + "\n"

    bench_dir = index_dir / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)
    json_payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    (bench_dir / "serie_b_summary.json").write_text(json_payload, encoding="utf-8")
    (bench_dir / "latest_results.json").write_text(json_payload, encoding="utf-8")
    (bench_dir / "terminal_output.txt").write_text(terminal_report, encoding="utf-8")

    print(terminal_report, end="")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
