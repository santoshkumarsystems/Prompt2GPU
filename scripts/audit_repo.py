#!/usr/bin/env python3
# Author: Santosh Kumar
# Project: Prompt2GPU
# Version: 1.0

"""Run the zero-tolerance pre-push audit for Prompt2GPU.

This audit uses only Python's standard library so it can run before optional
package installation.  It checks release hygiene, source headers, dependency
separation, accidental local artifacts, documentation conventions, committed
result integrity, and presence of the already-generated dashboard.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_HEADER = [
    "# Author: Santosh Kumar",
    "# Project: Prompt2GPU",
    "# Version: 1.0",
]


def fail(message: str, failures: list[str]) -> None:
    """Record one audit failure without stopping the remaining checks."""
    failures.append(message)
    print(f"FAIL: {message}")


def check_python_headers(failures: list[str]) -> None:
    """Ensure every maintained Python source file has author/project/version metadata."""
    paths = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    for path in paths:
        text = path.read_text(errors="replace")
        for line in REQUIRED_HEADER:
            if line not in text.splitlines()[:8]:
                fail(f"missing header '{line}' in {path.relative_to(ROOT)}", failures)


def check_release_hygiene(failures: list[str]) -> None:
    """Reject commit-worthy release artifacts while allowing properly ignored local environments."""

    local_only_dirs = {
        "__pycache__",
        ".venv",
        ".venv-gpu",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }

    required_gitignore_entries = {
        ".venv/",
        ".venv-gpu/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
    }

    gitignore_path = ROOT / ".gitignore"

    if not gitignore_path.exists():
        fail("missing .gitignore", failures)
        gitignore_text = ""
    else:
        gitignore_text = gitignore_path.read_text(errors="replace")

    gitignore_lines = {
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    for entry in sorted(required_gitignore_entries):
        if entry not in gitignore_lines:
            fail(f".gitignore missing required local exclusion: {entry}", failures)

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)

        # Local environments/caches are expected to exist after setup.
        # Their contents must never be audited as project source.
        if any(part in local_only_dirs for part in relative.parts):
            continue

        if path.is_file() and path.suffix.lower() in {".zip", ".whl", ".tar", ".gz"}:
            fail(
                f"archive/package artifact should not be committed: {relative}",
                failures,
            )


def check_docs_and_commands(failures: list[str]) -> None:
    """Check public docs for ambiguous Python/venv instructions and required provenance text."""
    text_files = [ROOT / "README.md", ROOT / "RUNBOOK.md", ROOT / "METHODOLOGY.md", ROOT / "RESULTS.md"]
    text_files += list((ROOT / "docs").glob("*.md"))

    ambiguous_python = re.compile(r"(^|\s)python\s", re.MULTILINE)

    for path in text_files:
        if not path.exists():
            continue
        text = path.read_text(errors="replace")
        if ambiguous_python.search(text):
            fail(f"ambiguous 'python' command in {path.relative_to(ROOT)}; use python3", failures)

    readme = (ROOT / "README.md").read_text(errors="replace")
    for required in ["Google Colab", "Tesla T4", "docs/index.html", "make preflight-gpu", "make setup-gpu"]:
        if required not in readme:
            fail(f"README missing required release guidance: {required}", failures)


def check_dependency_separation(failures: list[str]) -> None:
    """Ensure default requirements remain lightweight and GPU setup remains opt-in."""
    default_lines = [
        line.strip().lower()
        for line in (ROOT / "requirements.txt").read_text(errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    default = "\n".join(default_lines)
    for heavy in ["torch", "vllm", "cuda", "cudnn", "nccl", "nvidia-"]:
        if heavy in default:
            fail(f"default requirements.txt unexpectedly installs heavy GPU dependency '{heavy}'", failures)

    makefile = (ROOT / "Makefile").read_text(errors="replace")
    for required in ["preflight-gpu", "setup-gpu", "setup-dashboard", "audit"]:
        if required not in makefile:
            fail(f"Makefile missing target: {required}", failures)


def check_code_paths(failures: list[str]) -> None:
    """Reject personal or Colab-specific hard-coded project paths in executable code."""
    forbidden_patterns = [
        r"/home/[^/]+/",
        r"C:\\\\Users\\\\[^\\\\]+\\\\",
        r"/content/drive/MyDrive/Prompt2GPU",
    ]
    for path in sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "scripts").glob("*.py")):
        if path.name == "audit_repo.py":
            continue
        text = path.read_text(errors="replace")
        for pattern in forbidden_patterns:
            if re.search(pattern, text):
                fail(
                    f"hard-coded personal/project path matching '{pattern}' "
                    f"in {path.relative_to(ROOT)}",
                    failures,
                )


def check_dashboard_and_results(failures: list[str]) -> None:
    """Confirm committed dashboard, plots, raw results, and recomputation verification are present."""
    required_files = [
        ROOT / "docs" / "index.html",
        ROOT / "plots" / "baseline_throughput.png",
        ROOT / "plots" / "concurrency_throughput.png",
        ROOT / "plots" / "kv_cache_usage.png",
        ROOT / "results" / "hf_baseline_raw.csv",
        ROOT / "results" / "hf_baseline_summary.json",
        ROOT / "results" / "vllm_baseline_raw.csv",
        ROOT / "results" / "vllm_baseline_summary.json",
        ROOT / "results" / "vllm_concurrency_summary.json",
        ROOT / "results" / "vllm_kv_pressure_summary.json",
    ]
    for path in required_files:
        if not path.exists() or path.stat().st_size == 0:
            fail(f"missing/empty required artifact: {path.relative_to(ROOT)}", failures)

    html = (ROOT / "docs" / "index.html").read_text(errors="replace") if (ROOT / "docs" / "index.html").exists() else ""
    for marker in ["Prompt2GPU V1.0", "Tesla T4", "66.71", "1645.93", "89.3%"]:
        if marker not in html:
            fail(f"committed dashboard missing expected published marker: {marker}", failures)

    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_results.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        fail("verify_results.py failed: " + (completed.stderr or completed.stdout).strip(), failures)
    else:
        print(completed.stdout.strip())


def check_obvious_secrets(failures: list[str]) -> None:
    """Scan maintained text/code for obvious credential material before a public push."""
    patterns = [
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[=:]\s*['\"][^'\"]{8,}"),
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
    ]
    roots = [ROOT / "src", ROOT / "scripts", ROOT / "docs"]
    files = [ROOT / "README.md", ROOT / "RUNBOOK.md", ROOT / "METHODOLOGY.md", ROOT / "RESULTS.md"]
    for directory in roots:
        files.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".py", ".md", ".html", ".txt", ".json"})
    for path in files:
        if not path.exists():
            continue
        if path == ROOT / "docs" / "index.html":
            # Plotly bundles contain generic strings such as api_key/accessToken;
            # they are library identifiers, not repository credentials.
            continue
        text = path.read_text(errors="ignore")
        for pattern in patterns:
            if pattern.search(text):
                fail(f"possible secret/credential in {path.relative_to(ROOT)}", failures)
                break


def main() -> int:
    """Execute all release checks and return nonzero if any check fails."""
    print("=== Prompt2GPU Zero-Tolerance Pre-Push Audit ===")
    failures: list[str] = []

    check_python_headers(failures)
    check_release_hygiene(failures)
    check_docs_and_commands(failures)
    check_dependency_separation(failures)
    check_code_paths(failures)
    check_dashboard_and_results(failures)
    check_obvious_secrets(failures)

    print()
    if failures:
        print(f"AUDIT FAILED: {len(failures)} issue(s) must be resolved before git push.")
        return 1

    print("AUDIT PASS: repository is ready for human review before git push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
