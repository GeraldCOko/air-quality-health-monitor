"""Developer commands: ``format`` rewrites the code into the house style, ``verify`` checks it without changing it."""

import subprocess
import sys

_PYTHON_TARGETS: list[str] = ["aqmonitor", "scripts", "tests", "app.py"]
_MARKDOWN_TARGETS: list[str] = ["README.md", ".claude/CLAUDE.md"]


def _run(*command: str) -> int:
    print("$", " ".join(command), flush=True)
    return subprocess.run([sys.executable, "-m", *command], check=False).returncode


def format_code() -> None:
    """Sort imports, reformat Python with black and tidy the Markdown files."""
    codes = [
        _run("isort", *_PYTHON_TARGETS),
        _run("black", *_PYTHON_TARGETS),
        _run("mdformat", "--wrap", "120", *_MARKDOWN_TARGETS),
    ]
    sys.exit(max(codes))


def verify() -> None:
    """Check imports, formatting, lint rules, spelling and Markdown without changing any file."""
    codes = [
        _run("isort", "--check-only", *_PYTHON_TARGETS),
        _run("black", "--check", *_PYTHON_TARGETS),
        _run("flake8", *_PYTHON_TARGETS),
        _run("codespell_lib", *_PYTHON_TARGETS, *_MARKDOWN_TARGETS),
        _run("mdformat", "--check", "--wrap", "120", *_MARKDOWN_TARGETS),
    ]
    sys.exit(max(codes))
