"""
Repo-level pytest configuration.

This project uses a `src/` layout (package lives in `src/u_ask_qa`). When running
tests without installing the package (e.g. `pytest`), we need `src/` on
`sys.path` so imports like `from u_ask_qa.core import ...` work.
"""
from __future__ import annotations
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))


