"""Shared test set-up, loaded by pytest before any test file.

Makes the project modules (app, graph, tools, ...) and the shared fakes in
tests/fakes.py importable, so the test files need no sys.path code of their
own.
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent

for folder in (PROJECT_ROOT, TESTS_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
