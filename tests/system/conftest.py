"""tests/system: register the `slow` marker; slow tests run only with -m slow."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long fast-path runs (select with -m slow)")


def pytest_collection_modifyitems(config, items):
    if "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow")
    for it in items:
        if "slow" in it.keywords:
            it.add_marker(skip)
