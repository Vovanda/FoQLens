"""Every script imports: a name a script takes from the package and the package lost fails here, not at a night run."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent / "scripts"
# scripts that do their work at import (read argv and files at the top level): they cannot be imported alone
RUN_AT_IMPORT = {"judge_synthetic_run", "judge_synthetic_sheet"}
SCRIPTS = sorted(p for p in ROOT.glob("*.py") if p.stem not in RUN_AT_IMPORT)


@pytest.mark.parametrize("path", SCRIPTS, ids=[p.stem for p in SCRIPTS])
def test_the_script_imports(path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))  # scripts import their siblings as a run from scripts/ would
    name = f"script_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
