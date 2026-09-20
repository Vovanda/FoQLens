"""The code reads clean to pyflakes: a name shadowed before it is read, an unused import or an undefined name fails
here - the night of 20.09 lost a run to a local list named like the imported function it hid."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_pyflakes_finds_nothing():
    found = subprocess.run([sys.executable, "-m", "pyflakes", "src", "scripts", "tests"], cwd=ROOT,
                           capture_output=True, text=True)
    assert found.returncode == 0, found.stdout + found.stderr
