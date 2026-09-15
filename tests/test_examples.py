import os
import sys
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# Active example test targets (excluding BWB examples per request)
ACTIVE_EXAMPLES = [
    "showcase_examples/lift_plus_cruise/ex_lift_plus_cruise.py",
]


@pytest.mark.example
@pytest.mark.slow
@pytest.mark.parametrize("example_rel_path", ACTIVE_EXAMPLES)
def test_example_script_execution(example_rel_path):
    """Executes standalone example scripts in an isolated child process to ensure

    end-to-end functionality without cross-test state or memory pollution.
    """
    script_path = EXAMPLES_DIR / example_rel_path
    assert script_path.exists(), f"Example script does not exist: {script_path}"

    env = os.environ.copy()
    env["PYVISTA_OFF_SCREEN"] = "true"
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Example script '{example_rel_path}' failed with exit code {result.returncode}!\n"
        f"=== STDOUT (last 2000 characters) ===\n{result.stdout[-2000:]}\n"
        f"=== STDERR (last 2000 characters) ===\n{result.stderr[-2000:]}"
    )

