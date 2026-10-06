"""Acceptance test for BWB Graph integration with viscous drag modes."""

import pytest
import numpy as np
import subprocess
import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_invalid_viscous_drag_mode_raises():
    """Verify that an invalid viscous_drag_mode raises ValueError before graph construction."""
    cmd = [
        sys.executable,
        "examples/showcase_examples/rectangular_wing/ex_rectangular_wing_to_bwb.py"
    ]
    env = os.environ.copy()
    env['VISCOUS_DRAG_MODE'] = 'invalid_mode'
    env['SKIP_OPTIMIZATION'] = '1'

    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Unknown viscous_drag_mode: 'invalid_mode'" in res.stderr


def test_constant_cd0_legacy_compatibility():
    """Gate 5 & 6: Verify constant_cd0 mode reproduces legacy parasite drag."""
    cmd = [
        sys.executable,
        "examples/showcase_examples/rectangular_wing/ex_rectangular_wing_to_bwb.py"
    ]
    env = os.environ.copy()
    env['VISCOUS_DRAG_MODE'] = 'constant_cd0'
    env['SKIP_OPTIMIZATION'] = '1'
    env['BWB_RESOLUTION'] = 'fast'

    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed with stderr:\n{res.stderr}"
    assert "Selected Mode: CONSTANT_CD0" in res.stdout
    assert "CD_viscous:    80.00 counts (0.008000)" in res.stdout


def test_ibl_station_constraints_smoke():
    """Gate 6: Verify ibl mode compiles and runs post-processing with station-wise B-spline constraints."""
    cmd = [
        sys.executable,
        "examples/showcase_examples/rectangular_wing/ex_rectangular_wing_to_bwb.py"
    ]
    env = os.environ.copy()
    env['VISCOUS_DRAG_MODE'] = 'ibl'
    env['SKIP_OPTIMIZATION'] = '1'
    env['BWB_RESOLUTION'] = 'fast'

    res = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert res.returncode == 0, f"Script failed with stderr:\n{res.stderr}"
    assert "Selected Mode: IBL" in res.stdout
    assert "CD_viscous:" in res.stdout
    assert "H_max_ibl:" in res.stdout
