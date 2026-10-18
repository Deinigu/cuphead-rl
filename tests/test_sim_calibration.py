"""The simulator defaults must match the decompiled game constants."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cuphead_rl.sim.calibration import MAPPING, get_path
from cuphead_rl.sim.config import SimConfig

ROOT = Path(__file__).resolve().parents[1]
CONSTANTS = ROOT / "data" / "goopy_constants.json"


@pytest.mark.skipif(not CONSTANTS.exists(), reason="no decompiled constants checked in")
def test_defaults_match_game_constants():
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "calibrate_sim.py"), "--json", str(CONSTANTS)],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stdout[-3000:]


@pytest.mark.skipif(not CONSTANTS.exists(), reason="no decompiled constants checked in")
def test_from_game_constants_roundtrip():
    cfg = SimConfig.from_game_constants(CONSTANTS, live_path=None)
    data = json.loads(CONSTANTS.read_text())
    for path, key, fn in MAPPING:
        assert get_path(cfg, path) == pytest.approx(fn(data[key])) if isinstance(fn(data[key]), float) \
            else get_path(cfg, path) == fn(data[key])
