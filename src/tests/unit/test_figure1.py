"""Figure 1 scripts take their thresholds from config.yaml (figure1)."""
import subprocess
import sys
from pathlib import Path

import pytest

from src.starter import load_config

pytestmark = pytest.mark.unit
FIG1 = Path(__file__).resolve().parents[3] / "figures" / "fig1"


def _setting(key):
    return subprocess.run([sys.executable, str(FIG1 / "fig1_config.py"), key], capture_output=True, text=True,
                          check=True).stdout.strip()


def test_settings_come_from_config():
    cfg = load_config(script_name="figure1")
    assert _setting("min_plants") == str(cfg["min_plants"]) == "5"
    assert _setting("tag") == f"r{int(cfg['match_radius_m'])}" == "r10000"
    assert _setting("match_dir").endswith("figures/fig1/match")


def test_no_script_hardcodes_the_threshold_or_the_radius():
    for path in FIG1.rglob("*.py"):
        text = path.read_text()
        assert "default=5)" not in text, path
        assert '"r10000"' not in text and '"r5000"' not in text, path


@pytest.mark.parametrize("script", sorted(p.name for p in FIG1.rglob("*.sh")))
def test_wrappers_parse_and_use_the_config(script):
    path = next(FIG1.rglob(script))
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    assert "fig1_config.py" in text and "scripts/" not in text


@pytest.mark.parametrize("script", sorted(str(p.relative_to(FIG1)) for p in FIG1.rglob("*.py")))
def test_scripts_compile(script):
    import py_compile

    py_compile.compile(str(FIG1 / script), doraise=True)
