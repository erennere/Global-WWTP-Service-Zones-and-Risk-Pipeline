from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def _run_bash(command: str) -> subprocess.CompletedProcess[str]:
    if shutil.which("bash") is None:
        pytest.skip("bash is not available in PATH")

    src_dir = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("PYTHON_CMD", "/bin/echo")

    return subprocess.run(
        ["bash", "-lc", command],
        cwd=src_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_add_pop_sh_rejects_missing_index():
    result = _run_bash("./add_pop.sh")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    if combined.strip():
        assert ("Task ID not provided" in combined) or ("--index" in combined)


def test_add_pop_sh_rejects_missing_index_value():
    result = _run_bash("./add_pop.sh --index")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "--index requires a value" in combined


def test_add_pop_sh_rejects_unknown_argument():
    result = _run_bash("./add_pop.sh --index 0 --bogus")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "Unknown argument '--bogus'" in combined


def test_sweep_ver_ranking_sh_rejects_unknown_argument():
    result = _run_bash("./sensitivity_analysis_scripts/sweep_ver_ranking.sh --bogus")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "Unknown argument '--bogus'" in combined


def test_sweep_ver_ranking_sh_rejects_missing_shuffle_seed_value():
    result = _run_bash("./sensitivity_analysis_scripts/sweep_ver_ranking.sh --shuffle-seed")

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode != 0
    assert "--shuffle-seed requires a value" in combined


def test_parse_overrides_rejects_unknown_flag_with_exit_2():
    """A typo'd override flag must abort, not be silently shifted away.

    Before this was enforced, `--dynamic-buffer 0.5` (missing the trailing -k)
    ran a full job against config defaults and filed its output under the wrong
    directory.
    """
    result = _run_bash(
        "./pop_at_risk_river_calculations/find_unserved_pop.sh --dynamic-buffer 0.5"
    )

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 2
    assert "unknown option --dynamic-buffer" in combined


def test_parse_overrides_accepts_the_seven_standard_flags():
    result = _run_bash(
        "./pop_at_risk_river_calculations/find_unserved_pop.sh "
        "--level 8 --version 2 --buffer 9000 --weight-method linear "
        "--weight-func mult --dynamic-buffering false --radius-quantile 0.7"
    )

    assert result.returncode == 0


def test_run_stage_aborts_with_the_failing_stage_name_and_exit_code(tmp_path):
    """run_stage must surface the stage's own exit code, not tee's."""
    stub = tmp_path / "failing_python"
    stub.write_text(
        '#!/bin/sh\nif [ "$1" = "-c" ]; then exit 0; fi\necho "stage boom" >&2\nexit 7\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)

    if shutil.which("bash") is None:
        pytest.skip("bash is not available in PATH")

    src_dir = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHON_CMD"] = str(stub)
    result = subprocess.run(
        ["bash", "pop_at_risk_river_calculations/pop_differences_and_impact_polygons.sh"],
        cwd=src_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 7
    assert "stage find_unserved_pop failed with exit 7" in combined


def test_every_wrapper_is_executable_and_sets_strict_mode():
    src_dir = Path(__file__).resolve().parents[2]
    # The old-version tree is a frozen copy kept only for side-by-side comparison;
    # it is explicitly not to be modified, so it is not held to the wrapper standard.
    excluded_dirs = {".venv", "old-version-DO-NOT-CHANGE-THIS-ONLY-TO-COMPARE"}
    wrappers = [
        p for p in src_dir.rglob("*.sh")
        if not (excluded_dirs & set(p.parts)) and p.name != "utils.sh"
    ]
    assert wrappers, "no shell wrappers found"

    not_executable = [str(p.relative_to(src_dir)) for p in wrappers if not os.access(p, os.X_OK)]
    assert not_executable == [], f"missing +x: {not_executable}"

    missing_strict = [
        str(p.relative_to(src_dir))
        for p in wrappers
        if not any(
            line.startswith("set -Eeuo pipefail")
            for line in p.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert missing_strict == [], f"missing 'set -Eeuo pipefail': {missing_strict}"
