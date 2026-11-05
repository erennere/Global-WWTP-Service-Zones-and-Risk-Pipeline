"""Run manifests: what each stage run used, so an output (or a paper figure) can name its run.

``write_run_manifest(cfg, stage)`` is called by every stage right after it loads
its config. It writes ``<run_manifest.paths.manifest_dir>/<stage>_<UTC time>.yaml``
with the resolved config, the git commit and whether the working tree had
uncommitted changes, the versions of the geospatial packages, and the size,
modification time and (up to ``hash_max_mb``) SHA-256 of every input file the
config names that exists when the stage starts.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import subprocess
import sys
from importlib import metadata

import yaml

from .starter import load_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ("numpy", "pandas", "geopandas", "shapely", "pyproj", "pyogrio", "rasterio", "exactextract", "duckdb",
            "scipy", "scikit-learn", "opencv-python-headless", "pyarrow")


def _git(*args):
    try:
        res = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as err:
        return None, str(err)
    return (res.stdout.strip(), None) if res.returncode == 0 else (None, res.stderr.strip())


def code_version():
    """Commit, branch and whether tracked files differ from the commit."""
    sha, err = _git("rev-parse", "HEAD")
    branch, _ = _git("rev-parse", "--abbrev-ref", "HEAD")
    status, _ = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": sha, "branch": branch, "uncommitted_changes": bool(status) if status is not None else None,
            "error": err}


def package_versions():
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def file_fingerprint(path, hash_max_mb):
    """Size, modification time and, for files up to ``hash_max_mb``, SHA-256."""
    stat = os.stat(path)
    out = {"bytes": stat.st_size,
           "modified_utc": _dt.datetime.fromtimestamp(stat.st_mtime, _dt.timezone.utc).isoformat(timespec="seconds")}
    if stat.st_size <= hash_max_mb * 1024 * 1024:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        out["sha256"] = digest.hexdigest()
    return out


def _plain(value):
    """Config values as YAML-safe data (callables by name)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if callable(value):
        return f"<{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', repr(value))}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def write_run_manifest(cfg, stage, argv=None):
    """Write the manifest of this run of ``stage``; return its path."""
    settings = load_config(script_name="run_manifest")
    manifest_dir = settings["paths"]["manifest_dir"]
    hash_max_mb = float(settings["hash_max_mb"])
    now = _dt.datetime.now(_dt.timezone.utc)
    inputs = {}
    for key, path in cfg["paths"].items():
        if isinstance(path, str) and os.path.isfile(path):
            inputs[key] = {"path": path, **file_fingerprint(path, hash_max_mb)}
    manifest = {
        "stage": stage,
        "started_utc": now.isoformat(timespec="seconds"),
        "command": list(sys.argv if argv is None else argv),
        "host": platform.node(),
        "slurm_job": {k: os.environ[k] for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")
                      if k in os.environ},
        "code": code_version(),
        "packages": package_versions(),
        "inputs_found_at_start": inputs,
        "config": _plain(cfg),
    }
    os.makedirs(manifest_dir, exist_ok=True)
    out = os.path.join(manifest_dir, f"{stage}_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, allow_unicode=True)
    return out
