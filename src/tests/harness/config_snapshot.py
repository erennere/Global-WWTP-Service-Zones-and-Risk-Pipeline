"""Snapshot every config section as fully resolved by ``starter.load_config``.

Used to prove that a CLI/`os.chdir` migration changes nothing an entry point
actually reads. Run it on the pre-change tree, migrate, run it again, and diff:

    cd src
    python tests/harness/config_snapshot.py ../snap/before
    # ... migrate ...
    python tests/harness/config_snapshot.py ../snap/after
    diff -r ../snap/before ../snap/after

Each section is dumped twice - once with the process CWD at ``src/`` and once
from a scratch directory - and the two must match. That second dump is the check
that catches any path an ``os.chdir`` call was silently propping up.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR.parent))

from src.starter import load_config  # noqa: E402

# One representative override combination per snapshot, plus the no-override case.
OVERRIDE_SETS = {
    "defaults": {},
    "overridden": {
        "level": "8",
        "version": "2",
        "buffer": "9000",
        "weight_method": "linear",
        "weight_func": "mult",
        "dynamic_buffering": "false",
        "radius_quantile": "0.7",
    },
}


def stable_repr(value):
    """Serialize non-JSON values stably - a function's repr embeds its address."""
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', repr(value))}>"
    return str(value)


def section_names():
    with open(SRC_DIR / "config.yaml", encoding="utf-8") as handle:
        return list(yaml.safe_load(handle))


def resolve(script_name, overrides, cwd):
    """Resolve one section with the process CWD set to ``cwd``."""
    previous = os.getcwd()
    os.chdir(cwd)
    try:
        return load_config(
            script_name=script_name,
            config=str(SRC_DIR / "config.yaml"),
            **overrides,
        )
    finally:
        os.chdir(previous)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = []

    with tempfile.TemporaryDirectory() as elsewhere:
        for name in section_names():
            for label, overrides in OVERRIDE_SETS.items():
                target = out / f"{name}.{label}.json"
                try:
                    from_src = resolve(name, overrides, SRC_DIR)
                    from_elsewhere = resolve(name, overrides, elsewhere)
                except Exception as err:  # recorded, not raised: the diff is the product
                    target.write_text(f"ERROR {type(err).__name__}: {err}\n", encoding="utf-8")
                    failures.append(f"{name}.{label}: {type(err).__name__}: {err}")
                    continue

                dumped = json.dumps(from_src, sort_keys=True, indent=1, default=stable_repr)
                if dumped != json.dumps(from_elsewhere, sort_keys=True, indent=1, default=stable_repr):
                    failures.append(f"{name}.{label}: resolution depends on the working directory")
                target.write_text(dumped + "\n", encoding="utf-8")

    print(f"wrote {len(list(out.glob('*.json')))} snapshots to {out}")
    for line in failures:
        print(f"  FAIL {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: config_snapshot.py <output-dir>")
    sys.exit(main(sys.argv[1]))
