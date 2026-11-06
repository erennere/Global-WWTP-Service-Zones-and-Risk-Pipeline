"""Diff the *resolved* config of a baseline tree against the current tree.

``baseline_diff.py`` compares config.yaml as written. This compares it as
``starter.resolve_config`` actually delivers it - after null-inheritance from
earlier sections and after ``{placeholder}`` interpolation - which is what the
pipeline reads. A key can be textually unchanged and still resolve differently
because an *earlier* section it inherits from moved.

The baseline's ``starter`` is loaded by file path, not imported as a package:
the directory name is not a valid identifier and the live ``wwtp`` package is
already installed, so a plain ``import starter`` would silently resolve to the
wrong tree. Both trees are read-only here - nothing is executed from them
beyond ``resolve_config`` on a parsed YAML document.

Usage (from wwtp/):

    python tests/harness/baseline_config_diff.py old-version-DO-NOT-CHANGE-THIS-ONLY-TO-COMPARE
"""

import importlib.util
import sys
from pathlib import Path

import yaml

# Loading the baseline's starter.py writes __pycache__ into a tree that is
# supposed to be read-only. Set before any baseline module is loaded.
sys.dont_write_bytecode = True


def load_starter(tree_root, alias):
    """Import ``<tree_root>/starter.py`` under a unique module name."""
    spec = importlib.util.spec_from_file_location(alias, tree_root / "starter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def flatten(node, prefix=""):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: node}


def resolved_sections(tree_root, starter):
    """Return ``{section: flattened resolved config}``, or the error it raised."""
    raw = yaml.safe_load((tree_root / "config.yaml").read_text(encoding="utf-8"))
    out = {}
    for name in raw:
        try:
            out[name] = flatten(starter.resolve_config(name, raw))
        except Exception as err:
            out[name] = {"<<ERROR>>": f"{type(err).__name__}: {err}"}
    return out


def normalize(value, tree_root, label):
    """Strip the tree root so path values from two trees are comparable."""
    text = str(value)
    return text.replace(str(tree_root), f"<{label}>").replace(tree_root.name, f"<{label}>")


def main(baseline_name):
    new_root = Path(__file__).resolve().parents[2]
    old_root = new_root / baseline_name

    old = resolved_sections(old_root, load_starter(old_root, "_baseline_starter"))
    new = resolved_sections(new_root, load_starter(new_root, "_current_starter"))

    print("# Resolved-config diff\n")
    print(f"Baseline: `{baseline_name}`\n")

    print("## Sections that resolve in one tree but not the other\n")
    for name in sorted(set(old) | set(new)):
        old_err = old.get(name, {}).get("<<ERROR>>")
        new_err = new.get(name, {}).get("<<ERROR>>")
        if name not in old:
            print(f"- `{name}`: new only — {new_err or 'resolves'}")
        elif name not in new:
            print(f"- `{name}`: baseline only — {old_err or 'resolves'}")
        elif bool(old_err) != bool(new_err):
            print(f"- `{name}`: baseline `{old_err or 'resolves'}` -> current `{new_err or 'resolves'}`")

    print("\n## Resolved values that changed\n")
    print("| section | key | baseline | current |")
    print("|---|---|---|---|")
    for name in sorted(set(old) & set(new)):
        old_cfg, new_cfg = old[name], new[name]
        if "<<ERROR>>" in old_cfg or "<<ERROR>>" in new_cfg:
            continue
        for key in sorted(set(old_cfg) & set(new_cfg)):
            old_value = normalize(old_cfg[key], old_root, "TREE")
            new_value = normalize(new_cfg[key], new_root, "TREE")
            if old_value != new_value:
                print(f"| `{name}` | `{key}` | `{old_value}` | `{new_value}` |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: baseline_config_diff.py <baseline-dir-name-under-wwtp>")
    main(sys.argv[1].rstrip("/"))
