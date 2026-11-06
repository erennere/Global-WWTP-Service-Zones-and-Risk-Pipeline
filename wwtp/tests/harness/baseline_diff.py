"""Compare the current tree against a frozen baseline copy, surface by surface.

The two trees have diverged by tens of thousands of lines, so a line-level
``diff -r`` is unreadable and proves nothing. What stays reviewable is the set of
*behavior surfaces*: the public API of each module, the config keys and their
resolved values, the tuned constants, and what each shell wrapper parses.

Nothing here imports either tree - it is pure ``ast`` and ``yaml``, so a baseline
that no longer imports cleanly is still comparable. The baseline directory is
opened read-only and never written to.

Usage (from wwtp/):

    python tests/harness/baseline_diff.py old-version-DO-NOT-CHANGE-THIS-ONLY-TO-COMPARE

Writes a Markdown report to stdout; redirect it where you want it.
"""

import ast
import re
import sys
from pathlib import Path

import yaml

SKIP_DIR_PARTS = {"__pycache__", ".venv", ".vscode", "logs"}

# Modules that exist only in the new tree and are the declared destinations for
# consolidated logic. A name that vanished from its old module and reappeared in
# one of these is a *move*, not a deletion - the distinction this report exists for.
CONSOLIDATION_TARGETS = ("geo_utils.py", "utils.py", "figures_scripts/_shared.py")


def iter_python(root, baseline_name=None):
    """Yield ``(relative_path, parsed_module)`` for every parseable .py under root."""
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if SKIP_DIR_PARTS & set(rel.parts):
            continue
        if baseline_name and rel.parts and rel.parts[0] == baseline_name:
            continue
        try:
            yield rel, ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as err:
            tree_label = "baseline" if baseline_name is None else "current"
            print(f"<!-- unparseable ({tree_label} tree): {rel}: {err} -->")


def public_api(root, baseline_name=None):
    """Map ``module -> {name: signature}`` for top-level defs and classes."""
    api = {}
    for rel, tree in iter_python(root, baseline_name):
        entries = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spec = node.args
                args = [a.arg for a in spec.posonlyargs + spec.args]
                if spec.vararg:
                    args.append(f"*{spec.vararg.arg}")
                args += [f"{a.arg}=" for a in spec.kwonlyargs]
                if spec.kwarg:
                    args.append(f"**{spec.kwarg.arg}")
                entries[node.name] = args
            elif isinstance(node, ast.ClassDef):
                entries[node.name] = ["<class>"]
        if entries:
            api[str(rel)] = entries
    return api


def module_constants(root, baseline_name=None):
    """Map ``module -> {NAME: literal}`` for module-level literal assignments."""
    consts = {}
    for rel, tree in iter_python(root, baseline_name):
        entries = {}
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                entries[target.id] = repr(ast.literal_eval(node.value))
            except (ValueError, TypeError, SyntaxError):
                continue
        if entries:
            consts[str(rel)] = entries
    return consts


def config_leaves(path):
    """Flatten a config.yaml into ``{dotted.key: value}``."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    def walk(node, prefix):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                out.update(walk(value, f"{prefix}.{key}" if prefix else str(key)))
            return out
        return {prefix: node}

    return raw, walk(raw, "")


SET_RE = re.compile(r"^\s*set\s+-\S+")
FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")


def wrapper_profile(root, baseline_name=None):
    """Map ``wrapper -> {set line, sources utils.sh, parsed flags}``."""
    profile = {}
    for path in sorted(root.rglob("*.sh")):
        rel = path.relative_to(root)
        if SKIP_DIR_PARTS & set(rel.parts):
            continue
        if baseline_name and rel.parts and rel.parts[0] == baseline_name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        set_lines = [m.group(0).strip() for m in (SET_RE.match(l) for l in text.splitlines()) if m]
        profile[str(rel)] = {
            "set": set_lines[0] if set_lines else "<none>",
            "sources_utils": "lib/utils.sh" in text,
            "flags": sorted(set(FLAG_RE.findall(text))),
        }
    return profile


def section(title):
    print(f"\n## {title}\n")


def report(old_root, new_root, baseline_name):
    old_api = public_api(old_root)
    new_api = public_api(new_root, baseline_name)

    # Index new names by the module(s) that define them, so a name missing from its
    # old module can be reported as relocated rather than lost.
    new_locations = {}
    for module, entries in new_api.items():
        for name in entries:
            new_locations.setdefault(name, []).append(module)

    print("# Baseline diff audit\n")
    print(f"Baseline: `{baseline_name}`\n")
    print(
        f"- modules with top-level definitions: old {len(old_api)}, new {len(new_api)}\n"
        f"- top-level definitions: old {sum(len(v) for v in old_api.values())}, "
        f"new {sum(len(v) for v in new_api.values())}"
    )

    section("1a. Definitions gone from their old module")
    print("| old module | name | where it is now |")
    print("|---|---|---|")
    for module in sorted(old_api):
        for name in sorted(old_api[module]):
            if name in new_api.get(module, {}):
                continue
            homes = new_locations.get(name, [])
            if not homes:
                verdict = "**absent from the new tree**"
            elif any(h.replace("\\", "/") in CONSOLIDATION_TARGETS for h in homes):
                verdict = "moved to " + ", ".join(f"`{h}`" for h in homes)
            else:
                verdict = "also defined in " + ", ".join(f"`{h}`" for h in homes)
            print(f"| `{module}` | `{name}` | {verdict} |")

    section("1b. Signature changes at the same module and name")
    print("| module | name | old | new |")
    print("|---|---|---|---|")
    for module in sorted(old_api):
        for name, old_sig in sorted(old_api[module].items()):
            new_sig = new_api.get(module, {}).get(name)
            if new_sig is not None and new_sig != old_sig:
                print(f"| `{module}` | `{name}` | `{old_sig}` | `{new_sig}` |")

    old_raw, old_leaves = config_leaves(old_root / "config.yaml")
    new_raw, new_leaves = config_leaves(new_root / "config.yaml")

    section("2a. Config sections")
    print(f"- only in old: {sorted(set(old_raw) - set(new_raw)) or 'none'}")
    print(f"- only in new: {sorted(set(new_raw) - set(old_raw)) or 'none'}")

    section("2b. Config keys removed")
    for key in sorted(set(old_leaves) - set(new_leaves)):
        print(f"- `{key}` (was `{old_leaves[key]!r}`)")

    section("2c. Config keys added")
    for key in sorted(set(new_leaves) - set(old_leaves)):
        print(f"- `{key}` = `{new_leaves[key]!r}`")

    section("2d. Config values changed in place")
    print("| key | old | new |")
    print("|---|---|---|")
    for key in sorted(set(old_leaves) & set(new_leaves)):
        if old_leaves[key] != new_leaves[key]:
            print(f"| `{key}` | `{old_leaves[key]!r}` | `{new_leaves[key]!r}` |")

    old_consts = module_constants(old_root)
    new_consts = module_constants(new_root, baseline_name)

    section("3. Module-level constants changed")
    print("| module | name | old | new |")
    print("|---|---|---|---|")
    for module in sorted(old_consts):
        for name, old_value in sorted(old_consts[module].items()):
            new_value = new_consts.get(module, {}).get(name)
            if new_value is not None and new_value != old_value:
                print(f"| `{module}` | `{name}` | `{old_value}` | `{new_value}` |")

    old_sh = wrapper_profile(old_root)
    new_sh = wrapper_profile(new_root, baseline_name)

    section("4a. Wrappers added or removed")
    print(f"- only in old: {sorted(set(old_sh) - set(new_sh)) or 'none'}")
    print(f"- only in new: {sorted(set(new_sh) - set(old_sh)) or 'none'}")

    section("4b. Wrapper profile changes")
    print("| wrapper | field | old | new |")
    print("|---|---|---|---|")
    for name in sorted(set(old_sh) & set(new_sh)):
        for field in ("set", "sources_utils"):
            if old_sh[name][field] != new_sh[name][field]:
                print(f"| `{name}` | {field} | `{old_sh[name][field]}` | `{new_sh[name][field]}` |")
        gone = sorted(set(old_sh[name]["flags"]) - set(new_sh[name]["flags"]))
        added = sorted(set(new_sh[name]["flags"]) - set(old_sh[name]["flags"]))
        if gone:
            print(f"| `{name}` | flags removed | `{' '.join(gone)}` | |")
        if added:
            print(f"| `{name}` | flags added | | `{' '.join(added)}` |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: baseline_diff.py <baseline-dir-name-under-wwtp>")
    new_root = Path(__file__).resolve().parents[2]
    baseline = sys.argv[1].rstrip("/")
    old_root = new_root / baseline
    if not old_root.is_dir():
        sys.exit(f"baseline not found: {old_root}")
    report(old_root, new_root, baseline)
