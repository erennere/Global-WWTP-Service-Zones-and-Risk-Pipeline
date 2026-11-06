"""Per-script configuration loading and resolution for plant capacity workflows."""

from __future__ import annotations

import math
import json
import os
from string import Formatter
import sys
from copy import deepcopy
from typing import Any

import yaml


_MISSING = object()
_FORMATTER = Formatter()

# config.yaml lives next to this module. Resolving relative config paths against
# it - rather than against the process working directory - is what lets entry
# points run from anywhere without an os.chdir() call in every main().
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILENAME = "config.yaml"
CONFIG_PATH_ENV_VAR = "WWTP_SERVICE_PIPELINE_CONFIG"
# Overlays: YAML files that change values of the base config, applied in order:
# the files in WWTP_CONFIG_OVERLAYS (os.pathsep-separated, e.g. config.cluster.yaml
# on the cluster), then config.local.yaml next to the base config if it exists
# (git-ignored, per machine), then --set SECTION.KEY=VALUE from the command line.
# An overlay may only change keys the base config has, so a typo fails at load.
OVERLAYS_ENV_VAR = "WWTP_CONFIG_OVERLAYS"
LOCAL_OVERLAY_FILENAME = "config.local.yaml"
_OVERRIDE_FIELDS = (
    ("level", 0, False),
    ("version", 1, False),
    ("buffer", 2, False),
    ("weight_method", 3, False),
    ("weight_func", 4, True),
    ("dynamic_buffering", 5, False),
    ("size_measure", 6, False),
)
# Path token for the service-radius mode when dynamic_buffering is on.
SIZE_MEASURE_TOKENS = {"reported": "rep", "footprint": "fp"}
_OVERRIDE_FLAG_TO_FIELD = {
    "--level": "level",
    "--version": "version",
    "--buffer": "buffer",
    "--weight-method": "weight_method",
    "--weight-func": "weight_func",
    "--dynamic-buffering": "dynamic_buffering",
    "--size-measure": "size_measure",
}


def add_standard_override_arguments(parser):
    """Add the shared named config override flags used across scripts."""
    parser.add_argument("--level", default=None, help="Optional config level override")
    parser.add_argument("--version", default=None, help="Optional config version override")
    parser.add_argument("--buffer", default=None, help="Optional config buffer override")
    parser.add_argument("--weight-method", dest="weight_method", default=None, help="Optional config weight_method override")
    parser.add_argument("--weight-func", dest="weight_func", default=None, help="Optional config weight_func override: 'mult', 'add', or ''")
    parser.add_argument("--dynamic-buffering", dest="dynamic_buffering", default=None, help="Optional dynamic buffering override (true/false)")
    parser.add_argument("--size-measure", dest="size_measure", default=None, help="Optional plant size measure override: 'reported' or 'footprint'")
    parser.add_argument("--set", dest="sets", action="append", default=None, metavar="SECTION.KEY=VALUE",
                        help="Change any config value for this run (repeatable), e.g. --set create_voronoi.n_points=200")
    return parser


class ConfigResolutionError(KeyError):
    """Raised when a script config section or key cannot be resolved."""

    def __init__(self, script_name: str, key: str, message: str):
        self.script_name = script_name
        self.key = key
        super().__init__(message)


    ################################################################################
    # SECTION 1: CLI OVERRIDE NORMALIZATION & PARSING
    ################################################################################

    # Primitive value normalizers/parsers


def _normalize_optional_cli_value(value, preserve_empty=False):
    """Normalize optional CLI values, treating empty/None/NaN/null as omitted."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "":
            return "" if preserve_empty else None
        if normalized.lower() in {"none", "nan", "null"}:
            return None
        return normalized
    return value

def _parse_optional_int(value, field_name):
    """Parse optional integer overrides."""
    value = _normalize_optional_cli_value(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} '{value}'. Must be an integer.") from exc

def _parse_optional_weight_func(value, field_name="weight_func"):
    """Parse optional weight function mode override."""
    if value is None:
        return None

    normalized = str(value).strip().lower()
    if normalized in {"mult", "add", ""}:
        return normalized
    raise ValueError(f"Invalid {field_name} '{value}'. Must be one of: mult, add, ''.")

def _normalize_weight_func(value, field_name="weight_func"):
    """Return a normalized weight function token, defaulting missing values to ''."""
    normalized = _parse_optional_weight_func(value, field_name)
    return "" if normalized is None else normalized

def _parse_optional_bool(value, field_name):
    """Parse optional boolean overrides from common truthy/falsey strings."""
    value = _normalize_optional_cli_value(value)
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid {field_name} '{value}'. Must be a boolean value.")

def _parse_optional_size_measure(value, field_name="size_measure"):
    """Parse an optional size-measure override ('reported' or 'footprint')."""
    value = _normalize_optional_cli_value(value)
    if value is None:
        return None
    if value not in SIZE_MEASURE_TOKENS:
        raise ValueError(f"Invalid {field_name} '{value}'. Must be one of {sorted(SIZE_MEASURE_TOKENS)}.")
    return value

def _collect_raw_overrides(args: Any = None, argv: list[str] | None = None):
    argv_values: list[str] = list(sys.argv) if argv is None else list(argv)

    raw_overrides: dict[str, Any] = {}

    # When scripts pass named flags directly (without argparse in the script),
    # parse those first so wrappers can standardize on one CLI style.
    if args is None:
        named_values: dict[str, Any] = {}
        named_sets: list[str] = []
        idx = 1
        while idx < len(argv_values):
            token = argv_values[idx]
            if not isinstance(token, str):
                idx += 1
                continue

            if token == "--set" and idx + 1 < len(argv_values):
                named_sets.append(argv_values[idx + 1])
                idx += 2
                continue
            if isinstance(token, str) and token.startswith("--set="):
                named_sets.append(token.split("=", 1)[1])
                idx += 1
                continue

            if token in _OVERRIDE_FLAG_TO_FIELD:
                field_name = _OVERRIDE_FLAG_TO_FIELD[token]
                next_value = argv_values[idx + 1] if idx + 1 < len(argv_values) else None
                if isinstance(next_value, str) and next_value.startswith("--"):
                    next_value = None
                    idx += 1
                else:
                    idx += 2
                named_values[field_name] = next_value
                continue

            if token.startswith("--") and "=" in token:
                flag, flag_value = token.split("=", 1)
                if flag in _OVERRIDE_FLAG_TO_FIELD:
                    field_name = _OVERRIDE_FLAG_TO_FIELD[flag]
                    named_values[field_name] = flag_value
            idx += 1

        for field_name, _, preserve_empty in _OVERRIDE_FIELDS:
            raw_overrides[field_name] = _normalize_optional_cli_value(
                named_values.get(field_name),
                preserve_empty=preserve_empty,
            )
        raw_overrides["sets"] = named_sets
        return raw_overrides

    for field_name, _, preserve_empty in _OVERRIDE_FIELDS:
        raw_overrides[field_name] = _normalize_optional_cli_value(
            getattr(args, field_name, None),
            preserve_empty=preserve_empty,
        )
    raw_overrides["sets"] = list(getattr(args, "sets", None) or [])
    return raw_overrides


# Public parser used by scripts before load_config()
def parse_config_overrides(args: Any = None, argv: list[str] | None = None):
    """Parse optional config overrides for ``load_config``.

    Overrides come either from an argparse namespace built with
    ``add_standard_override_arguments`` or from named ``--flag value`` tokens in
    ``argv``. Positional overrides are not supported: a stray argv token used to
    become ``level`` silently, which filed a run's output under the wrong path.
    """
    raw_overrides = _collect_raw_overrides(args=args, argv=argv)

    return {
        "level": raw_overrides["level"],
        "version": raw_overrides["version"],
        "buffer": _parse_optional_int(raw_overrides["buffer"], "buffer"),
        "weight_method": raw_overrides["weight_method"],
        "weight_func": _parse_optional_weight_func(raw_overrides["weight_func"], "weight_func"),
        "dynamic_buffering": _parse_optional_bool(raw_overrides["dynamic_buffering"], "dynamic_buffering"),
        "size_measure": _parse_optional_size_measure(raw_overrides["size_measure"]),
        "sets": parse_set_assignments(raw_overrides["sets"]),
    }


def parse_set_assignments(assignments):
    """``["create_voronoi.n_points=200", ...]`` -> ``[(("create_voronoi", "n_points"), 200), ...]`` (values as YAML)."""
    parsed = []
    for item in assignments or []:
        key, sep, value = str(item).partition("=")
        parts = tuple(p for p in key.strip().split(".") if p)
        if not sep or len(parts) < 2:
            raise ValueError(f"Invalid --set {item!r}. Use SECTION.KEY=VALUE, e.g. create_voronoi.n_points=200.")
        parsed.append((parts, yaml.safe_load(value)))
    return parsed


################################################################################
# SECTION 2: RAW CONFIG LOADING & SECTION INHERITANCE
################################################################################


def _load_raw_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        raise TypeError("config.yaml must contain a top-level mapping of script sections")
    return raw_config


def config_overlay_paths(config_path: str) -> list[str]:
    """Overlay files for ``config_path``, in the order they are applied."""
    paths = [resolve_config_path(p) for p in os.environ.get(OVERLAYS_ENV_VAR, "").split(os.pathsep) if p.strip()]
    local = os.path.join(os.path.dirname(config_path), LOCAL_OVERLAY_FILENAME)
    if os.path.isfile(local):
        paths.append(local)
    return paths


def _overlay(base: dict, overlay: dict, origin: str, where: tuple = ()) -> None:
    for key, value in overlay.items():
        if key not in base:
            raise KeyError(f"{origin}: {'.'.join(where + (key,))} is not a key of the base config")
        if isinstance(value, dict) and isinstance(base[key], dict):
            _overlay(base[key], value, origin, where + (key,))
        else:
            base[key] = value


def load_layered_config(config_path: str, sets=None) -> tuple[dict, list[str]]:
    """Base config with overlay files and ``--set`` assignments applied; also the layers used."""
    raw_config = _load_raw_config(config_path)
    layers = [config_path]
    for path in config_overlay_paths(config_path):
        overlay = _load_raw_config(path)
        _overlay(raw_config, overlay, path)
        layers.append(path)
    for parts, value in sets or []:
        nested = value
        for part in reversed(parts):
            nested = {part: nested}
        _overlay(raw_config, nested, "--set " + ".".join(parts))
        layers.append("--set " + ".".join(parts) + "=" + json.dumps(value, default=str))
    return raw_config, layers


def _require_mapping_section(section_name: str, section_cfg) -> dict:
    if not isinstance(section_cfg, dict):
        raise ConfigResolutionError(
            section_name,
            "<section>",
            f"Config section '{section_name}' must be a mapping.",
        )
    return section_cfg


def _ordered_sections(raw_config: dict) -> list[str]:
    return list(raw_config)


def _nested_value_or_missing(section_cfg, key_path):
    current = section_cfg
    for key in key_path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _merge_first_defined(base: dict, candidate: dict) -> dict:
    merged = deepcopy(base)
    for key, value in candidate.items():
        if value is None:
            continue

        if isinstance(value, dict):
            existing = merged.get(key, _MISSING)
            if existing is _MISSING or existing is None:
                merged[key] = _merge_first_defined({}, value)
                continue
            if isinstance(existing, dict):
                merged[key] = _merge_first_defined(existing, value)
            continue

        if key not in merged or merged[key] is None:
            merged[key] = deepcopy(value)
    return merged


def _build_base_config(script_name: str, raw_config: dict, ordered_sections: list[str]) -> dict:
    base_cfg = {}
    for section_name in ordered_sections:
        if section_name == script_name:
            break

        section_cfg = _require_mapping_section(section_name, raw_config[section_name])
        base_cfg = _merge_first_defined(base_cfg, section_cfg)
    return base_cfg


def _resolve_value_from_base(script_name: str, key_path: tuple[str, ...], base_cfg: dict):
    candidate = _nested_value_or_missing(base_cfg, key_path)
    if candidate is not _MISSING:
        return deepcopy(candidate)

    dotted_key = ".".join(key_path)
    raise ConfigResolutionError(
        script_name,
        dotted_key,
        f"Config key '{dotted_key}' for script '{script_name}' could not be resolved from earlier canonical sections.",
    )


def _resolve_mapping(script_name: str, key_path: tuple[str, ...], mapping: dict, base_cfg: dict) -> dict:
    resolved = {}

    for key, value in mapping.items():
        current_path = (*key_path, key)
        if isinstance(value, dict):
            resolved[key] = _resolve_mapping(script_name, current_path, value, base_cfg)
            continue
        if value is None:
            resolved[key] = _resolve_value_from_base(script_name, current_path, base_cfg)
            continue
        resolved[key] = deepcopy(value)
    return resolved


def resolve_config(script_name: str, raw_config: dict) -> dict:
    """Resolve a script section using earlier YAML sections as inheritance sources."""
    if script_name not in raw_config:
        raise ConfigResolutionError(
            script_name,
            "<section>",
            f"Config section '{script_name}' is missing from config.yaml.",
        )

    section_cfg = _require_mapping_section(script_name, raw_config[script_name])

    ordered_sections = _ordered_sections(raw_config)
    if script_name not in ordered_sections:
        raise ConfigResolutionError(
            script_name,
            "<section>",
            f"Config section '{script_name}' is not part of the config section order.",
        )

    base_cfg = _build_base_config(script_name, raw_config, ordered_sections)
    return _resolve_mapping(script_name, tuple(), section_cfg, base_cfg)


################################################################################
# SECTION 3: RUNTIME OVERRIDES & DERIVED VALUES
################################################################################


def _apply_optional_override(cfg: dict, key: str, override, parser=None) -> None:
    if override is None and key not in cfg:
        return

    value = cfg[key] if override is None else override
    if parser is not None:
        value = parser(value, key)
    cfg[key] = value


def _apply_runtime_overrides(
    resolved_cfg: dict,
    *,
    level=None,
    version=None,
    buffer=None,
    weight_method=None,
    weight_func=None,
    dynamic_buffering=None,
    size_measure=None,
) -> dict:
    cfg = deepcopy(resolved_cfg)
    _apply_optional_override(cfg, "level", level)
    _apply_optional_override(cfg, "version", version)
    _apply_optional_override(cfg, "buffer", buffer)
    _apply_optional_override(cfg, "weight_method", weight_method)
    _apply_optional_override(
        cfg,
        "weight_func",
        weight_func,
        _normalize_weight_func,
    )
    _apply_optional_override(cfg, "dynamic_buffering", dynamic_buffering, _parse_optional_bool)
    _apply_optional_override(cfg, "size_measure", size_measure, _parse_optional_size_measure)
    return cfg


def _derive_runtime_values(cfg: dict, default_distance_additive, default_distance_multiplicative) -> dict:
    derived = {}
    weight_type_map = {
        "linear": "li",
        "square_root": "sq",
        "logarithmic": "log",
        "sigmoid": "sig",
        "size": "sz",
    }
    if "weight_method" in cfg:
        if cfg["weight_method"] not in weight_type_map:
            valid_methods = ", ".join(weight_type_map)
            raise ValueError(
                f"Invalid weight_method '{cfg['weight_method']}'. Must be one of: {valid_methods}."
            )
        derived["weight_type"] = weight_type_map[cfg["weight_method"]]

    if "weight_func" in cfg:
        weight_func = _normalize_weight_func(cfg["weight_func"], "weight_func")
        derived["weight_func_suffix"] = {
            "mult": "_mult",
            "add": "_add",
            "": "",
        }[weight_func]
        derived["distance_fn"] = (
            default_distance_multiplicative
            if weight_func in {"", "mult"}
            else default_distance_additive
        )

    if "weight_method" in cfg and cfg["weight_method"] == "size" and "weight_func" in cfg:
        if _normalize_weight_func(cfg["weight_func"], "weight_func") != "mult":
            raise ValueError("weight_method 'size' needs weight_func 'mult' (multiplicative weighted distance)")

    if "size_measure" in cfg and cfg["size_measure"] is not None and cfg["size_measure"] not in SIZE_MEASURE_TOKENS:
        raise ValueError(f"size_measure must be one of {sorted(SIZE_MEASURE_TOKENS)}, got {cfg['size_measure']!r}")

    if "buffer" in cfg and "dynamic_buffering" in cfg:
        if cfg["dynamic_buffering"] and ("size_measure" not in cfg or cfg["size_measure"] is None):
            raise ValueError("size_measure must be configured when dynamic_buffering is true")

        if cfg["dynamic_buffering"]:
            derived["buffer_path_token"] = SIZE_MEASURE_TOKENS[cfg["size_measure"]]
        else:
            derived["buffer_path_token"] = str(int(cfg["buffer"]))

    calculate_buffer_keys = {"buffer", "dynamic_buffering", "max_buffer", "size_measure", "footprint_to_people",
                             "reported_size_columns", "reported_size_skip", "radius_calibration"}
    if calculate_buffer_keys.issubset(cfg):
        kwargs = {
            "buffer": cfg["buffer"],
            "dynamic_buffering": cfg["dynamic_buffering"],
            "max_buffer": cfg["max_buffer"],
        }
        if cfg["size_measure"] is not None:
            # The plant size is needed by the radius and by weight_method 'size'.
            kwargs.update({
                "size_measure": cfg["size_measure"],
                "footprint_to_people": _footprint_to_people(cfg["footprint_to_people"]),
                "reported_size_columns": list(cfg["reported_size_columns"]),
                "reported_size_skip": dict(cfg["reported_size_skip"]),
            })
        if cfg["dynamic_buffering"]:
            row = _calibration_row(cfg["radius_calibration"], cfg["size_measure"])
            kwargs.update({
                "radius_exponent": float(row["radius_exponent"]),
                "radius_at_10k_people_km": float(row["radius_at_10k_people_km"]),
            })
        derived["calculate_buffer_kwargs"] = kwargs

    return derived


def _footprint_to_people(value):
    keys = ("intercept", "slope", "unsized_people")
    if not isinstance(value, dict) or set(keys) - set(value):
        raise ValueError(f"footprint_to_people must map {keys} to numbers")
    return {key: float(value[key]) for key in keys}


def _calibration_row(table, size_measure):
    """The radius_calibration entry for ``size_measure``."""
    keys = ("radius_exponent", "radius_at_10k_people_km")
    if not isinstance(table, dict) or not table:
        raise ValueError(f"radius_calibration must map each size measure to {keys}")
    if size_measure not in table:
        raise ValueError(
            f"size_measure {size_measure!r} has no radius_calibration entry; choose one of {sorted(table)} "
            "or add it with wwtp.calibration.service_radius"
        )
    row = table[size_measure]
    if not isinstance(row, dict) or set(keys) - set(row):
        raise ValueError(f"radius_calibration[{size_measure!r}] must have {keys}")
    return row


################################################################################
# SECTION 4: PATH TEMPLATE EXPANSION
################################################################################


def _normalize_cfg_path(path_value, base_dir):
    """Return an absolute filesystem path for a config entry, passing URLs through unchanged."""
    if not isinstance(path_value, str):
        return path_value
    if "://" in path_value:
        return path_value

    expanded = os.path.expanduser(path_value)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(base_dir, expanded))


def _format_field_names(template: str) -> list[str]:
    field_names = []
    for _, field_name, _, _ in _FORMATTER.parse(template):
        if field_name is None:
            continue
        normalized = field_name.split(".", 1)[0].split("[", 1)[0]
        if normalized:
            field_names.append(normalized)
    return field_names


def _expand_paths(script_name: str, paths_cfg: dict, config_dir: str, cfg: dict, derived: dict) -> dict:
    raw_data_dir = paths_cfg["data_dir"]
    context = {}
    if raw_data_dir is not None:
        context["data_dir"] = raw_data_dir

    for key in ("level", "version", "industrial_min_cells"):
        if key in cfg:
            value = cfg[key]
            context[key] = str(value) if key == "industrial_min_cells" else value

    if "buffer_path_token" in derived:
        context["buffer"] = derived["buffer_path_token"]
    if "weight_type" in derived:
        context["weight_type"] = derived["weight_type"]
    if "weight_func_suffix" in derived:
        context["weight_func"] = derived["weight_func_suffix"]

    expanded_paths = {}
    for key, value in paths_cfg.items():
        formatted = value
        if isinstance(value, str):
            missing_fields = [field for field in _format_field_names(value) if field not in context]
            if missing_fields:
                missing_list = ", ".join(sorted(set(missing_fields)))
                raise ConfigResolutionError(
                    script_name,
                    f"paths.{key}",
                    f"Path 'paths.{key}' for script '{script_name}' requires unresolved config field(s): {missing_list}.",
                )
            formatted = value.format(**context)
        expanded_paths[key] = _normalize_cfg_path(formatted, config_dir)
    return expanded_paths


################################################################################
# SECTION 5: PUBLIC CONFIG API
################################################################################


def get_runtime_params(cfg: dict) -> dict:
    """Return validated impact-model runtime parameters from the loaded config."""
    section = cfg["impact_polygons_pop_params"]
    if not isinstance(section, dict):
        raise TypeError("cfg['impact_polygons_pop_params'] must be a dict")

    return {
        "org_per_pop": float(section["org_per_pop"]),
        "c_limit": float(section["c_limit"]),
        "base_k": float(section["base_k"]),
        "theta": float(section["theta"]),
        "step_m": float(section["step_m"]),
        "least_discharge_cms": float(section["least_discharge_cms"]),
        "velocity_m_s": float(section["velocity_m_s"]),
        "water_temp_equator_c": float(section["water_temp_equator_c"]),
        "impact_radii": [float(value) for value in section["impact_radii"]],
    }


def resolve_config_path(config=None):
    """Return the absolute path of the config file to load.

    Resolution order: an explicit ``config`` argument, then the
    ``WWTP_SERVICE_PIPELINE_CONFIG`` environment variable, then ``config.yaml``
    next to this module. Relative values are resolved against the package
    directory, never against the process working directory.
    """
    if config is None:
        config = os.environ.get(CONFIG_PATH_ENV_VAR) or DEFAULT_CONFIG_FILENAME
    config = os.path.expanduser(str(config))
    if os.path.isabs(config):
        return os.path.abspath(config)
    return os.path.abspath(os.path.join(_PACKAGE_DIR, config))


def load_config(
    script_name,
    config=None,
    level=None,
    version=None,
    buffer=None,
    weight_method=None,
    weight_func=None,
    dynamic_buffering=None,
    size_measure=None,
    sets=None,
):
    """Load and resolve the per-script configuration section for a workflow.

    The base config (``config``, ``WWTP_SERVICE_PIPELINE_CONFIG`` or
    ``config.yaml``) gets the overlay files and ``sets`` (parsed ``--set``
    assignments) first; ``cfg["config_layers"]`` lists what was applied.
    """
    from .service_areas import default_distance_additive, default_distance_multiplicative

    config_path = resolve_config_path(config)
    config_dir = os.path.dirname(config_path)
    raw_config, layers = load_layered_config(config_path, sets)

    # Resolve the script section against earlier canonical sections first.
    script_cfg = resolve_config(script_name, raw_config)

    # Apply optional CLI/runtime overrides onto the resolved section.
    script_cfg = _apply_runtime_overrides(
        script_cfg,
        level=level,
        version=version,
        buffer=buffer,
        weight_method=weight_method,
        weight_func=weight_func,
        dynamic_buffering=dynamic_buffering,
        size_measure=size_measure,
    )

    # Derive helper values and expand any templated output/input paths last.
    derived = _derive_runtime_values(script_cfg, default_distance_additive, default_distance_multiplicative)
    if "paths" in script_cfg:
        script_cfg["paths"] = _expand_paths(script_name, script_cfg["paths"], config_dir, script_cfg, derived)
    script_cfg.update(derived)
    script_cfg["config_layers"] = layers
    return script_cfg


def main(argv=None):
    """``python -m wwtp.starter show <section> [--set ...]``: print the resolved config of a stage."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m wwtp.starter", description=main.__doc__)
    parser.add_argument("command", choices=["show"])
    parser.add_argument("section", help="config section, e.g. create_voronoi")
    add_standard_override_arguments(parser)
    args = parser.parse_args(argv)
    overrides = parse_config_overrides(args=args)
    cfg = load_config(script_name=args.section, **overrides)
    printable = {k: (f"<{getattr(v, '__qualname__', v)}>" if callable(v) else v) for k, v in cfg.items()}
    print(yaml.safe_dump(printable, sort_keys=False, allow_unicode=True), end="")


if __name__ == "__main__":
    main()