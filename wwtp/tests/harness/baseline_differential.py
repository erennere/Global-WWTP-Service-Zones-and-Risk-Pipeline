"""Run the baseline's implementation and the current one on identical inputs.

Layers 1-3 compare *surfaces* - names, signatures, config keys. They cannot see
that a consolidated helper computes a different number than the four functions it
replaced. This does: it loads the baseline module by file path, loads the current
one normally, feeds both the same fixed inputs, and reports every case where the
two disagree.

A disagreement is not automatically a bug - several consolidations were approved
*because* the old behavior was wrong. So each case is declared with an
``expect`` of ``"same"`` or ``"differ"``, and the report calls out both
directions: a "same" case that differs is a regression, and a "differ" case that
matches means the fix did not actually land.

The baseline is loaded read-only, by file path under a private module alias -
never imported as a package, since the live ``wwtp`` package is installed and
would shadow it.

Usage (from wwtp/):

    python tests/harness/baseline_differential.py old-version-DO-NOT-CHANGE-THIS-ONLY-TO-COMPARE
"""

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from wwtp import service_areas as sa

warnings.filterwarnings("ignore")

# Importing a baseline module writes __pycache__ next to it, which edits a tree
# that is supposed to be read-only. Set before any baseline module is loaded.
sys.dont_write_bytecode = True

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC.parent))

CASES = []


def case(name, expect, note):
    """Register a comparison. ``expect`` is "same" or "differ"."""

    def wrap(fn):
        CASES.append((name, expect, note, fn))
        return fn

    return wrap


def load_baseline(root, relative):
    """Load ``<root>/<relative>.py`` under a private alias."""
    alias = "_baseline_" + relative.replace("/", "_")
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, root / (relative + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def normalize(value):
    """Render a result comparably: floats rounded, geometries as WKT."""
    if value is None:
        return "None"
    if isinstance(value, float):
        return "nan" if np.isnan(value) else f"{value:.9g}"
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=6, threshold=50)
    if isinstance(value, (pd.Series, pd.Index)):
        return normalize(np.asarray(value, dtype=object))
    if hasattr(value, "wkt"):
        return value.wkt[:200]
    return repr(value)


# --------------------------------------------------------------------------
# H4 - auto_weight_scale: the old body reduced with nanmean while its name,
# docstring and return doc all said median.
# --------------------------------------------------------------------------
@case("H4 auto_weight_scale", "differ", "old computed nanmean, name/docstring said median; H4 made it nanmedian")
def _auto_weight_scale(old_root):
    old_cv = load_baseline(old_root, "create_voronoi")

    rng = np.random.default_rng(0)
    # A deliberately skewed spread, so mean and median cannot coincide by luck.
    points = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(40, 1, (3, 2))]).tolist()
    return old_cv.auto_weight_scale(points), sa.auto_weight_scale(points)


# --------------------------------------------------------------------------
# G6 - is_valid_geom gated its non-finite check behind hasattr(geom, "coords"),
# which only Point/LineString have, so polygons were never checked.
# --------------------------------------------------------------------------
@case(
    "G6 is_valid_geom / polygon with NaN vertex",
    "same",
    "both reject it, but for different reasons: shapely already reports "
    "is_valid=False, so this input never reached either coordinate check",
)
def _is_valid_polygon_nan(old_root):
    old_cv = load_baseline(old_root, "create_voronoi")

    bad = Polygon([(0, 0), (1, 0), (1, float("nan")), (0, 1)])
    return old_cv.is_valid_geom(bad), sa.is_valid_geom(bad)


@case(
    "G6 is_valid_geom / clean polygon",
    "differ",
    "hasattr(polygon, 'coords') raises NotImplementedError under Shapely 2, and "
    "the baseline's blanket `except Exception` turned that into False - so the "
    "baseline rejected EVERY polygon, not merely skipping the finite check",
)
def _is_valid_polygon_clean(old_root):
    old_cv = load_baseline(old_root, "create_voronoi")

    good = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    return old_cv.is_valid_geom(good), sa.is_valid_geom(good)


@case("G6 is_valid_geom / point with NaN", "same", "Point has .coords, so the old guard did run - must be unchanged")
def _is_valid_point_nan(old_root):
    old_cv = load_baseline(old_root, "create_voronoi")

    bad = Point(float("nan"), 0)
    return old_cv.is_valid_geom(bad), sa.is_valid_geom(bad)


@case("G6 is_valid_geom / None and self-intersecting", "same", "the two pre-existing rejection paths must be untouched")
def _is_valid_other(old_root):
    old_cv = load_baseline(old_root, "create_voronoi")

    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    return (
        [old_cv.is_valid_geom(None), old_cv.is_valid_geom(bowtie)],
        [sa.is_valid_geom(None), sa.is_valid_geom(bowtie)],
    )


# --------------------------------------------------------------------------
# H5 - geometry repair. The current geo_utils.repair_geometry short-circuits on
# already-valid input; the baseline variants always went through make_valid.
# --------------------------------------------------------------------------
@case(
    "H5 repair_geometry vs download_and_vectorize.fix_geometry",
    "same",
    "the most complete baseline variant was the one promoted to geo_utils",
)
def _repair_geometry(old_root):
    old_dv = load_baseline(old_root, "industrial_analysis/download_and_vectorize")
    from wwtp.geo_utils import repair_geometry

    old_fn = old_dv._repair_geometry
    inputs = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # valid
        Polygon([(0, 0), (1, 1), (1, 0), (0, 1)]),  # self-intersecting
        Point(3, 4),  # non-polygon
        Polygon(),  # empty
        None,
    ]
    return [normalize(old_fn(g)) for g in inputs], [normalize(repair_geometry(g)) for g in inputs]


# --------------------------------------------------------------------------
# H3 - latest *_zonal_sum resolver, 5 baseline implementations into one.
# Includes the malformed `500_2020_zonal_sum` column the audit flagged.
# --------------------------------------------------------------------------
@case(
    "H3 zonal-sum resolver / well-formed columns",
    "same",
    "consolidated resolver must pick the same column as the five it replaced",
)
def _zonal_sum_clean(old_root):
    old_pf = load_baseline(old_root, "figures_scripts/piechart_figure")
    from wwtp.utils import resolve_latest_zonal_sum_column

    preferred = "population_served_index"
    frames = [
        ["id", "2014_zonal_sum", "2020_zonal_sum", "2024_zonal_sum"],
        ["id", "population_served_index", "2020_zonal_sum"],
        ["id", "2020_zonal_sum"],
    ]
    # The current helper returns (year, column); only the column is the decision.
    return (
        [old_pf.resolve_zonal_sum_columns(pd.DataFrame(columns=c), preferred) for c in frames],
        [resolve_latest_zonal_sum_column(pd.DataFrame(columns=c), preferred)[1] for c in frames],
    )


@case(
    "H3 zonal-sum resolver / radius-prefixed column",
    "differ",
    "find_pop_in_danger_pop emits `500_2020_zonal_sum`; every baseline copy "
    "split on the FIRST underscore and read year 500, so a 2020 layer lost to "
    "a 2014 one. The consolidated parser reads the token before the suffix.",
)
def _zonal_sum_prefixed(old_root):
    old_pf = load_baseline(old_root, "figures_scripts/piechart_figure")
    from wwtp.utils import resolve_latest_zonal_sum_column

    frame = pd.DataFrame(columns=["id", "2014_zonal_sum", "500_2020_zonal_sum"])
    return (
        old_pf.resolve_zonal_sum_columns(frame, "population_served_index"),
        resolve_latest_zonal_sum_column(frame, "population_served_index")[1],
    )


# --------------------------------------------------------------------------
# H8 - robust bounds. pop_at_risk_figures used iqr_factor 1.5 and raised on
# empty; composite used 1.0 and passed empty through. That gap is deliberate.
# --------------------------------------------------------------------------
@case(
    "H8 robust_bounds vs pop_at_risk_figures baseline",
    "same",
    "iqr_factor 1.5 + raise-on-empty semantics must be preserved exactly",
)
def _robust_bounds(old_root):
    old_pf = load_baseline(old_root, "figures_scripts/pop_at_risk_figures")
    from wwtp.utils import robust_bounds

    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(100, 15, 200), [1e6, -5.0, np.nan, np.inf]])
    old_fn = old_pf._robust_bounds
    return (
        [normalize(v) for v in old_fn(values)],
        [normalize(v) for v in robust_bounds(values)],
    )


@case(
    "H8 robust_bounds positive_only",
    "same",
    "the positive-only branch feeds log-scaled axes; a shift moves every marker",
)
def _robust_bounds_positive(old_root):
    old_pf = load_baseline(old_root, "figures_scripts/pop_at_risk_figures")
    from wwtp.utils import robust_bounds

    rng = np.random.default_rng(11)
    values = np.concatenate([rng.lognormal(3, 1, 150), [0.0, -2.0, np.nan]])
    old_fn = old_pf._robust_bounds
    return (
        [normalize(v) for v in old_fn(values, positive_only=True)],
        [normalize(v) for v in robust_bounds(values, positive_only=True)],
    )


@case(
    "H8 clip_to_robust_bounds vs composite baseline",
    "same",
    "composite keeps iqr_factor 1.0 and empty-passthrough - the documented gap",
)
def _clip_outliers(old_root):
    old_ca = load_baseline(old_root, "figures_scripts/composite_area_population_plots")
    from wwtp.utils import clip_to_robust_bounds

    rng = np.random.default_rng(13)
    series = pd.Series(np.concatenate([rng.normal(50, 8, 300), [900.0, -400.0]]))
    old_fn = old_ca.clip_outliers
    return (
        normalize(old_fn(series, 0.005, 0.995).to_numpy()),
        normalize(clip_to_robust_bounds(series, 0.005, 0.995).to_numpy()),
    )


# --------------------------------------------------------------------------
# H7 - piechart size scaling. The interactive variant had no finite guard, so
# NaN reached the SVG; the figure variant used a log scale with a mid-degenerate.
# --------------------------------------------------------------------------
@case(
    "H7 calculate_size vs piechart_figure baseline",
    "same",
    "log scale, mid-degenerate, no non-positive floor - figure variant semantics",
)
def _calculate_size_figure(old_root):
    old_pf = load_baseline(old_root, "figures_scripts/piechart_figure")
    from wwtp.figures_scripts.piechart_figure import calculate_size

    old_fn = old_pf.calculate_size
    trials = [(50, 1, 1000, 5, 40), (1, 1, 1000, 5, 40), (1000, 1, 1000, 5, 40), (10, 10, 10, 5, 40)]
    return ([normalize(old_fn(*t)) for t in trials], [normalize(calculate_size(*t)) for t in trials])


@case(
    "H7 calculate_size NaN guard (interactive)",
    "differ",
    "the interactive baseline let NaN through to the SVG; H7 added the guard",
)
def _calculate_size_nan(old_root):
    old_pi = load_baseline(old_root, "figures_scripts/piechart_interactive")
    from wwtp.figures_scripts.piechart_interactive import calculate_size

    old_fn = old_pi.calculate_size
    return normalize(old_fn(float("nan"), 1, 1000, 5, 40)), normalize(calculate_size(float("nan"), 1, 1000, 5, 40))


def main(baseline_name):
    old_root = SRC / baseline_name
    if not old_root.is_dir():
        sys.exit(f"baseline not found: {old_root}")

    print("# Differential test: baseline vs current\n")
    print(f"Baseline: `{baseline_name}`\n")
    print("| case | expected | observed | verdict |")
    print("|---|---|---|---|")

    details, problems = [], 0
    for name, expect, note, fn in CASES:
        try:
            old_value, new_value = fn(old_root)
            old_text, new_text = normalize(old_value), normalize(new_value)
            observed = "same" if old_text == new_text else "differ"
            if observed == expect:
                verdict = "OK"
            else:
                verdict = "**REGRESSION**" if expect == "same" else "**FIX DID NOT LAND**"
                problems += 1
        except Exception as err:
            old_text = new_text = ""
            observed, verdict = "error", f"**ERROR** {type(err).__name__}: {err}"
            problems += 1
        print(f"| {name} | {expect} | {observed} | {verdict} |")
        details.append((name, note, old_text, new_text, observed))

    print("\n## Recorded values\n")
    for name, note, old_text, new_text, observed in details:
        print(f"### {name}\n")
        print(f"{note}\n")
        print(f"- baseline: `{old_text}`")
        print(f"- current:  `{new_text}`")
        print(f"- observed: **{observed}**\n")

    print(f"\n{len(CASES)} cases, {problems} needing attention.")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: baseline_differential.py <baseline-dir-name-under-wwtp>")
    sys.exit(main(sys.argv[1].rstrip("/")))
