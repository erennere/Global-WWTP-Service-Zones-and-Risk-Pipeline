"""Run the whole pipeline on the mock region with two code versions and compare.

Each version is checked out into its own git worktree, gets its own copy of the
mock world (``wwtp/tests/fixtures/mock_world.py``, same seed) and its own copy of
its ``config.yaml`` with ``data_dir`` pointed at that copy. Every stage then
runs through its real CLI entry point (``python -m <package>.<module>``), exactly as
the SLURM wrappers call it, and the outputs are summarised into one table.
The package is ``wwtp`` from the rename on and ``src`` in older commits; each
checkout's own name is used (``package_of``).

Usage (from the repository root)::

    python -m wwtp.tests.harness.mock_before_after --old-ref 438428d --new-ref HEAD \
        --workdir /tmp/mock_before_after

Writes ``<workdir>/report.md`` and ``<workdir>/metrics.json``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EQUAL_AREA = 6933
PYTHON = sys.executable  # interpreter the stages run with; --python overrides

STAGES = [
    ("create_voronoi", ["create_voronoi", "--approach", "0", "1"]),
    ("add_pop", None),  # once per Voronoi file, see run_version
    ("create_rasters", ["pop_at_risk_river_calculations.create_rasters", "--job-index", "0", "--total-jobs", "1"]),
    ("find_unserved_pop", ["pop_at_risk_river_calculations.find_unserved_pop"]),
    ("assign_rivers_to_basin", ["pop_at_risk_river_calculations.assign_rivers_to_basin", "--max-workers", "2"]),
    ("find_intersection_river", ["pop_at_risk_river_calculations.find_intersection_river", "--max-workers", "2"]),
    ("impact_polygons_pop", ["pop_at_risk_river_calculations.impact_polygons_pop", "--max-workers", "2"]),
    ("find_diff_pop", ["pop_at_risk_river_calculations.find_diff_pop", "--index", "0", "--is-parallel", "true"]),
    ("find_pop_in_danger_pop", ["pop_at_risk_river_calculations.find_pop_in_danger_pop"]),
    ("verification_script", ["pop_validation_scripts.verification_script"]),
    ("hw_comparison", ["pop_validation_scripts.hw_comparison"]),
    ("eu_comparison", ["pop_validation_scripts.eu_comparison"]),
    ("find_unconnected_industrial_areas", ["industrial_analysis.find_unconnected_industrial_areas", "--approach", "1"]),
]

PATH_SCRIPTS = ["create_voronoi", "add_pop", "create_rasters", "find_unserved_pop", "assign_rivers_to_basin",
                "find_intersection_river", "impact_polygons_pop", "find_diff_pop", "find_pop_in_danger_pop",
                "verification_script", "eu_comparison", "find_unconnected_industrial_areas"]


# --------------------------------------------------------------------------- #
# Setup                                                                       #
# --------------------------------------------------------------------------- #
def package_of(code_root):
    """The checkout's package folder: ``wwtp``, or ``src`` before the rename."""
    return "wwtp" if (code_root / "wwtp" / "__init__.py").exists() else "src"


def checkout(ref, dest):
    if dest.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(dest)], cwd=REPO, check=False,
                       capture_output=True)
        shutil.rmtree(dest, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=REPO, check=False, capture_output=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(dest), ref], cwd=REPO, check=True,
                   capture_output=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=dest, check=True, capture_output=True,
                         text=True).stdout.strip()
    return sha


def write_config(code_root, data_dir, out_path, settings=None):
    """The version's own config with data_dir pointed at the mock copy and 2 workers.

    ``settings`` maps ``"section.key"`` (or ``"section.sub.key"``) to a value and
    is applied on top, e.g. ``{"create_voronoi.area_ceiling": False}``.
    """
    text = (code_root / package_of(code_root) / "config.yaml").read_text()
    text, n = re.subn(r"^(\s*)data_dir: '\.\./data'", rf"\1data_dir: '{data_dir}'", text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError(f"Could not set data_dir in {code_root}/{package_of(code_root)}/config.yaml")
    text = re.sub(r"^(\s*\w*(?:max_workers|_workers|concurrency)): \d+", r"\1: 2", text, flags=re.M)
    if settings:
        import yaml

        cfg = yaml.safe_load(text)
        for dotted, value in settings.items():
            *parents, key = dotted.split(".")
            node = cfg
            for part in parents:
                node = node[part]
            if key not in node:
                raise KeyError(f"{dotted} is not in {code_root}/{package_of(code_root)}/config.yaml")
            node[key] = value
        text = yaml.safe_dump(cfg, sort_keys=False)
    out_path.write_text(text)
    return out_path


def env_for(code_root, cfg_path):
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(code_root),
        "WWTP_SERVICE_PIPELINE_CONFIG": str(cfg_path),
        "MPLBACKEND": "Agg",
        "PYTHONHASHSEED": "0",
    })
    return env


def resolve_paths(code_root, cfg_path):
    snippet = textwrap.dedent(f"""
        import json
        from {package_of(code_root)}.starter import load_config
        out = {{}}
        for name in {PATH_SCRIPTS!r}:
            try:
                cfg = load_config(script_name=name)
                out[name] = {{k: v for k, v in cfg['paths'].items() if isinstance(v, str)}}
                out[name]['__figures_approach'] = str((cfg.get('figures') or {{}}).get('approach', ''))
                for key in ('eu_reference_factor', 'threshold'):
                    if key in cfg:
                        out[name]['__' + key] = cfg[key]
            except Exception as err:
                out[name] = {{'__error': repr(err)}}
        print(json.dumps(out, default=str))
    """)
    res = subprocess.run([PYTHON, "-c", snippet], cwd=code_root, env=env_for(code_root, cfg_path),
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


def run_module(code_root, cfg_path, args, log_dir, label):
    start = time.time()
    module, *rest = args
    res = subprocess.run([PYTHON, "-m", f"{package_of(code_root)}.{module}", *rest], cwd=code_root, env=env_for(code_root, cfg_path),
                         capture_output=True, text=True, timeout=3600)
    (log_dir / f"{label}.log").write_text(res.stdout + "\n--- stderr ---\n" + res.stderr)
    err_lines = [ln for ln in res.stderr.splitlines() if ln.strip()]
    return {"returncode": res.returncode, "seconds": round(time.time() - start, 1),
            "error": (err_lines[-1][:300] if res.returncode and err_lines else "")}


def file_hash(path):
    if not path or not os.path.exists(path):
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _gpkgs(directory, pattern="*.gpkg"):
    return sorted(p for p in glob.glob(os.path.join(directory or "", pattern)) if not os.path.basename(p).startswith("temp_"))


def _area_km2(gdf):
    if gdf is None or gdf.empty:
        return 0.0
    return float(gdf.to_crs(EQUAL_AREA).geometry.area.sum() / 1e6)


def _latest_zonal_col(df):
    cols = sorted((c for c in df.columns if str(c).endswith("_zonal_sum") and str(c)[:4].isdigit()), key=lambda c: int(c[:4]))
    return cols[-1] if cols else None


def voronoi_metrics(path, sites_total):
    gdf = gpd.read_file(path)
    m = {"regions": len(gdf), "area_km2": round(_area_km2(gdf), 1)}
    if len(gdf):
        areas = gdf.to_crs(EQUAL_AREA).geometry.area
        m["median_region_km2"] = round(float(areas.median() / 1e6), 2)
        if "area" in gdf.columns:
            rel = (gdf["area"] - areas).abs() / areas.replace(0, np.nan)
            m["area_column_off_by_>2%"] = int((rel > 0.02).sum())
        if "POP_SERVED" in gdf.columns:
            m["sum_POP_SERVED"] = int(gdf["POP_SERVED"].fillna(0).sum())
        overlap = gdf.geometry.union_all().area
        m["overlapping_area_%"] = round(100 * (1 - overlap / gdf.geometry.area.sum()), 2) if gdf.geometry.area.sum() else 0.0
    m["input_plants"] = sites_total  # duplicates within the clustering threshold share one region
    return m


def pop_metrics(path):
    gdf = gpd.read_file(path)
    col = _latest_zonal_col(gdf)
    return {"rows": len(gdf), "year_col": col,
            "population": round(float(gdf[col].sum()), 0) if col else None,
            "rows_without_population": int(gdf[col].isna().sum()) if col else len(gdf)}


def table_sum(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path) if path.endswith(".csv") else gpd.read_file(path)
    num = df.select_dtypes("number")
    pop_cols = [c for c in num.columns if "pop" in c.lower() or c.lower().endswith("_sum") or c.lower() == "sum"]
    return {"rows": len(df), **{f"sum[{c}]": round(float(num[c].sum()), 1) for c in pop_cols[:4]}}


def hw_eu_metrics(code_root, cfg_path, pop_file, eu_ref, factor, threshold):
    """HW and EU agreement scores, computed with the version's own comparison code."""
    snippet = textwrap.dedent(f"""
        import json, math
        import {package_of(code_root)}.sensitivity_analysis_scripts.compare_pop_sweep_hw_eu as c
        c._init_summary_worker({eu_ref!r}, {factor!r}, {threshold!r})
        record = dict(alias='a', filepath={pop_file!r}, filename='f', version='2', level='8', buffer='x',
                      weight_type='w', weight_func='', approach='1', only_round=False)
        try:
            rows = c._process_single_record(record)
            out = {{r['source']: {{k: (None if isinstance(v, float) and math.isnan(v) else v)
                                  for k, v in r.items() if k in ('n', 'median_ndi', 'median_abs_ndi', 'median_alpha',
                                                                 'spearman_r')}} for r in rows}}
        except Exception as err:
            out = {{'error': repr(err)[:300]}}
        print(json.dumps(out, default=float))
    """)
    res = subprocess.run([PYTHON, "-c", snippet], cwd=code_root, env=env_for(code_root, cfg_path),
                         capture_output=True, text=True)
    lines = res.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else {"error": res.stderr.strip().splitlines()[-1][:300]}


def collect(code_root, cfg_path, paths, data_dir, sites_total, hashes):
    out = {}
    cv = paths["create_voronoi"]
    for f in _gpkgs(cv.get("buffers_dir")):
        out[f"buffers/{os.path.basename(f).split('_v')[0]}"] = {"groups": len(gpd.read_file(f)),
                                                                 "area_km2": round(_area_km2(gpd.read_file(f)), 1)}
    for f in _gpkgs(cv.get("voronoi_dir")):
        out[f"voronoi/{os.path.basename(f).split('_v')[0]}"] = voronoi_metrics(f, sites_total)
    for f in _gpkgs(paths["add_pop"].get("pop_output_dir")):
        out[f"add_pop/{os.path.basename(f).replace('pop_added_', '').split('_v')[0]}"] = pop_metrics(f)
    cr = paths["create_rasters"]
    out["create_rasters/country_stats"] = table_sum(cr.get("csv_output_filepath", "").replace(".gpkg", ".csv"))
    out["create_rasters/non_served"] = table_sum(cr.get("non_served_outpath", "").replace(".gpkg", ".csv"))
    fu = paths["find_unserved_pop"]
    out["find_unserved_pop/above_threshold"] = table_sum(fu.get("non_served_above_threshold_outpath"))
    ar = paths["assign_rivers_to_basin"].get("rivershed_output_path")
    if ar and os.path.exists(ar):
        r = gpd.read_file(ar)
        out["assign_rivers_to_basin"] = {"segments": len(r), "with_basin": int(r["HYBAS_ID"].notna().sum()) if "HYBAS_ID" in r else 0}
    fi = paths["find_intersection_river"].get("non_served_nxt_river_outpath")
    if fi and os.path.exists(fi):
        g = gpd.read_file(fi)
        out["find_intersection_river"] = {"rows": len(g), "with_river": int(g["NXT_DIS"].notna().sum()) if "NXT_DIS" in g else 0}
    ip = paths["impact_polygons_pop"].get("impact_pop_polygons_outpath", "")
    for f in sorted(glob.glob(ip.replace(".gpkg", "_*.gpkg"))):
        g = gpd.read_file(f)
        out[f"impact_polygons/r{f.rsplit('_', 1)[-1].replace('.gpkg', '')}"] = {"polygons": len(g), "area_km2": round(_area_km2(g), 1)}
    pr = paths["find_pop_in_danger_pop"].get("pop_at_risk_output_filepath")
    if pr and os.path.exists(pr):
        df = pd.read_parquet(pr)
        num = df.select_dtypes("number")
        out["pop_at_risk"] = {"rows": len(df), **{f"sum[{c}]": round(float(num[c].sum()), 1)
                                                   for c in num.columns if "pop" in c.lower() or "sum" in c.lower()}}
    for f in _gpkgs(paths["find_diff_pop"].get("pop_dif_output_dir")):
        out[f"find_diff_pop/{os.path.basename(f)[:40]}"] = pop_metrics(f)
    vd = paths["verification_script"].get("verification_dir")
    for f in _gpkgs(vd):
        name = os.path.basename(f)
        if "appr_1" in name and "only_round" not in name:
            out[f"verification/{name.split('_pop_added')[0]}"] = {"rows": len(gpd.read_file(f))}
    out["comparison_plots"] = {"png_files": len(glob.glob(os.path.join(vd or "", "plots", "**", "*.png"), recursive=True))}
    eu = paths["eu_comparison"]
    pop1 = [f for f in _gpkgs(paths["add_pop"].get("pop_output_dir")) if "appr_1_v" in os.path.basename(f)]
    if pop1:
        out["validation_scores"] = hw_eu_metrics(code_root, cfg_path, pop1[0], eu.get("eu_ref_filepath"),
                                                 eu.get("__eu_reference_factor", 1), eu.get("__threshold", 500))
    ind = paths["find_unconnected_industrial_areas"].get("industrial_unconnected_output")
    if ind and os.path.exists(ind):
        g = gpd.read_file(ind)
        out["industrial/unconnected"] = {"areas": len(g), "area_km2": round(_area_km2(g), 2)}
    out["industrial/main_voronoi_overwritten"] = {"changed": hashes["before"] != hashes["after"]}
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
def run_version(label, ref, workdir, seed, settings=None):
    from wwtp.tests.fixtures.mock_world import build_mock_world

    code_root = workdir / f"code_{label}"
    sha = checkout(ref, code_root)
    data_dir = workdir / f"data_{label}"
    shutil.rmtree(data_dir, ignore_errors=True)
    import yaml

    level = (settings or {}).get("combine_watersheds.level") or \
        yaml.safe_load((code_root / package_of(code_root) / "config.yaml").read_text())["combine_watersheds"]["level"]
    build_mock_world(data_dir, seed=seed, level=level)
    sites_total = len(gpd.read_file(data_dir / "all_annotated_v2.gpkg"))
    cfg_path = write_config(code_root, data_dir, workdir / f"config_{label}.yaml", settings)
    log_dir = workdir / f"logs_{label}"
    shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True)
    paths = resolve_paths(code_root, cfg_path)

    status, hashes = {}, {}
    main_voronoi = os.path.join(paths["create_voronoi"].get("voronoi_dir", ""), "")
    for name, args in STAGES:
        if name == "find_unconnected_industrial_areas":
            appr1 = [f for f in _gpkgs(main_voronoi) if os.path.basename(f).startswith("appr_1_v")]
            hashes["before"] = file_hash(appr1[0]) if appr1 else None
        if name == "add_pop":
            files = _gpkgs(paths["add_pop"].get("voronoi_dir"))
            results = [run_module(code_root, cfg_path, ["add_pop", "--index", str(i)], log_dir, f"add_pop_{i}")
                       for i in range(len(files))]
            bad = [r for r in results if r["returncode"]]
            status[name] = {"returncode": bad[0]["returncode"] if bad else 0,
                            "seconds": round(sum(r["seconds"] for r in results), 1),
                            "error": bad[0]["error"] if bad else "", "files": len(files)}
        else:
            status[name] = run_module(code_root, cfg_path, args, log_dir, name)
        print(f"[{label}] {name}: rc={status[name]['returncode']} {status[name]['seconds']}s {status[name]['error'][:120]}",
              flush=True)
    appr1 = [f for f in _gpkgs(main_voronoi) if os.path.basename(f).startswith("appr_1_v")]
    hashes["after"] = file_hash(appr1[0]) if appr1 else None
    metrics = collect(code_root, cfg_path, paths, str(data_dir), sites_total, hashes)
    return {"ref": ref, "sha": sha, "sites": sites_total, "settings": settings or {}, "status": status,
            "metrics": metrics}


def _fmt(value):
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.0f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return "—" if value is None else str(value)


def _delta(old, new):
    if isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool) and old:
        return f"{100 * (new - old) / abs(old):+.1f} %"
    return "" if old == new else "changed"


def report(result):
    labels = ["old", "new"] + [k for k in result.get("variants", []) if k in result]
    runs = [result[k] for k in labels]
    head = " | ".join(labels)
    lines = [f"# Before/after on the mock region (seed {result['seed']})", ""]
    for label, run in zip(labels, runs):
        extra = f", settings {run['settings']}" if run.get("settings") else ""
        lines.append(f"- **{label}**: `{run['ref']}` ({run['sha']}){extra}")
    lines += ["", f"{runs[0]['sites']} mock plants.", "", "## Stage status", "",
              f"| Stage | {head} |", "|---" * (len(labels) + 1) + "|"]
    cell = lambda st: ("ok" if st.get("returncode") == 0 else f"FAILED: {st.get('error', '')[:90]}")
    for name, _ in STAGES:
        lines.append(f"| {name} | " + " | ".join(cell(r["status"].get(name, {})) for r in runs) + " |")
    lines += ["", "## Outputs", "", "Change = new vs old.", "",
              f"| Output | Metric | {head} | Change |", "|---" * (len(labels) + 3) + "|"]
    keys = list(dict.fromkeys(k for r in runs for k in r["metrics"]))
    for key in keys:
        blocks = [r["metrics"].get(key) or {} for r in runs]
        flat = []
        for blk in blocks:
            row = {}
            for metric, value in blk.items():
                if isinstance(value, dict):
                    row.update({f"{metric}/{m2}": v2 for m2, v2 in value.items()})
                else:
                    row[metric] = value
            flat.append(row)
        for metric in dict.fromkeys(m for row in flat for m in row):
            values = [row.get(metric) for row in flat]
            lines.append(f"| {key} | {metric} | " + " | ".join(_fmt(v) for v in values)
                         + f" | {_delta(values[0], values[1])} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old-ref", default="438428d")
    parser.add_argument("--new-ref", default="HEAD")
    parser.add_argument("--workdir", default="/tmp/mock_before_after")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default=None, help="Comma-separated run labels to (re)run; others are kept")
    parser.add_argument("--variant", action="append", default=[], metavar="LABEL:section.key=value[,...]",
                        help="Extra run of --new-ref with config changes, to attribute a difference, "
                             "e.g. 'fp:create_voronoi.size_measure=footprint'")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for the stages, e.g. a venv that matches the cluster's package versions")
    args = parser.parse_args()
    global PYTHON
    PYTHON = args.python
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    result_path = workdir / "metrics.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result["seed"] = args.seed
    import yaml

    runs = [("old", args.old_ref, None), ("new", args.new_ref, None)]
    for spec in args.variant:
        label, _, assignments = spec.partition(":")
        settings = {k: yaml.safe_load(v) for k, _, v in (a.partition("=") for a in assignments.split(","))}
        runs.append((label, args.new_ref, settings))
    result["variants"] = [label for label, _, settings in runs if settings]
    for label, ref, settings in runs:
        if args.only and label not in args.only.split(","):
            continue
        result[label] = run_version(label, ref, workdir, args.seed, settings)
        result_path.write_text(json.dumps(result, indent=1, default=str))
    if "old" in result and "new" in result:
        (workdir / "report.md").write_text(report(result))
        print(f"Report: {workdir / 'report.md'}")


if __name__ == "__main__":
    main()
