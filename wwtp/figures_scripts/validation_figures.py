"""Validation of the recommended sweep candidate as maps, plots and tables (paper / supplement).

Reads the outputs of ``sweep_validation`` (``sweep_validation.paths.output_dir``): ``choice.csv`` (the
recommended candidate), ``units.parquet`` (people per reference unit), ``references/units.csv``,
``matches.csv`` and ``plants.csv`` (reference people, our plants and their countries),
``national_countries.csv`` and ``national_regions.csv``. Writes to ``validation_figures.paths.output_dir``:

- ``map_plants_<family>.pdf/.png`` for every ``plant_maps`` family: per country, the typical error of
  people per plant against the family's references, for all plants and per size class (reference
  people below 10,000, 10,000-100,000, above 100,000). ``catchments``: real catchments (GB, NL, US;
  sweep_validation); ``reported``: plant numbers from national statistics in our plant layer
  (``register_references``: HydroWASTE POP_SERVED reported (QUAL_POP 1), US CWNS 2022, UWWTD load
  entering), one plant per unit. Reported numbers are also inputs of the plant sizes, so that family
  shows agreement, not an independent test. Numbers in population equivalents (UWWTD loads: 60 g
  BOD5 a day, residents plus industry, commerce, commuters and tourists) are divided by the country's
  p.e. per connected resident (``pe_conversion``; ``pe_per_resident.csv``).
- ``plants_countries.pdf/.png``: the same per country and size class as dots, one panel per family.
- ``map_national_<reference>.pdf/.png`` for every ``national_references``: our share of people inside
  service areas, the reference's share connected to treatment, their difference, and both per
  country.
- ``regions_countries.pdf/.png``: per SDG region, ours minus each national reference; per SDG region
  and family, the typical error per plant by size class; per country, ours against each reference.
- CSV tables: ``countries_plants.csv``, ``regions_plants.csv`` (per-plant measures by size class),
  ``pe_per_resident.csv`` (p.e. per connected resident per country),
  ``countries_national.csv``, ``regions_national.csv``, ``summary_national.csv``, and
  ``plant_units.csv`` (people per reference unit: the input of the plant-level tables, to redraw
  without the parquet file).
- ``validation_results.zip`` with all of the above.

Typical error: ``10 ** median |log10(ours / reference)|`` - half of the plants are closer than this
factor; a plant with nobody in its service area counts as infinitely wrong. Style as the calibration
figures (``calibration_figures``). Command: ``python -m wwtp.figures_scripts.validation_figures``.
"""

from __future__ import annotations

import argparse
import logging
import os
import zipfile

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from shapely.geometry import box  # noqa: E402

from ..calibration.service_radius import SIZE_CLASSES  # noqa: E402
from ..sensitivity_analysis_scripts.sweep_validation import candidates, read_columns, zonal_columns  # noqa: E402
from ..starter import load_config  # noqa: E402
from ..utils import atomic_output, configure_logging, get_iso_codes  # noqa: E402
from .calibration_figures import DOT, MM, SMALL, apply_style, is_true, panel_letter, place_labels, save  # noqa: E402

logger = logging.getLogger(__name__)

SIZE_LABELS = ("all plants", "under 10,000 people", "10,000-100,000 people", "over 100,000 people")
SIZE_KEYS = ("all", "<10k", "10k-100k", ">100k")


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def read_inputs(folder):
    """The recommended candidate's people per unit with countries, and its national tables."""

    def read(name, **kw):
        return pd.read_csv(os.path.join(folder, name), keep_default_na=False, na_values=[""], **kw)

    choice = read("choice.csv")
    picked = choice[is_true(choice["recommended"])]
    if len(picked) != 1:
        raise ValueError(f"choice.csv has {len(picked)} recommended candidates, expected one")
    cand = picked["candidate"].iloc[0]
    units = pd.read_parquet(os.path.join(folder, "units.parquet"), columns=["candidate", "reference", "unit", "pred"])
    units = units[units["candidate"] == cand].drop(columns="candidate")
    units["unit"] = units["unit"].astype(str)
    refs = read("references/units.csv", dtype={"unit": str})
    matches = read("references/matches.csv", dtype={"unit": str})
    plants = read("references/plants.csv", usecols=["SITE_UID", "iso2"])
    national = read("national_countries.csv")
    national = national[national["candidate"] == cand].drop(columns="candidate")
    regions = read("national_regions.csv")
    regions = regions[regions["candidate"] == cand].drop(columns="candidate")
    return cand, plant_units(units, refs, matches, plants, national), national, regions


def register_units(registers, conversion, path, country_col, national):
    """One unit per plant of the candidate's population layer with a reported number (``registers``),
    in residents (``pe_to_residents``)."""
    zonal = zonal_columns(path)
    years = {int(r["year"]) for r in registers.values()}
    missing = sorted(years - set(zonal))
    if missing:
        raise KeyError(f"{path} has no zonal sums for {missing}")
    columns = {"SITE_UID", country_col, *(zonal[y] for y in years)}
    columns |= {r["column"] for r in registers.values()} | {c for r in registers.values() for c in r["only"]}
    layer = read_columns(path, sorted(columns), ignore_geometry=True)
    region = national.drop_duplicates("iso2").set_index("iso2")["region_sdg"]
    parts = []
    for name, r in registers.items():
        value = pd.to_numeric(layer[r["column"]], errors="coerce")
        keep = value > 0
        for column, allowed in r["only"].items():
            keep &= layer[column].isin(allowed)
        part = layer[keep]
        parts.append(
            pd.DataFrame(
                {
                    "reference": name,
                    "unit": part["SITE_UID"].astype(str).to_numpy(),
                    "ref": value[keep].to_numpy(dtype=float),
                    "pred": part[zonal[int(r["year"])]].fillna(0.0).to_numpy(dtype=float),
                    "iso2": part[country_col].astype(str).to_numpy(),
                }
            )
        )
        logger.info("%s: %d plants with a reported number", name, int(keep.sum()))
    out = pd.concat(parts, ignore_index=True)
    out["region_sdg"] = out["iso2"].map(region)
    return pe_to_residents(out, registers, conversion, national)


def pe_to_residents(units, registers, conversion, national):
    """Reported numbers in population equivalents (``unit: pe``) to residents.

    A country's p.e. per connected resident f_c = (sum of its plants' loads) / (its residents connected
    to treatment: people x share of the first ``conversion.national_references`` with a number). Each
    load is divided by f_c. A country below ``min_pe_per_resident`` (residents alone make that many
    p.e.; lower means loads are missing, e.g. a large plant without one) or without a national number
    takes the ratio of all other countries together. Adds ``ref_reported`` (the number as reported),
    ``pe_per_resident`` (f used; empty for numbers in people), ``pe_per_resident_national`` (f_c) and
    ``pe_per_resident_from`` (the national reference or 'other countries')."""
    out = units.assign(ref_reported=units["ref"], pe_per_resident=np.nan, pe_per_resident_national=np.nan,
                       pe_per_resident_from=None)  # fmt: skip
    connected = pd.Series(np.nan, index=pd.Index(national["iso2"].unique(), name="iso2"))
    source = pd.Series(None, index=connected.index, dtype=object)
    for ref in reversed(conversion["national_references"]):  # the first reference with a number wins
        p = national[(national["reference"] == ref) & national["ref_share"].notna()].set_index("iso2")
        residents = p["people"] * p["ref_share"]
        connected.loc[residents.index] = residents
        source.loc[residents.index] = ref
    for name, r in registers.items():
        if r["unit"] != "pe":
            continue
        rows = out["reference"] == name
        load = out[rows].groupby("iso2")["ref"].sum()
        f = load / connected.reindex(load.index)
        valid = f >= float(conversion["min_pe_per_resident"])
        if not valid.any():
            raise ValueError(f"{name}: no country has a national number and at least "
                             f"{conversion['min_pe_per_resident']} p.e. per connected resident")  # fmt: skip
        pooled = load[valid].sum() / connected.reindex(load.index)[valid].sum()
        used = f.where(valid, pooled)
        origin = source.reindex(load.index).where(valid, "other countries")
        iso = out.loc[rows, "iso2"]
        out.loc[rows, "pe_per_resident"] = iso.map(used).to_numpy()
        out.loc[rows, "pe_per_resident_national"] = iso.map(f).to_numpy()
        out.loc[rows, "pe_per_resident_from"] = iso.map(origin).to_numpy()
        out.loc[rows, "ref"] = out.loc[rows, "ref_reported"] / out.loc[rows, "pe_per_resident"]
        logger.info("%s: p.e. per connected resident %.2f (other countries); %s", name, pooled,
                    ", ".join(f"{i} {v:.2f}" for i, v in used.sort_values().items()))  # fmt: skip
    return out


def pe_table(units, names):
    """Per register in p.e. and country: plants, load, p.e. per connected resident and where it is from."""
    pe = units[units["pe_per_resident"].notna()]
    t = (
        pe.groupby(["reference", "iso2"])
        .agg(
            plants=("unit", "size"),
            load_pe=("ref_reported", "sum"),
            residents=("ref", "sum"),
            pe_per_resident=("pe_per_resident", "first"),
            pe_per_resident_national=("pe_per_resident_national", "first"),
            pe_per_resident_from=("pe_per_resident_from", "first"),
        )  # fmt: skip
        .reset_index()
    )
    t.insert(2, "country", [names.get(i, i) for i in t["iso2"]])
    return t


def plant_units(units, refs, matches, plants, national):
    """One row per reference unit: reference, unit, iso2, region, ref (people), pred (ours)."""
    country = matches.merge(plants, on="SITE_UID").drop_duplicates(["reference", "unit"])
    out = refs[["reference", "unit", "ref"]].merge(units, on=["reference", "unit"], how="left")
    out = out.fillna({"pred": 0.0}).merge(country[["reference", "unit", "iso2"]], on=["reference", "unit"], how="left")
    region = national.drop_duplicates("iso2").set_index("iso2")["region_sdg"]
    out["region_sdg"] = out["iso2"].map(region)
    return out.assign(ref_reported=out["ref"], pe_per_resident=np.nan, pe_per_resident_national=np.nan,
                      pe_per_resident_from=None)  # fmt: skip


# --------------------------------------------------------------------------- #
# Measures                                                                    #
# --------------------------------------------------------------------------- #
def size_key(ref):
    return np.array(SIZE_KEYS[1:])[np.digitize(ref, SIZE_CLASSES)]


def plant_measures(df):
    """Measures of ours (pred) against the reference people (ref) for one group of units."""
    pred, ref = df["pred"].to_numpy(float), df["ref"].to_numpy(float)
    err = np.where(pred > 0, np.abs(np.log10(np.maximum(pred, 1.0) / ref)), np.inf)
    return pd.Series(
        {
            "n": len(ref),
            "typical_error_x": 10 ** float(np.median(err)),
            "within_factor_1_5": float(np.mean(err <= np.log10(1.5))),
            "within_factor_2": float(np.mean(err <= np.log10(2))),
            "within_factor_3": float(np.mean(err <= np.log10(3))),
            "median_ratio": float(np.median(pred / ref)),
            "total_ratio": float(pred.sum() / ref.sum()),
        }
    )


def plant_table(units, key, references):
    """Measures per ``key`` (iso2 or region_sdg) and size class, over the units of ``references``
    (a country with several references pools their units)."""
    u = units[units["reference"].isin(references) & units[key].notna()].assign(size=lambda d: size_key(d["ref"]))
    parts = [u.assign(size="all"), u]
    t = pd.concat(parts).groupby([key, "size"]).apply(plant_measures, include_groups=False).reset_index()
    refs = u.groupby(key)["reference"].agg(lambda s: "+".join(sorted(s.unique()))).rename("references")
    t = t.merge(refs, on=key)
    t["size"] = pd.Categorical(t["size"], SIZE_KEYS, ordered=True)
    return t.sort_values([key, "size"]).reset_index(drop=True)


def national_wide(national, names):
    """One row per country: our share (of the JMP comparison) and each reference's share, in %."""
    first = national.drop_duplicates("iso2").set_index("iso2")
    out = first[["iso3", "region_sdg", "people"]].copy()
    out.insert(0, "country", [names.get(i, i) for i in out.index])
    out["ours_pct"] = national.groupby("iso2")["served_share"].first() * 100
    for ref, part in national.groupby("reference"):
        p = part.set_index("iso2")
        out[f"{ref}_pct"] = p["ref_share"] * 100
        out[f"{ref}_diff_pp"] = (p["served_share"] - p["ref_share"]) * 100
    return out.reset_index()


def national_summary(national):
    rows = []
    for ref, p in national.dropna(subset=["ref_share"]).groupby("reference"):
        diff = (p["served_share"] - p["ref_share"]).to_numpy() * 100
        w = p["people"].to_numpy(float)
        rows.append(
            {
                "reference": ref,
                "n_countries": len(p),
                "mean_abs_error_pp": float(np.mean(np.abs(diff))),
                "median_abs_error_pp": float(np.median(np.abs(diff))),
                "people_weighted_abs_error_pp": float(np.sum(np.abs(diff) * w) / w.sum()),
                "bias_pp": float(np.mean(diff)),
                "within_10pp": float(np.mean(np.abs(diff) <= 10)),
                "correlation_r": float(np.corrcoef(p["served_share"], p["ref_share"])[0, 1]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Maps                                                                        #
# --------------------------------------------------------------------------- #
def countries_layer(path, crs, south):
    """Natural Earth countries with ISO alpha-2 codes (ISO_A2_EH: France and Norway have ISO_A2 -99)."""
    g = gpd.read_file(path, columns=["ISO_A2_EH", "NAME"])
    g = g.rename(columns={"ISO_A2_EH": "iso2"})
    g = g.clip(box(-180, south, 180, 90))
    return g.to_crs(crs)


def extent_box(extent, crs):
    lon0, lon1, lat0, lat1 = extent
    return gpd.GeoSeries([box(lon0, lat0, lon1, lat1)], crs=4326).to_crs(crs).total_bounds


def draw_map(ax, world, values, cmap, norm, cfg, extent=None):
    """Countries coloured by ``values`` (Series by iso2); countries without a value in the no-data grey."""
    colors = cfg["colors"]
    g = world.assign(value=world["iso2"].map(values))
    g[g["value"].isna()].plot(ax=ax, color=cfg["nodata_color"], edgecolor=colors["surface"], linewidth=0.15)
    if g["value"].notna().any():
        g[g["value"].notna()].plot(
            ax=ax, column="value", cmap=cmap, norm=norm, edgecolor=colors["surface"], linewidth=0.15
        )
    if extent is not None:
        x0, y0, x1, y1 = extent
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    ax.set_axis_off()


def map_pair(fig, spec, world, values, cmap, norm, cfg):
    """World map with Europe enlarged beside it (GB and NL are a few pixels wide on the world map)."""
    sub = spec.subgridspec(1, 2, width_ratios=cfg["europe_panel_ratio"], wspace=0.02)
    ax_w, ax_e = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    draw_map(ax_w, world, values, cmap, norm, cfg, extent=world.total_bounds)
    x0, y0, x1, y1 = extent_box(cfg["europe_extent"], world.crs)
    draw_map(ax_e, world, values, cmap, norm, cfg, extent=(x0, y0, x1, y1))
    edge = cfg["colors"]["secondary"]
    ax_w.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.4, ec=edge))
    ax_e.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.6, ec=edge, clip_on=False))
    ax_e.text(0.03, 0.97, "Europe", transform=ax_e.transAxes, ha="left", va="top",
              fontsize=cfg["font_size_pt"] - 1.5, color=edge)  # fmt: skip
    return ax_w, ax_e


def label_countries(axes, world, table, cfg):
    """Label each country of ``table`` (index iso2: typical_error_x, n) on the axes whose view holds it."""
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        points, texts = [], []
        for iso2, r in table.iterrows():
            shape = world[world["iso2"] == iso2]
            if shape.empty:
                continue
            pt = shape.geometry.representative_point().iloc[0]
            if x0 <= pt.x <= x1 and y0 <= pt.y <= y1 and (ax is axes[-1] or not _in_any(pt, axes[1:])):
                points.append((pt.x, pt.y))
                texts.append(f"×{r['typical_error_x']:.2f} (n {int(r['n']):,})")
        if points:
            place_labels(ax, points, texts, cfg)


def _in_any(pt, axes):
    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        if x0 <= pt.x <= x1 and y0 <= pt.y <= y1:
            return True
    return False


def error_scale(cfg):
    bins = [1.0, *cfg["error_bins"], np.inf]
    cmap = ListedColormap(cfg["error_colors"])
    return bins, cmap, BoundaryNorm(bins[:-1] + [np.finfo(float).max], cmap.N, clip=True)


def map_plants(plants_by_country, world, family, cfg):
    bins, cmap, norm = error_scale(cfg)
    fig = plt.figure(figsize=(cfg["width_mm"] * MM, cfg["heights_mm"]["plants_map"] * MM))
    grid = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.06)
    for k, (key, label) in enumerate(zip(SIZE_KEYS, SIZE_LABELS)):
        part = plants_by_country[(plants_by_country["size"] == key) & (plants_by_country["n"] >= cfg["min_units"])]
        table = part.set_index("iso2")
        values = table["typical_error_x"].clip(upper=np.finfo(float).max / 2)  # inf: nobody served
        ax_w, ax_e = map_pair(fig, grid[k // 2, k % 2], world, values, cmap, norm, cfg)
        if len(table) <= cfg["max_map_labels"]:  # more would cover the map; the values are in plants_countries
            label_countries([ax_w, ax_e], world, table, cfg)
        map_title(ax_w, "abcd"[k], label, cfg)
    fig.legend(
        [Patch(facecolor=c, edgecolor="none") for c in cfg["error_colors"]] + [Patch(facecolor=cfg["nodata_color"])],
        [_bin_label(lo, hi) for lo, hi in zip(bins[:-1], bins[1:])] + ["no plant reference"],
        loc="upper center",
        ncol=len(cfg["error_colors"]) + 1,
        bbox_to_anchor=(0.5, 0.06),
        handlelength=1.0,
        columnspacing=0.9,
        title=f"Typical error per plant against {cfg['plant_map_titles'][family]}: half of the plants are closer "
        "than this factor",
        title_fontsize=cfg["font_size_pt"] - 0.5,
    )
    save(fig, f"map_plants_{family}", cfg)


def map_title(ax, letter, title, cfg):
    """Bold panel letter and the title on one line above a map."""
    panel_letter(ax, letter, cfg, x=0.035, y=1.0)
    ax.text(0.05, 1.0, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=cfg["font_size_pt"])


def _bin_label(lo, hi):
    return f"over ×{lo:g}" if not np.isfinite(hi) else f"×{lo:g}–{hi:g}"


def diverging_scale(cfg):
    edges = [float(e) for e in cfg["difference_bins_pp"]]
    colors = list(cfg["diverging_colors"])
    bounds = [-1000.0, *edges, 1000.0]
    return bounds, ListedColormap(colors), BoundaryNorm(bounds, len(colors))


def map_national(national, world, reference, extent_name, cfg):
    p = national[(national["reference"] == reference)].dropna(subset=["ref_share"]).set_index("iso2")
    if p.empty:
        logger.warning("no countries with %s", reference)
        return
    colors, name = cfg["colors"], cfg["reference_labels"][reference]
    seq = LinearSegmentedColormap.from_list("share", cfg["share_colors"])
    share_norm = plt.Normalize(0, 100)
    bounds, div, div_norm = diverging_scale(cfg)
    extent = world.total_bounds if extent_name == "world" else extent_box(cfg["europe_extent"], world.crs)
    fig = plt.figure(figsize=(cfg["width_mm"] * MM, cfg["heights_mm"][f"national_map_{extent_name}"] * MM))
    grid = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.08)
    panels = [
        ("Our share: people inside service areas", p["served_share"] * 100, seq, share_norm),
        (f"{name}: share connected to treatment", p["ref_share"] * 100, seq, share_norm),
        (f"Difference, ours − {name} (percentage points)", (p["served_share"] - p["ref_share"]) * 100, div, div_norm),
    ]
    for k, (title, values, cmap, norm) in enumerate(panels):
        if extent_name == "world":
            ax, _ = map_pair(fig, grid[k // 2, k % 2], world, values, cmap, norm, cfg)
        else:
            ax = fig.add_subplot(grid[k // 2, k % 2])
            draw_map(ax, world, values, cmap, norm, cfg, extent=extent)
        map_title(ax, "abc"[k], title, cfg)
        if k == 1:
            cax = ax.inset_axes([0.15, -0.07, 0.7, 0.035])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=share_norm, cmap=seq), cax=cax, orientation="horizontal")
            cb.set_label("% of the country's people (panels a and b)", fontsize=cfg["font_size_pt"] - 1, labelpad=1)
            _thin(cb, cfg)
        if k == 2:
            cax = ax.inset_axes([0.1, -0.07, 0.8, 0.035])
            sm = plt.cm.ScalarMappable(norm=div_norm, cmap=div)
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal", boundaries=bounds[1:-1], spacing="uniform")
            cb.set_ticks(bounds[1:-1])
            cb.set_ticklabels([f"{b:+g}" if b else "0" for b in bounds[1:-1]])
            cb.set_label("ours lower (red) ← → ours higher (blue)", fontsize=cfg["font_size_pt"] - 1, labelpad=1)
            _thin(cb, cfg)

    ax = fig.add_subplot(grid[1, 1])
    x, y = p["ref_share"].to_numpy() * 100, p["served_share"].to_numpy() * 100
    ax.plot([0, 100], [0, 100], color=colors["muted"], lw=0.5)
    ax.fill_between([0, 100], [-10, 90], [10, 110], color=colors["band"], lw=0)
    ax.scatter(x, y, s=DOT, color=colors["series"][0], linewidths=0, zorder=3)
    diff = y - x
    ax.text(0.03, 0.97, f"{len(p)} countries\nmean error {np.mean(np.abs(diff)):.1f} pp\nbias {np.mean(diff):+.1f} pp\n"
            f"r = {np.corrcoef(x, y)[0, 1]:.2f}", transform=ax.transAxes, va="top", ha="left",
            fontsize=cfg["font_size_pt"] - SMALL)  # fmt: skip
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{name}: connected to treatment (%)")
    ax.set_ylabel("Ours: inside service areas (%)")
    ax.text(0.97, 0.03, "grey band: within 10 points", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=cfg["font_size_pt"] - 1.5, color=colors["secondary"])  # fmt: skip
    panel_letter(ax, "d", cfg, x=-0.2, y=1.0)
    save(fig, f"map_national_{reference}", cfg)


def _thin(cb, cfg):
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(width=0.3, length=1.5, labelsize=cfg["font_size_pt"] - 1)


# --------------------------------------------------------------------------- #
# Regions and countries                                                       #
# --------------------------------------------------------------------------- #
def figure_regions_countries(regions, national, plants_by_region, cfg):
    colors = cfg["colors"]
    refs = [r for r in cfg["national_references"] if r in set(regions["reference"])]
    fig = plt.figure(figsize=(cfg["width_mm"] * MM, cfg["heights_mm"]["regions_countries"] * MM))
    rows = fig.add_gridspec(2, 1, hspace=0.38, height_ratios=[1.15, 1.0])
    top = rows[0].subgridspec(1, 2, wspace=0.75)
    bottom = rows[1].subgridspec(1, len(refs), wspace=0.35)

    ax = fig.add_subplot(top[0])
    reg = regions[regions["reference"].isin(refs)]
    order = (
        reg[reg["reference"] == refs[0]].sort_values("ref_share")["region_sdg"].tolist()
        if refs
        else sorted(reg["region_sdg"].unique())
    )
    order = [r for r in order if r != "World"] + ["World"]
    order += [r for r in reg["region_sdg"].unique() if r not in order]
    y = {r: i for i, r in enumerate(order)}
    ax.axvline(0, color=colors["muted"], lw=0.5)
    ax.axvspan(-5, 5, color=colors["band"], lw=0)
    offsets = np.linspace(-0.22, 0.22, max(len(refs), 1))
    for slot, ref in enumerate(refs):
        r = reg[reg["reference"] == ref]
        ax.scatter(r["error_pp"], [y[k] + offsets[slot] for k in r["region_sdg"]], s=DOT + 6,
                   color=colors["series"][slot], linewidths=0, zorder=3, label=cfg["reference_labels"][ref])  # fmt: skip
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o.replace(" and ", " & ") for o in order], fontsize=cfg["font_size_pt"] - 1)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Ours − reference, people-weighted (percentage points)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(refs), handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.1)  # fmt: skip
    ax.text(0.99, 0.01, "grey: within 5 points", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=cfg["font_size_pt"] - 1.5, color=colors["secondary"])  # fmt: skip
    panel_letter(ax, "a", cfg, x=-0.02, y=1.08)

    ax = fig.add_subplot(top[1])
    rows_b = plants_by_region[plants_by_region["n"] >= cfg["min_units"]].copy()
    rows_b["label"] = [
        f"{r.replace(' and ', ' & ')}\n{cfg['plant_map_short'][f]}" for f, r in zip(rows_b["map"], rows_b["region_sdg"])
    ]
    size_dots(ax, rows_b, cfg)
    panel_letter(ax, "b", cfg, x=-0.02, y=1.14)

    for slot, ref in enumerate(refs):
        ax = fig.add_subplot(bottom[slot])
        p = national[national["reference"] == ref].dropna(subset=["ref_share"])
        x, yv = p["ref_share"].to_numpy() * 100, p["served_share"].to_numpy() * 100
        ax.fill_between([0, 100], [-10, 90], [10, 110], color=colors["band"], lw=0)
        ax.plot([0, 100], [0, 100], color=colors["muted"], lw=0.5)
        ax.scatter(x, yv, s=DOT - 1, color=colors["series"][slot], linewidths=0, zorder=3)
        diff = yv - x
        ax.text(0.03, 0.97, f"{len(p)} countries\nmean error {np.mean(np.abs(diff)):.1f} pp\n"
                f"bias {np.mean(diff):+.1f} pp\nr = {np.corrcoef(x, yv)[0, 1]:.2f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=cfg["font_size_pt"] - SMALL)  # fmt: skip
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{cfg['reference_labels'][ref]}: connected (%)")
        if slot == 0:
            ax.set_ylabel("Ours: inside service areas (%)")
        panel_letter(ax, "cdefg"[slot], cfg)
    save(fig, "regions_countries", cfg)


def size_dots(ax, rows, cfg):
    """Typical error per plant as dots by size class (diamond: all plants), one row per ``label``."""
    colors = cfg["colors"]
    labels = list(dict.fromkeys(rows["label"]))
    yy = {lab: i for i, lab in enumerate(labels)}
    ax.axvline(2, color=colors["muted"], lw=0.5, ls=(0, (3, 2)))
    for slot, key in enumerate(SIZE_KEYS):
        part = rows[rows["size"] == key]
        ax.scatter(part["typical_error_x"].clip(upper=cfg["error_axis_max"]), [yy[lab] for lab in part["label"]],
                   s=DOT + (4 if key == "all" else 2), marker="D" if key == "all" else "o",
                   color=colors["ink"] if key == "all" else colors["series"][slot - 1], linewidths=0,
                   zorder=3 + (key == "all"), label=SIZE_LABELS[slot])  # fmt: skip
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=cfg["font_size_pt"] - 1)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xscale("log")
    largest = float(rows["typical_error_x"].replace(np.inf, np.nan).max())
    end = min(cfg["error_axis_max"], max(2.5, largest * 1.15))
    ax.set_xlim(1, end)
    ticks = [t for t in (1, 1.25, 1.5, 2, 3, 4, 6) if t <= end]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"×{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("Typical error per plant (half of the plants are closer)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.1)  # fmt: skip


def figure_plants_countries(by_country, cfg):
    """Typical error per plant by country and size class, one panel per plant-map family."""
    names = get_iso_codes()[3]
    families = list(cfg["plant_maps"])
    rows = by_country[by_country["n"] >= cfg["min_units"]].copy()
    counts = [max(rows[rows["map"] == f]["iso2"].nunique(), 1) for f in families]
    per_row = cfg["heights_mm"]["plants_countries_row"]
    height = sum(counts) * per_row + len(families) * cfg["heights_mm"]["plants_countries_panel"]
    fig = plt.figure(figsize=(cfg["width_mm"] * MM, height * MM))
    grid = fig.add_gridspec(len(families), 1, height_ratios=[c + 4 for c in counts], hspace=0.25)
    for k, family in enumerate(families):
        ax = fig.add_subplot(grid[k])
        part = rows[rows["map"] == family].copy()
        if part.empty:
            ax.set_axis_off()
            continue
        n_all = part[part["size"] == "all"].set_index("iso2")["n"]
        part["label"] = [
            f"{names.get(i, i)} ({' + '.join(cfg['reference_labels'][x] for x in r.split('+'))}, {int(n_all[i]):,})"
            for i, r in zip(part["iso2"], part["references"])
        ]
        part = part.sort_values(["label", "size"])
        size_dots(ax, part, cfg)
        ax.legend().set_visible(k == 0)
        ax.text(1.0, 1.0, cfg["plant_map_titles"][family], transform=ax.transAxes, ha="right", va="bottom",
                fontsize=cfg["font_size_pt"])  # fmt: skip
        panel_letter(ax, "ab"[k], cfg, x=-0.02, y=1.0)
    save(fig, "plants_countries", cfg)


# --------------------------------------------------------------------------- #
# Run                                                                         #
# --------------------------------------------------------------------------- #
def write_csv(frame, name, cfg):
    with atomic_output(os.path.join(cfg["paths"]["output_dir"], name)) as tmp:
        frame.to_csv(tmp, index=False, float_format="%.4g")


def render(cfg, cand, units, national, regions):
    apply_style(cfg)
    out = cfg["paths"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    names = get_iso_codes()[3]
    families = cfg["plant_maps"]
    by_country = pd.concat(
        [plant_table(units, "iso2", refs).assign(map=family) for family, refs in families.items()], ignore_index=True
    )
    by_country.insert(1, "country", [names.get(i, i) for i in by_country["iso2"]])
    by_region = pd.concat(
        [plant_table(units, "region_sdg", refs).assign(map=family) for family, refs in families.items()],
        ignore_index=True,
    )
    used = {r for refs in families.values() for r in refs}
    write_csv(units[units["reference"].isin(used)], "plant_units.csv", cfg)
    write_csv(by_country, "countries_plants.csv", cfg)
    write_csv(by_region, "regions_plants.csv", cfg)
    write_csv(pe_table(units[units["reference"].isin(used)], names), "pe_per_resident.csv", cfg)
    write_csv(national_wide(national, names), "countries_national.csv", cfg)
    write_csv(regions, "regions_national.csv", cfg)
    write_csv(national_summary(national), "summary_national.csv", cfg)

    world = countries_layer(cfg["paths"]["boundaries"], cfg["map_crs"], float(cfg["south_limit_deg"]))
    for family in families:
        map_plants(by_country[by_country["map"] == family], world, family, cfg)
    for ref, extent in cfg["national_references"].items():
        map_national(national, world, ref, extent, cfg)
    figure_plants_countries(by_country, cfg)
    figure_regions_countries(regions, national, by_region, cfg)

    with (
        atomic_output(os.path.join(out, "validation_results.zip")) as tmp,
        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z,
    ):
        for name in sorted(os.listdir(out)):
            if name.endswith((".csv", ".pdf", ".png")):
                z.write(os.path.join(out, name), name)
    logger.info("validation figures for %s: %d plant units, %d countries", cand, len(units), national["iso2"].nunique())


def run(cfg):
    sv = load_config(script_name="sweep_validation")
    cand, units, national, regions = read_inputs(sv["paths"]["output_dir"])
    path = candidates().set_index("candidate").loc[cand, "path"]
    registers = register_units(cfg["register_references"], cfg["pe_conversion"], path, sv["country_output_column"],
                               national)  # fmt: skip
    render(cfg, cand, pd.concat([units, registers], ignore_index=True), national, regions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    run(load_config(script_name="validation_figures"))


if __name__ == "__main__":
    configure_logging()
    main()
