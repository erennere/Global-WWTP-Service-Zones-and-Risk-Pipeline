"""Figures and numbers of the calibration for the paper (Methods / Extended Data).

Reads the tables of the Snakemake target ``calibration`` (``connection_radius.paths.output_dir``, the
same files as in ``calibration_results.zip``) and writes to ``calibration_figures.paths.output_dir``:

- ``ed_plant_size.pdf/.png``: plant size in people. a, reported people against detected tank area
  with the world-wide fit; b, the correction per country; c, the national size scale against the
  share of people connected to treatment.
- ``ed_service_radius.pdf/.png``: the radius scale from the national connection rate. a, share of
  people inside service areas against the radius scale for example countries, and the radius each
  needs to reach its JMP share; b, that needed radius against the connection rate for every country,
  with the chosen rule; c and d, share inside service areas (each region predicted from the others)
  against JMP (fitted) and SDG 6.3.1 (not fitted), fixed radius and rule; e, the same errors per SDG
  region.
- ``ed_catchments.pdf/.png``: the check on real catchments. a-d, people per catchment against the
  reference with the chosen configuration; e, per-plant against national error for every
  configuration, the fixed radius and the choice.
- ``calibration_numbers.tex``: the numbers the Methods text quotes, as LaTeX macros
  (``paper/methods_calibration.tex``).

Layout: Nature Extended Data (``width_mm`` wide, ``font_size_pt`` text, bold lowercase panel letters),
vector PDF and a ``dpi`` PNG; marker sizes and line widths below are fixed style constants (points).
Command: ``python -m wwtp.figures_scripts.calibration_figures``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from ..calibration.connection_radius import logit, needed_radius, share_at  # noqa: E402
from ..calibration.service_radius import class_balanced_error  # noqa: E402
from ..starter import load_config  # noqa: E402
from ..utils import atomic_output, configure_logging, get_iso_codes  # noqa: E402

logger = logging.getLogger(__name__)

MM = 1 / 25.4  # inches per millimetre
WORLDPOP_CELL_KM = 0.1  # WorldPop rasters have 100 m cells (connection_radius.aggregate_cells sums them)
DOT = 7  # scatter marker area (pt^2)
SMALL = 1.0  # points smaller than the body text for annotations


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #
def read_tables(folder):
    """The calibration tables (as written by connection_radius and plant_check)."""

    def read(name, **kw):
        return pd.read_csv(os.path.join(folder, name), keep_default_na=False, na_values=[""], **kw)

    return {
        "countries": read("countries.csv"),
        "curves": read("served_share_curves.csv"),
        "plants": read("plants.csv", dtype={"uwwCode": str, "cwns": str}),
        "references": read("reference_table.csv"),
        "sizes": read("plant_size_countries.csv"),
        "by_cap": read("calibration_by_cap.csv"),
        "by_country": read("calibration_countries.csv"),
        "plant_check": read("plant_check.csv"),
        "plant_status": read("plant_check_status.csv"),
        "districts": read("district_check.csv"),
        "plant_units": read("plant_units.csv.gz", dtype={"unit": str}),
        "plant_references": read("plant_references.csv", dtype={"unit": str}),
        "choice": read("plant_choice.csv"),
        "configurations": read("plant_configurations.csv"),
    }


def is_true(column):
    return column.astype(str).str.lower().isin({"true", "1"})


def chosen(tables, measure):
    """The configuration plant_check chose for ``measure`` (all reference countries)."""
    c = tables["choice"]
    row = c[(c["size_measure"] == measure) & (c["left_out"].fillna("") == "")]
    if len(row) != 1:
        raise ValueError(f"plant_choice.csv has no single choice for {measure}")
    row = row.iloc[0]
    if row["choice"] != "rule":
        raise ValueError(f"plant_check kept the fixed radius for {measure}; these figures show the rule")
    return row


def same_configuration(frame, choice):
    return (
        (frame["size_measure"] == choice["size_measure"])
        & np.isclose(frame["scale_min"], choice["scale_min"])
        & np.isclose(frame["scale_max"], choice["scale_max"])
        & np.isclose(frame["cap_km"], choice["cap_km"])
    )


def fitted_plants(plants):
    """Plants with detected tanks and a reported size: the points of the size fit."""
    return plants[is_true(plants["sized"]) & plants["reported"].notna() & (plants["reported"] > 0)]


def country_curves(tables, measure, exponent, checks):
    """Per country with a JMP share: radii, served share (non-decreasing), people, region, and the JMP
    share and the shares of the ``checks`` references (NaN where a reference has no value)."""
    refs = tables["references"].dropna(subset=["jmp_share", "region_sdg"]).set_index("iso3")
    info = tables["countries"]
    info = info[(info["status"] == "ok") & info["iso3"].isin(refs.index)].set_index("iso2")
    cu = tables["curves"]
    cu = cu[(cu["size_measure"] == measure) & np.isclose(cu["radius_exponent"], exponent)]
    out = {}
    for iso2, c in cu.groupby("iso2"):
        if iso2 not in info.index:
            continue
        c = c.sort_values("radius_at_10k_people_km")
        people = float(info.loc[iso2, "total_people"])
        ref = refs.loc[info.loc[iso2, "iso3"]]
        out[iso2] = {
            "radii": c["radius_at_10k_people_km"].to_numpy(),
            "shares": np.maximum.accumulate(c["served_people"].to_numpy() / people),
            "people": people,
            "region": ref["region_sdg"],
            **{name: float(ref[f"{name}_share"]) for name in ["jmp", *checks]},
        }
    return out


def national_shares(curves, choice, exponent, fixed_r10k, by_country):
    """Per country: share served with the fixed radius (no scale), and with the chosen rule (its region
    left out, from calibrate), the radius it needs and its size scale."""
    detail = by_country[same_configuration(by_country, choice) & np.isclose(by_country["radius_exponent"], exponent)]
    detail = detail.set_index("iso2")
    rows = []
    for iso2, c in curves.items():
        if iso2 not in detail.index:
            continue
        rows.append(
            {
                "iso2": iso2,
                "region": c["region"],
                **{name: value for name, value in c.items() if name not in ("radii", "shares", "people", "region")},
                "people": c["people"],
                "fixed": share_at(c["radii"], c["shares"], fixed_r10k),
                "rule": float(detail.loc[iso2, "served_share_region_left_out"]),
                "needed": float(detail.loc[iso2, "needed_r10k"]),
                # as calibrate: a country whose unscaled curve never crosses its share is not fitted
                "inside": needed_radius(c["radii"], c["shares"], c["jmp"])[1] == "ok",
            }
        )
    return pd.DataFrame(rows)


def mean_abs_pp(pred, ref):
    """Mean absolute difference in percentage points over the countries with a reference."""
    keep = np.isfinite(ref)
    return float(np.mean(np.abs(pred[keep] - ref[keep])) * 100)


def fixed_national_error(shares, national):
    return float(np.mean([mean_abs_pp(shares["fixed"].to_numpy(), shares[n].to_numpy()) for n in national]))


# --------------------------------------------------------------------------- #
# Style                                                                       #
# --------------------------------------------------------------------------- #
def apply_style(cfg):
    colors, size = cfg["colors"], cfg["font_size_pt"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(cfg["font_family"]),
            "font.size": size,
            "axes.titlesize": size,
            "axes.labelsize": size,
            "xtick.labelsize": size - 0.5,
            "ytick.labelsize": size - 0.5,
            "legend.fontsize": size - 0.5,
            "axes.linewidth": 0.4,
            "axes.edgecolor": colors["secondary"],
            "axes.labelcolor": colors["ink"],
            "text.color": colors["ink"],
            "xtick.color": colors["secondary"],
            "ytick.color": colors["secondary"],
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.minor.size": 1.2,
            "ytick.minor.size": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "lines.linewidth": 0.9,
            "mathtext.default": "regular",
            "pdf.fonttype": 42,  # editable text in the PDF
            "svg.fonttype": "none",
        }
    )


def panel_letter(ax, letter, cfg, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=cfg["font_size_pt"] + 2, fontweight="bold", color=cfg["colors"]["ink"])  # fmt: skip


def log_ticks(ax, axis, values):
    """Plain-number ticks on a log axis (no minor labels)."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(values))
    target.set_minor_locator(NullLocator())
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([f"{v:g}" for v in values])


def logit_axis(ax, ticks_percent):
    """x axis in logit of a share, labelled in percent."""
    ticks = [logit(p / 100) for p in ticks_percent]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{p:g}" for p in ticks_percent])
    ax.set_xlim(ticks[0] - 0.3, ticks[-1] + 0.3)


def place_labels(ax, points, texts, cfg):
    """Label points, moving a label to the next free side when it would overlap a placed one."""
    fig = ax.figure
    fig.canvas.draw()
    placed = []
    halo = [patheffects.withStroke(linewidth=1.5, foreground=cfg["colors"]["surface"])]
    offsets = [(3, 0, "left", "center"), (0, 4, "center", "bottom"), (0, -4, "center", "top"),
               (-3, 0, "right", "center"), (3, 4, "left", "bottom"), (3, -4, "left", "top")]  # fmt: skip
    for (x, y), text in zip(points, texts):
        for dx, dy, ha, va in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, zorder=5,
                            fontsize=cfg["font_size_pt"] - SMALL, color=cfg["colors"]["secondary"], path_effects=halo)  # fmt: skip
            box = t.get_window_extent(fig.canvas.get_renderer()).expanded(1.05, 1.1)
            if not any(box.overlaps(b) for b in placed):
                placed.append(box)
                break
            t.remove()
        else:
            ax.annotate(text, (x, y), xytext=offsets[0][:2], textcoords="offset points", ha="left", va="center",
                        zorder=5, fontsize=cfg["font_size_pt"] - SMALL, color=cfg["colors"]["secondary"],
                        path_effects=halo)  # fmt: skip


def save(fig, name, cfg):
    out = cfg["paths"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        with atomic_output(os.path.join(out, f"{name}.{ext}")) as tmp:
            fig.savefig(tmp, format=ext, dpi=int(cfg["dpi"]), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    logger.info("%s written", name)


# --------------------------------------------------------------------------- #
# Figure: plant size                                                          #
# --------------------------------------------------------------------------- #
def figure_plant_size(tables, voronoi, cfg):
    colors, measure = cfg["colors"], cfg["size_measure"]
    f = voronoi["footprint_to_people"]
    fit = fitted_plants(tables["plants"])
    iso3 = get_iso_codes()[1]
    fig = plt.figure(figsize=(cfg["width_mm"] * MM, cfg["heights_mm"]["plant_size"] * MM))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.75, 1.0], wspace=0.5)

    ax = fig.add_subplot(grid[0])
    hb = ax.hexbin(fit["footprint_m2"], fit["reported"], xscale="log", yscale="log", gridsize=40, mincnt=1,
                   cmap=cfg["sequential_cmap"], norm=LogNorm(), linewidths=0.0)  # fmt: skip
    x = np.logspace(np.log10(fit["footprint_m2"].quantile(0.001)), np.log10(fit["footprint_m2"].quantile(0.999)), 50)
    ax.plot(x, 10 ** (f["intercept"] + f["slope"] * np.log10(x)), color=colors["ink"], lw=1.0)
    ax.set_xlabel("Detected tank area, F (m²)")
    ax.set_ylabel("Reported people served, P")
    ax.text(0.03, 0.97, f"log$_{{10}}$P = {f['intercept']:.3f} + {f['slope']:.2f} log$_{{10}}$F\n"
            f"n = {len(fit):,} plants", transform=ax.transAxes, va="top", ha="left")  # fmt: skip
    cax = ax.inset_axes([0.62, 0.1, 0.32, 0.035])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label("plants per cell", fontsize=cfg["font_size_pt"] - SMALL, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_linewidth(0.3)
    cb.ax.tick_params(width=0.3, length=1.5, labelsize=cfg["font_size_pt"] - SMALL)
    panel_letter(ax, "a", cfg)

    ax = fig.add_subplot(grid[1])
    s = tables["sizes"].dropna(subset=["offset"]).sort_values("offset").reset_index(drop=True)
    y = np.arange(len(s))
    factor = 10 ** s["offset"]
    ax.axvline(1.0, color=colors["muted"], lw=0.5)
    ax.hlines(y, 1.0, factor, color=colors["grid_ink"], lw=0.6)
    ax.scatter(factor, y, s=DOT + 2, color=colors["series"][0], zorder=3, linewidths=0)
    ax.set_xscale("log")
    log_ticks(ax, "x", [0.3, 0.5, 1, 2, 5, 10])
    ax.set_xlim(0.3, 12)
    ax.set_yticks(y)
    ax.set_yticklabels([iso3[c] for c in s["country"]], fontsize=cfg["font_size_pt"] - 1.5)
    ax.set_ylim(-0.8, len(s) - 0.2)
    for yy, n in zip(y, s["n_fit"]):
        ax.text(1.0, yy, f"{int(n):,}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=cfg["font_size_pt"] - 2, color=colors["muted"])  # fmt: skip
    ax.text(1.0, len(s) - 0.2, "plants", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=cfg["font_size_pt"] - 2, color=colors["muted"])  # fmt: skip
    ax.set_xlabel("Country correction, 10$^{δ_c}$\n(median reported / estimated)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    panel_letter(ax, "b", cfg, x=-0.12)

    ax = fig.add_subplot(grid[2])
    limits = chosen(tables, measure)
    bc = tables["by_country"]
    bc = bc[(bc["size_measure"] == measure)].drop_duplicates("iso2").dropna(subset=["size_scale_unlimited"])
    lo, hi = float(limits["scale_min"]), float(limits["scale_max"])
    if lo < 1 or hi > 1:
        ax.axhspan(1e-4, lo, color=colors["band"], lw=0)
        ax.axhspan(hi, 1e5, color=colors["band"], lw=0)
    ax.axhline(1.0, color=colors["muted"], lw=0.5)
    x = logit(bc["reference"].to_numpy())
    ax.scatter(x, bc["size_scale_unlimited"], s=DOT, color=colors["series"][0], linewidths=0, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 3000)
    log_ticks(ax, "y", [0.01, 0.1, 1, 10, 100, 1000])
    logit_axis(ax, [1, 5, 20, 50, 80, 95, 99])
    ax.set_xlabel("Population connected to treatment, JMP, C$_c$ (%, logit scale)")
    ax.set_ylabel("National size scale before limits, k$_c$")
    lab = bc["iso2"].isin(cfg["label_countries"]).to_numpy()
    place_labels(ax, list(zip(x[lab], bc["size_scale_unlimited"][lab])), [iso3[c] for c in bc["iso2"][lab]], cfg)
    ax.text(0.98, 0.03, f"grey: outside the chosen limits {lo:.2g}–{hi:g}" if lo < 1 or hi > 1
            else "chosen: no national scale", transform=ax.transAxes, ha="right", va="bottom",
            color=colors["secondary"], fontsize=cfg["font_size_pt"] - SMALL)  # fmt: skip
    panel_letter(ax, "c", cfg)
    save(fig, "ed_plant_size", cfg)


# --------------------------------------------------------------------------- #
# Figure: service radius                                                      #
# --------------------------------------------------------------------------- #
def figure_service_radius(tables, voronoi, cfg, shares, curves, fixed_r10k):
    colors, measure = cfg["colors"], cfg["size_measure"]
    choice = chosen(tables, measure)
    iso3 = get_iso_codes()[1]
    fig = plt.figure(figsize=(cfg["width_mm"] * MM, cfg["heights_mm"]["service_radius"] * MM))
    rows = fig.add_gridspec(2, 1, hspace=0.42, height_ratios=[1.0, 0.95])
    top = rows[0].subgridspec(1, 2, wspace=0.28)
    bottom = rows[1].subgridspec(1, 3, wspace=0.45, width_ratios=[1, 1, 1.05])

    ax = fig.add_subplot(top[0])
    for slot, code in enumerate(cfg["example_countries"]):
        c, color = curves[code], colors["series"][slot]
        ax.plot(c["radii"], c["shares"] * 100, color=color, lw=1.0, label=iso3[code])
        r, where = needed_radius(c["radii"], c["shares"], c["jmp"])
        if where == "ok":
            ax.plot([r, r], [0, c["jmp"] * 100], color=color, lw=0.5)
            ax.scatter([r], [c["jmp"] * 100], s=16, color=color, edgecolors=colors["surface"], linewidths=0.6,
                       zorder=4)  # fmt: skip
    ax.axvline(fixed_r10k, color=colors["muted"], lw=0.6, ls=(0, (3, 2)))
    ax.text(fixed_r10k * 1.04, 100, f"fixed {fixed_r10k:g} km", color=colors["secondary"], va="top", ha="left",
            fontsize=cfg["font_size_pt"] - SMALL)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(0.5, 40)
    ax.set_ylim(0, 100)
    log_ticks(ax, "x", [0.5, 1, 2, 5, 10, 20, 40])
    ax.set_xlabel("Radius of a 10,000-people plant, r$_{10k}$ (km)")
    ax.set_ylabel("Population inside service areas (%)")
    ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.2)
    ax.text(0.02, 0.98, "dot: r$_{10k}$ at which the share\nreaches the country's JMP share", transform=ax.transAxes,
            va="top", color=colors["secondary"], fontsize=cfg["font_size_pt"] - SMALL)  # fmt: skip
    panel_letter(ax, "a", cfg)

    ax = fig.add_subplot(top[1])
    x = logit(shares["jmp"].to_numpy())
    inside = shares["inside"].to_numpy()
    tried = next(iter(curves.values()))["radii"]
    ax.scatter(x[inside], shares["needed"][inside], s=DOT, color=colors["series"][0], linewidths=0, zorder=3,
               label="needed r$_{10k}$, one dot per country")  # fmt: skip
    ax.scatter(x[~inside], shares["needed"][~inside], s=DOT, facecolors="none", edgecolors=colors["series"][0],
               linewidths=0.5, zorder=3, label=f"not fitted: share not reached in {tried[0]:g}–{tried[-1]:g} km")  # fmt: skip
    xs = np.linspace(logit(0.0), logit(1.0), 200)
    cap = float(choice["cap_km"])
    ax.plot(xs, np.minimum(cap, np.exp(choice["a"] + choice["b"] * xs)), color=colors["ink"], lw=1.0,
            label="chosen rule")  # fmt: skip
    ax.axhline(fixed_r10k, color=colors["muted"], lw=0.6, ls=(0, (3, 2)), label=f"fixed {fixed_r10k:g} km")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 50)
    log_ticks(ax, "y", [0.5, 1, 2, 5, 10, 20, 40])
    logit_axis(ax, [1, 5, 20, 50, 80, 95, 99])
    ax.set_xlabel("Population connected to treatment, JMP, C$_c$ (%, logit scale)")
    ax.set_ylabel("r$_{10k}$ (km)")
    ax.legend(loc="lower right", handlelength=1.4, borderaxespad=0.2)
    panel_letter(ax, "b", cfg)

    errors = {}
    for col, (ref, label) in enumerate((("jmp", "JMP (fitted)"), ("sdg631", "SDG 6.3.1 (not fitted)"))):
        ax = fig.add_subplot(bottom[col])
        ref_values = shares[ref].to_numpy()
        errors[ref] = {k: mean_abs_pp(shares[k].to_numpy(), ref_values) for k in ("fixed", "rule")}
        ax.plot([0, 100], [0, 100], color=colors["muted"], lw=0.5)
        ax.scatter(ref_values * 100, shares["fixed"] * 100, s=DOT - 1, color=colors["series"][1], linewidths=0,
                   label=f"fixed {fixed_r10k:g} km: {errors[ref]['fixed']:.1f} pp")  # fmt: skip
        ax.scatter(ref_values * 100, shares["rule"] * 100, s=DOT - 1, color=colors["series"][0], linewidths=0,
                   label=f"rule: {errors[ref]['rule']:.1f} pp")  # fmt: skip
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Connected to treatment, {label} (%)")
        ax.set_ylabel("Population inside service areas (%)")
        ax.legend(loc="upper left", handlelength=0.8, borderaxespad=0.1, handletextpad=0.3,
                  title="mean absolute error", title_fontsize=cfg["font_size_pt"] - 0.5)  # fmt: skip
        ax.text(0.98, 0.03, f"{int(np.isfinite(ref_values).sum())} countries", transform=ax.transAxes,
                ha="right", va="bottom", color=colors["secondary"], fontsize=cfg["font_size_pt"] - SMALL)  # fmt: skip
        panel_letter(ax, "cd"[col], cfg)

    ax = fig.add_subplot(bottom[2])
    per = (
        shares.assign(
            e_fixed=(shares["fixed"] - shares["jmp"]).abs() * 100, e_rule=(shares["rule"] - shares["jmp"]).abs() * 100
        )  # fmt: skip
        .groupby("region")
        .agg(fixed=("e_fixed", "mean"), rule=("e_rule", "mean"), n=("iso2", "size"))
        .sort_values("fixed")
    )
    y = np.arange(len(per))
    ax.hlines(y, per["rule"], per["fixed"], color=colors["grid_ink"], lw=1.2, zorder=1)
    ax.scatter(per["fixed"], y, s=DOT + 5, color=colors["series"][1], linewidths=0, zorder=3, label="fixed radius")
    ax.scatter(per["rule"], y, s=DOT + 5, color=colors["series"][0], linewidths=0, zorder=3,
               label="rule, region left out")  # fmt: skip
    for yy, (region, n) in zip(y, zip(per.index, per["n"])):  # region names above their row
        ax.text(0.0, yy + 0.22, f"{region} ({n})", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=cfg["font_size_pt"] - SMALL, color=colors["secondary"])  # fmt: skip
    ax.set_yticks([])
    ax.set_ylim(-0.5, len(per) - 0.1)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, None)
    ax.set_xlabel("Mean absolute error against JMP (pp)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=0.8, borderaxespad=0.1,
              handletextpad=0.3, columnspacing=1.0)  # fmt: skip
    panel_letter(ax, "e", cfg, x=-0.02, y=1.1)
    save(fig, "ed_service_radius", cfg)
    return errors


# --------------------------------------------------------------------------- #
# Figure: real catchments                                                     #
# --------------------------------------------------------------------------- #
def unit_table(tables, conf_id):
    """People per catchment (reference, prediction) of one configuration."""
    p = tables["plant_units"]
    p = p[p["conf"] == conf_id][["reference", "unit", "pred"]]
    return tables["plant_references"].merge(p, on=["reference", "unit"], how="left").fillna({"pred": 0.0})


def configuration_ids(tables, measure):
    """conf ids of the fixed radius and of the chosen configuration."""
    choice = chosen(tables, measure)
    pc = tables["plant_check"]
    pc = pc[pc["size_measure"] == measure]
    confs = tables["configurations"]
    fixed_conf = int(pc.loc[pc["cap_km"].isna(), "conf"].iloc[0])
    pick = confs[same_configuration(confs, choice)]
    return fixed_conf, int(pick["conf"].iloc[0]), pick.iloc[0]


def figure_catchments(tables, cal, cfg, fixed_national):
    colors, measure = cfg["colors"], cfg["size_measure"]
    confs = tables["configurations"]
    confs = confs[confs["size_measure"] == measure]
    pc = tables["plant_check"]
    fixed_conf, pick_conf, pick = configuration_ids(tables, measure)
    units = unit_table(tables, pick_conf)
    panels = cfg["catchment_panels"]

    fig = plt.figure(figsize=(cfg["width_mm"] * MM, cfg["heights_mm"]["catchments"] * MM))
    rows = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.2], hspace=0.45)
    top = rows[0].subgridspec(1, len(panels), wspace=0.42)
    bottom = rows[1].subgridspec(1, 4, wspace=0.1, width_ratios=[0.08, 1.0, 1.0, 0.95])
    letters = "abcdefg"
    for col, (reference, title) in enumerate(panels.items()):
        ax = fig.add_subplot(top[col])
        u = units[units["reference"] == reference]
        ref, pred = u["ref"].to_numpy(float), u["pred"].to_numpy(float)
        lo, hi = 50.0, float(max(ref.max(), pred.max())) * 2
        xs = np.array([lo, hi])
        ax.fill_between(xs, xs / 2, xs * 2, color=colors["band"], lw=0)
        ax.plot(xs, xs, color=colors["muted"], lw=0.5)
        ax.scatter(np.maximum(ref, lo), np.maximum(pred, lo), s=2.5, color=colors["series"][0], linewidths=0,
                   alpha=0.6, rasterized=True)  # fmt: skip
        typical = 10 ** class_balanced_error(pred, ref)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title("\n".join(title), color=colors["ink"], pad=3, fontsize=cfg["font_size_pt"])
        ax.text(0.04, 0.97, f"n = {len(u):,}\ntypical ×{typical:.2f}\nsum ratio {pred.sum() / ref.sum():.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=cfg["font_size_pt"] - SMALL)  # fmt: skip
        ax.set_xlabel("Reference people")
        if col == 0:
            ax.set_ylabel("People inside the service area")
        panel_letter(ax, letters[col], cfg, y=1.13)  # level with the two title lines

    ax = fig.add_subplot(bottom[1:3])
    for slot, ((lo_, hi_), g) in enumerate(confs.groupby(["scale_min", "scale_max"], sort=True)):
        g = g.sort_values("cap_km")
        color = colors["series"][slot]
        label = "no national scale" if lo_ == 1 and hi_ == 1 else f"scale limits {lo_:.2g}–{hi_:g}"
        ax.plot(g["plants_x"], g["national_error_pp"], color=color, lw=0.8, label=label, zorder=2)
        ok = is_true(g["allowed"])
        ax.scatter(g.loc[ok, "plants_x"], g.loc[ok, "national_error_pp"], s=DOT + 3, color=color, linewidths=0,
                   zorder=3)  # fmt: skip
        ax.scatter(g.loc[~ok, "plants_x"], g.loc[~ok, "national_error_pp"], s=DOT + 3, facecolors=colors["surface"],
                   edgecolors=color, linewidths=0.6, zorder=3)  # fmt: skip
        if np.isclose(lo_, pick["scale_min"]) and np.isclose(hi_, pick["scale_max"]):
            for _, r in g[g["cap_km"].isin(cfg["label_caps_km"])].iterrows():
                ax.annotate(f"cap {r['cap_km']:g} km" if r["cap_km"] < max(cfg["label_caps_km"]) else "no cap",
                            (r["plants_x"], r["national_error_pp"]), xytext=(3, 3), textcoords="offset points",
                            fontsize=cfg["font_size_pt"] - 1.5, color=colors["secondary"])  # fmt: skip
    base = 10 ** float(pc[pc["conf"] == fixed_conf].groupby("iso2")["class_balanced_error"].mean().mean())
    ax.axvline(base, color=colors["muted"], lw=0.5, ls=(0, (3, 2)))
    ax.scatter([base], [fixed_national], marker="D", s=18, color=colors["ink"], zorder=4, label="fixed radius")
    ax.scatter([pick["plants_x"]], [pick["national_error_pp"]], s=70, facecolors="none", edgecolors=colors["ink"],
               linewidths=0.9, zorder=5, label="chosen")  # fmt: skip
    countries = ", ".join(sorted(pc["iso2"].unique()))
    ax.set_xlabel(f"Per-plant error: typical factor, mean over {countries} (×)")
    ax.set_ylabel("National error: mean of JMP and SDG 6.3.1 (pp)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="", markerfacecolor=colors["surface"],
                          markeredgecolor=colors["secondary"], markersize=3.5, markeredgewidth=0.6))  # fmt: skip
    labels.append(f"open: worse per plant than the fixed\nradius in ≥{float(cal['plant_confidence']) * 100:.0f} % "
                  "of the resamples")  # fmt: skip
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.0), handlelength=1.4)
    panel_letter(ax, letters[len(panels)], cfg)
    save(fig, "ed_catchments", cfg)
    return base


# --------------------------------------------------------------------------- #
# Numbers for the Methods text                                                #
# --------------------------------------------------------------------------- #
def tex_int(value):
    return f"{int(round(float(value))):,}".replace(",", "{,}")


def macro_name(text):
    """A LaTeX macro name part from a reference name (letters only, CamelCase)."""
    return "".join(w.capitalize() for w in re.split(r"[^A-Za-z]+", text) if w)


def signed(value):
    """One decimal with a typographic minus (text mode)."""
    return f"{float(value):.1f}".replace("-", "\\textminus{}")


def limit_list(limits):
    """'none, 0.33--3 and 0.1--10' from connection_radius.calibration_size_scale_limits."""
    names = [
        "none" if lo == 1 and hi == 1 else f"{lo:.2g}--{hi:g}" for lo, hi in ((d["min"], d["max"]) for d in limits)
    ]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def write_numbers(tables, voronoi, cal, cfg, shares, errors, fixed_national, plant_base):
    measure = cfg["size_measure"]
    f = voronoi["footprint_to_people"]
    rc = voronoi["radius_calibration"][measure]
    choice = chosen(tables, measure)
    fit = fitted_plants(tables["plants"])
    lx, ly = np.log10(fit["footprint_m2"]), np.log10(fit["reported"])
    res = ly - (f["intercept"] + f["slope"] * lx)
    sizes = tables["sizes"]
    offsets = sizes.dropna(subset=["offset"]).set_index("country")["offset"]
    with_offset = fit["iso2"].isin(offsets.index).to_numpy()
    before = 10 ** np.median(np.abs(res[with_offset]))
    after = 10 ** np.median(np.abs(res[with_offset] - fit.loc[with_offset, "iso2"].map(offsets)))
    confs = tables["configurations"]
    confs = confs[confs["size_measure"] == measure]
    _, _, pick = configuration_ids(tables, measure)
    row = tables["by_cap"]
    row = row[same_configuration(row, choice) & np.isclose(row["radius_exponent"], rc["radius_exponent"])].iloc[0]
    jmp_weighted_fixed = float(
        np.sum(np.abs(shares["fixed"] - shares["jmp"]) * shares["people"]) / shares["people"].sum()
    )
    reported = tables["countries"]
    by_country = tables["by_country"]
    raw = by_country[(by_country["size_measure"] == measure) & by_country["iso2"].isin(shares["iso2"])]
    raw = raw.drop_duplicates("iso2")["size_scale_unlimited"]
    numbers = {
        "Year": str(cal["year"]),
        "PlantsAll": tex_int(len(tables["plants"])),
        "PlantsSized": tex_int(int(is_true(tables["plants"]["sized"]).sum())),
        "PlantsReported": tex_int(int(tables["plants"]["reported"].notna().sum())),
        "FitPlants": tex_int(len(fit)),
        "FitIntercept": f"{f['intercept']:.3f}",
        "FitSlope": f"{f['slope']:.2f}",
        "FitRsq": f"{np.corrcoef(lx, ly)[0, 1] ** 2:.2f}",
        "OffsetCountries": tex_int(len(offsets)),
        "OffsetMin": f"{10 ** offsets.min():.2f}",
        "OffsetMax": f"{10 ** offsets.max():.1f}",
        "OffsetMinPlants": tex_int(cal["size_country_min_plants"]),
        "EstimateErrorBefore": f"{before:.2f}",
        "EstimateErrorAfter": f"{after:.2f}",
        "UnsizedOwn": tex_int(int((sizes["unsized_from"] == "own").sum())),
        "UnsizedRatioCountries": tex_int(int((sizes["unsized_from"] == "ratio").sum())),
        "UnsizedRatio": f"{float(sizes['unsized_ratio'].iloc[0]):.3f}",
        "UnsizedDefault": tex_int(f["unsized_people"]),
        "CurveCountries": tex_int(int((reported["status"] == "ok").sum())),
        "Countries": tex_int(len(shares)),
        "RadiusExponent": f"{float(rc['radius_exponent']):g}",
        "FixedRadius": f"{float(rc['radius_at_10k_people_km']):g}",
        "MaxBufferKm": f"{float(cal['max_buffer']) / 1000:g}",
        "CandidatePlants": tex_int(cal["candidate_plants"]),
        "GridStart": f"{float(cal['radius_grid']['start']):g}",
        "GridStop": f"{float(cal['radius_grid']['stop']):g}",
        "GridCount": tex_int(cal["radius_grid"]["count"]),
        "CapCount": tex_int(len(cal["calibration_caps_km"])),
        "CapLow": f"{min(cal['calibration_caps_km']):g}",
        "CapHigh": f"{max(cal['calibration_caps_km']):g}",
        "LimitCount": tex_int(len(cal["calibration_size_scale_limits"])),
        "LimitList": limit_list(cal["calibration_size_scale_limits"]),
        "Regions": tex_int(shares["region"].nunique()),
        "ScaleRawMin": f"{raw[raw > 0].min():.1g}",  # countries with a JMP share of 0 have k = 0
        "ScaleRawMax": tex_int(raw.max()) if raw.max() >= 10 else f"{raw.max():.2g}",
        "MatchDistance": tex_int(cal["polygon_match_distance_m"]),
        "CellKm": f"{float(cal['aggregate_cells']) * WORLDPOP_CELL_KM:g}",
        "ScaleMin": f"{float(choice['scale_min']):.3g}",
        "ScaleMax": f"{float(choice['scale_max']):g}",
        "ScaleChosen": limit_list([{"min": choice["scale_min"], "max": choice["scale_max"]}]),
        "Cap": f"{float(choice['cap_km']):g}",
        "RuleA": f"{float(choice['a']):.3f}",
        "RuleB": f"{float(choice['b']):.3f}",
        "ErrFixedJmp": f"{errors['jmp']['fixed']:.1f}",
        "ErrFixedJmpWeighted": f"{jmp_weighted_fixed * 100:.1f}",
        "ErrRuleJmp": f"{float(row['mean_abs_error_pp']):.1f}",
        "ErrRuleJmpWeighted": f"{float(row['people_weighted_abs_error_pp']):.1f}",
        "BiasRuleJmp": signed(row["bias_pp"]),
        "NationalFixed": f"{fixed_national:.1f}",
        "NationalChosen": f"{float(pick['national_error_pp']):.1f}",
        "PlantErrFixed": f"{plant_base:.2f}",
        "PlantErrChosen": f"{float(pick['plants_x']):.2f}",
        "PlantErrLow": f"{confs['plants_x'].min():.2f}",
        "PlantErrHigh": f"{confs['plants_x'].max():.2f}",
        "Configurations": tex_int(len(confs)),
        "AllowedConfigurations": tex_int(int(is_true(confs["allowed"]).sum())),
        "BootstrapSamples": tex_int(cal["plant_bootstrap_samples"]),
        "Confidence": f"{float(cal['plant_confidence']) * 100:.0f}",
    }
    for r in tables["plant_status"].itertuples():
        if r.status == "ok":
            numbers[f"Units{macro_name(r.reference)}"] = tex_int(r.units)
            numbers[f"People{macro_name(r.reference)}"] = f"{float(r.people) / 1e6:.2g}"
    for name in cal["check_references"]:  # national references that are not fitted
        known = shares[name].notna()
        numbers[f"{macro_name(name)}Countries"] = tex_int(int(known.sum()))
        if not known.any():  # no country of the curves has this reference: no errors to quote
            continue
        numbers[f"ErrFixed{macro_name(name)}"] = (
            f"{mean_abs_pp(shares['fixed'].to_numpy(), shares[name].to_numpy()):.1f}"
        )
        numbers[f"ErrRule{macro_name(name)}"] = f"{float(row[f'{name}_mean_abs_error_pp']):.1f}"
        numbers[f"BiasRule{macro_name(name)}"] = signed(row[f"{name}_bias_pp"])
    districts = tables["districts"]
    for r in districts[same_configuration(districts, choice)].itertuples():
        numbers[f"Served{macro_name(r.reference)}"] = f"{r.served_share * 100:.0f}"
        numbers[f"Drainage{macro_name(r.reference)}"] = f"{r.drainage_share * 100:.0f}"
        numbers[f"Districts{macro_name(r.reference)}"] = tex_int(r.n_districts)
    lines = ["% generated by wwtp.figures_scripts.calibration_figures from the calibration tables; do not edit"]
    lines += [f"\\newcommand{{\\cal{k}}}{{{v}}}" for k, v in numbers.items()]
    with (
        atomic_output(os.path.join(cfg["paths"]["output_dir"], "calibration_numbers.tex")) as tmp,
        open(tmp, "w") as fh,
    ):
        fh.write("\n".join(lines) + "\n")
    return numbers


def run(cfg):
    cal = load_config(script_name="connection_radius")
    render(cfg, cal, load_config(script_name="create_voronoi"), read_tables(cal["paths"]["output_dir"]))


def render(cfg, cal, voronoi, tables):
    """The three figures and the numbers file from the calibration ``tables``."""
    apply_style(cfg)
    measure = cfg["size_measure"]
    rc = voronoi["radius_calibration"][measure]
    exponent, fixed_r10k = float(rc["radius_exponent"]), float(rc["radius_at_10k_people_km"])
    curves = country_curves(tables, measure, exponent, cal["check_references"])
    shares = national_shares(curves, chosen(tables, measure), exponent, fixed_r10k, tables["by_country"])
    fixed_national = fixed_national_error(shares, cal["choice_national"])
    figure_plant_size(tables, voronoi, cfg)
    errors = figure_service_radius(tables, voronoi, cfg, shares, curves, fixed_r10k)
    plant_base = figure_catchments(tables, cal, cfg, fixed_national)
    numbers = write_numbers(tables, voronoi, cal, cfg, shares, errors, fixed_national, plant_base)
    logger.info("calibration figures for %s written with %d numbers", measure, len(numbers))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    run(load_config(script_name="calibration_figures"))


if __name__ == "__main__":
    configure_logging()
    main()
