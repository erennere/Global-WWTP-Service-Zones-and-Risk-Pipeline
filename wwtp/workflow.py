"""What the Snakemake workflow (workflow/Snakefile) runs: variants, and each stage's inputs and outputs.

A *variant* is one choice of the seven run parameters (level, version, buffer,
weight_method, weight_func, dynamic_buffering, size_measure). The default
variant is the one config.yaml sets for create_voronoi; the sweep grid comes
from the ``sweep`` section. Every path here is computed by the same code the
stages use (``starter.load_config`` and the path helpers in ``pipelines``), so
the files Snakemake expects are the files the stages write.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from functools import lru_cache

from . import settings, starter
from .pipelines import create_output_paths, create_pop_output_paths
from .pop_at_risk_river_calculations.impact_polygons_pop import impact_polygons_output_path

RUN_KEYS = ("level", "version", "buffer", "weight_method", "weight_func", "dynamic_buffering", "size_measure")

# Stages that prepare inputs once. The workflow defines their rules only when a run names them
# (workflow.run_once, or ``snakemake --config run_once=...``); otherwise their outputs are plain inputs.
ONE_TIME_STAGES = {
    "combine_locations": ("combine_locations",),
    "annotations": ("annotation_grids", "wastewater_join", "download_bing_annotate", "merge_annotations"),
    "annotation_grids": ("annotation_grids",),
    "wastewater_join": ("wastewater_join",),
    "download_bing_annotate": ("download_bing_annotate",),
    "merge_annotations": ("merge_annotations",),
    "download_pop": ("download_pop",),
}


@dataclass(frozen=True)
class Variant:
    level: str
    version: str
    buffer: int
    weight_method: str
    weight_func: str
    dynamic_buffering: bool
    size_measure: str | None

    @classmethod
    def from_section(cls, cfg):
        return cls(**{key: cfg[key] for key in RUN_KEYS})

    def overrides(self):
        """Keyword arguments for ``starter.load_config``."""
        return {key: getattr(self, key) for key in RUN_KEYS}

    def cli(self):
        """The same values as command-line flags of every stage.

        An empty weight_func goes as ``--weight-func=``: Snakemake's ``{params.cli:q}`` does not quote an
        empty element, so ``--weight-func ''`` reached the stage as a flag without a value. Other values
        keep the two-word form, so the params of finished jobs (and their rerun state) do not change.
        """
        args = [
            "--level",
            self.level,
            "--version",
            self.version,
            "--buffer",
            str(self.buffer),
            "--weight-method",
            self.weight_method,
            *(["--weight-func", self.weight_func] if self.weight_func else ["--weight-func="]),
            "--dynamic-buffering",
            str(self.dynamic_buffering).lower(),
        ]
        if self.size_measure is not None:
            args += ["--size-measure", self.size_measure]
        return args

    @property
    def tag(self):
        """The suffix the stages put in file names, e.g. ``v2_lvl6_bfrep_sz_mult``; also the rule-name suffix."""
        cfg = config("create_voronoi", self)
        return f"v{self.version}_lvl{self.level}_bf{cfg['buffer_path_token']}_{cfg['weight_type']}{cfg['weight_func_suffix']}"


@lru_cache(maxsize=None)
def config(stage, variant=None):
    """The resolved config of ``stage`` for ``variant`` (the configured run when None)."""
    return starter.load_config(script_name=stage, **(variant.overrides() if variant else {}))


def default_variant():
    return Variant.from_section(config("create_voronoi"))


def sweep_variants(weight_funcs=None):
    """The sensitivity grid of the ``sweep`` section.

    Rigid buffers and size-based radii for every level and weighting. Without a
    distance weighting (weight_func '') the weight_method has no effect, so only
    'linear' runs; 'size' weights need the multiplicative distance. The
    additive weighting is in the grid only when ``sweep_weight_funcs`` (or
    ``weight_funcs``) lists 'add'. Size-based runs keep the default buffer, which
    then only names nothing (their files carry the size measure instead).
    """
    cfg = config("sweep")
    default = default_variant()
    variants = []
    for level in cfg["sweep_levels"]:
        for weight_func in cfg["sweep_weight_funcs"] if weight_funcs is None else weight_funcs:
            for method in cfg["sweep_weight_methods"]:
                if (weight_func == "" and method != "linear") or (method == "size" and weight_func != "mult"):
                    continue
                for buffer in cfg["sweep_rigid_buffers"]:
                    # size_measure still picks the plant size that weight_method 'size' uses
                    variants.append(
                        Variant(
                            str(level), default.version, int(buffer), method, weight_func, False, default.size_measure
                        )
                    )
                for measure in cfg["sweep_size_measures"]:
                    variants.append(
                        Variant(str(level), default.version, default.buffer, method, weight_func, True, measure)
                    )
    return variants


def one_time_rules(snakemake_config):
    """The one-time rules this run includes.

    ``snakemake --config run_once=annotations,download_pop`` names them for one run (a comma-separated
    string or a YAML list); without it, ``workflow.run_once`` in config.yaml does (default: none).
    """
    names = snakemake_config["run_once"] if "run_once" in snakemake_config else config("workflow")["run_once"]
    if isinstance(names, str):
        names = [name.strip() for name in names.split(",") if name.strip()]
    unknown = [name for name in names if name not in ONE_TIME_STAGES]
    if unknown:
        raise settings.SettingsError(f"run_once: unknown stage(s) {unknown}; choose from {list(ONE_TIME_STAGES)}")
    return {rule for name in names for rule in ONE_TIME_STAGES[name]}


def approach0_source(variant):
    """The variant whose approach-0 files ``variant`` reuses, or None when it makes its own.

    Approach 0 groups plants by overlapping buffers and never reads the basins, so its service areas
    and population layer are the same at every basin level. The sweep makes them once, at the
    configured run's level (or the first sweep level when that one is not swept), and links them to
    the other levels' file names.
    """
    levels = [str(level) for level in config("sweep")["sweep_levels"]]
    shared = default_variant().level if default_variant().level in levels else levels[0]
    return None if variant.level == shared else replace(variant, level=shared)


def approaches(variant):
    """Service-area approaches built for a variant: 0 and 1, and 2 (cities) when city_voronoi is on."""
    return ["0", "1"] + (["2"] if config("create_voronoi", variant)["city_voronoi"] else [])


def figure_approach(variant):
    """The approach the risk chain and figures read (``figures.approach``)."""
    return str(config("create_rasters", variant)["figures"]["approach"])


def service_areas(variant, approach):
    return create_output_paths(config("create_voronoi", variant))["voronoi"][approach]


def population_layer(variant, approach):
    return create_pop_output_paths(config("add_pop", variant))["voronoi"][approach]


def _csv(path):
    return os.path.abspath(path.replace(".gpkg", ".csv"))


def raster_stats(variant):
    """create_rasters' final CSVs: per-country served/unserved population and the non-served islands."""
    paths = config("create_rasters", variant)["paths"]
    return [_csv(paths["csv_output_filepath"]), _csv(paths["non_served_outpath"])]


def raster_shards(variant):
    """Number of create_rasters jobs (country shards) per variant."""
    return int(config("create_rasters", variant)["shards"])


def raster_shard_files(variant, index):
    from .pop_at_risk_river_calculations.create_rasters import shard_path

    return [shard_path(path, index, raster_shards(variant)) for path in raster_stats(variant)]


@lru_cache(maxsize=None)
def watershed(level):
    """The merged HydroBASINS layer of one level (combine_watersheds' output)."""
    return starter.load_config(script_name="combine_watersheds", level=str(level))["paths"]["watershed"]


@lru_cache(maxsize=None)
def industrial_land(level, version):
    """download_and_vectorize's merged industrial-land layer for one basin level."""
    cfg = starter.load_config(script_name="download_and_vectorize", level=str(level), version=version)
    return cfg["paths"]["industrial_merged_filepath"]


def impact_polygons(variant):
    cfg = config("impact_polygons_pop", variant)
    return [
        impact_polygons_output_path(cfg["paths"]["impact_pop_polygons_outpath"], radius)
        for radius in cfg["impact_polygons_pop_params"]["impact_radii"]
    ]


def diff_layer(variant):
    cfg = config("find_diff_pop", variant)
    return os.path.join(
        cfg["paths"]["pop_dif_output_dir"],
        f"diff_{os.path.basename(population_layer(variant, figure_approach(variant)))}",
    )


def annotation_grid():
    """NEW_01's tiles: ``grids_<corrected plants file name>`` in annotations_grid_dir."""
    paths = config("NEW_01_GENERATEGRIDS")["paths"]
    return os.path.join(paths["annotations_grid_dir"], f"grids_{os.path.basename(paths['corrected_all_filepath'])}")


def osm_layers():
    """NEW_03's merged OSM polygons and lines (``<annotations>/data``), read by download_bing_annotate."""
    folder = os.path.join(
        os.path.dirname(config("NEW_03_WASTEWATERJOIN_GEOJSON")["paths"]["annotations_grid_dir"]), "data"
    )
    return [os.path.join(folder, "merged_polygons.parquet"), os.path.join(folder, "merged_lines.parquet")]


def resources(stage):
    """SLURM resources of a stage's jobs (``workflow.resources``)."""
    return dict(config("workflow")["resources"][stage])


def sizing_params(variant):
    """Extra rule params of a size-based variant: the plant sizes and the radius calibration it runs with.

    Its file names do not carry them, so Snakemake would keep service areas made with older values; as
    rule params, a change reruns them. Variants that use no plant size (rigid buffer without size
    weights) get none and stay.
    """
    if not (variant.dynamic_buffering or variant.weight_method == "size"):
        return {}
    kwargs = config("create_voronoi", variant)["calculate_buffer_kwargs"]

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]

    sizes = {"footprint_to_people": digest(kwargs["footprint_to_people"]), "size_scale": digest(kwargs["size_scale"])}
    params = {"sizes": json.dumps(sizes, sort_keys=True)}
    if variant.dynamic_buffering:
        signature = {
            "exponent": kwargs["radius_exponent"],
            "r10k": kwargs["radius_at_10k_people_km"],
            "by_connection": kwargs["radius_by_connection"],
            "connected_share": digest(kwargs["connected_share"]),
        }
        params["radius"] = json.dumps(signature, sort_keys=True)
    return params


def settings_params(section, *keys):
    """Rule param of a step without variant: a digest of the settings of ``section`` it runs with.

    Its outputs do not carry them, so a change (the plant-size model for the calibration curves, the
    caps and limits for ``calibrate``, the register list for ``national_registers``) would leave old
    tables in place; as a rule param it reruns the step.
    """
    cfg = config(section)
    payload = json.dumps({key: cfg[key] for key in keys}, sort_keys=True, default=str)
    return {"settings": hashlib.sha256(payload.encode()).hexdigest()[:12]}


def stamp(name, variant=None):
    """Marker file for stages whose outputs Snakemake cannot list up front (plots, per-country folders)."""
    return os.path.join(config("workflow")["paths"]["stamp_dir"], f"{name}{'__' + variant.tag if variant else ''}.done")


def log(name, variant=None):
    return os.path.join(config("workflow")["paths"]["log_dir"], f"{name}{'__' + variant.tag if variant else ''}.log")
