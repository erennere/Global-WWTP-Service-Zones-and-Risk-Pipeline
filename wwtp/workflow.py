"""What the Snakemake workflow (workflow/Snakefile) runs: variants, and each stage's inputs and outputs.

A *variant* is one choice of the seven run parameters (level, version, buffer,
weight_method, weight_func, dynamic_buffering, size_measure). The default
variant is the one config.yaml sets for create_voronoi; the sweep grid comes
from the ``sweep`` section. Every path here is computed by the same code the
stages use (``starter.load_config`` and the path helpers in ``pipelines``), so
the files Snakemake expects are the files the stages write.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from . import starter
from .pipelines import create_output_paths, create_pop_output_paths
from .pop_at_risk_river_calculations.impact_polygons_pop import impact_polygons_output_path

RUN_KEYS = ("level", "version", "buffer", "weight_method", "weight_func", "dynamic_buffering", "size_measure")


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
        """The same values as command-line flags of every stage (quote them: weight_func can be '')."""
        args = [
            "--level",
            self.level,
            "--version",
            self.version,
            "--buffer",
            str(self.buffer),
            "--weight-method",
            self.weight_method,
            "--weight-func",
            self.weight_func,
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


def stamp(name, variant=None):
    """Marker file for stages whose outputs Snakemake cannot list up front (plots, per-country folders)."""
    return os.path.join(config("workflow")["paths"]["stamp_dir"], f"{name}{'__' + variant.tag if variant else ''}.done")


def log(name, variant=None):
    return os.path.join(config("workflow")["paths"]["log_dir"], f"{name}{'__' + variant.tag if variant else ''}.log")
