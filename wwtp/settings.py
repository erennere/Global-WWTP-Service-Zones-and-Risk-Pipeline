"""Typed settings: one pydantic model per config.yaml section.

``starter.load_config`` validates every section after the layers (base file,
overlays, ``config.local.yaml``, ``--set``) and the CLI overrides are applied, and
before paths are expanded. A wrong type, an unknown or missing key, a value
outside its allowed set (``weight_method``, ``size_measure``, ...) or a broken
cross-field rule stops the stage at load with the section and key named.

Values come back as plain dicts, as before, with two normalisations: ``level``
and ``version`` are always strings (``--set combine_watersheds.level=7`` gives
the int 7), and numbers typed as float here are floats.

Adding a key to config.yaml means adding it to its model here;
``wwtp/tests/unit/test_settings.py`` checks every section of config.yaml
validates and every model has a section.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (BaseModel, BeforeValidator, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat,
                      PositiveInt, ValidationError, model_validator)


def _digits(value):
    """Accept 6 or '6' for a level/version and keep it as the string the path templates use."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value


Level = Annotated[str, BeforeValidator(_digits), Field(pattern=r"^(0?[1-9]|1[0-2])$")]  # HydroBASINS levels 1-12
Version = Annotated[str, BeforeValidator(_digits), Field(pattern=r"^\w+$")]
WeightMethod = Literal["linear", "square_root", "logarithmic", "sigmoid", "size"]
WeightFunc = Literal["", "mult", "add"]
SizeMeasure = Literal["reported", "footprint"]
Paths = dict[str, str]  # resolved path templates; every key must have a value by now
Fraction = Annotated[float, Field(ge=0, le=1)]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Section(Strict):
    """A config.yaml section."""


class RunParams(Section):
    """The seven run parameters that pick a service-area variant (and its output paths)."""
    level: Level
    version: Version
    buffer: NonNegativeInt
    weight_method: WeightMethod
    weight_func: WeightFunc
    dynamic_buffering: bool
    size_measure: SizeMeasure | None

    @model_validator(mode="after")
    def _consistent(self):
        if self.weight_method == "size" and self.weight_func != "mult":
            raise ValueError("weight_method 'size' needs weight_func 'mult' (multiplicative weighted distance)")
        if self.dynamic_buffering and self.size_measure is None:
            raise ValueError("size_measure must be configured when dynamic_buffering is true")
        return self


class Figures(Strict):
    approach: Literal[0, 1]


class Workers(Strict):
    max_workers: PositiveInt


class FootprintToPeople(Strict):
    """people = 10 ** (intercept + slope * log10(footprint m2)); unsized plants get unsized_people."""
    intercept: float
    slope: PositiveFloat
    unsized_people: PositiveFloat


class RadiusCalibration(Strict):
    """Circle radius = radius_at_10k_people_km * (people / 10,000) ** radius_exponent."""
    radius_exponent: PositiveFloat
    radius_at_10k_people_km: PositiveFloat


class ServiceAreaParams(RunParams):
    """Keys create_voronoi and find_unconnected_industrial_areas share (both build service areas)."""
    max_workers: PositiveInt
    n_points: PositiveInt
    threshold: NonNegativeInt
    sigma: NonNegativeInt
    percent_threshold: NonNegativeInt
    basin_column_name: str
    country_output_column: str
    country_boundary_column: str
    site_id_column: str
    old_site_id_column: str
    calculate_area_fn: str
    calculate_buffer_fn: str
    area_fn_kwargs: dict[str, Any]
    prepare_data_fn: str
    overwrite_existing: bool
    temp_voronoi_overwrite: bool
    flush_size: PositiveInt
    max_buffer: PositiveInt
    reported_size_columns: list[str]
    reported_size_skip: dict[str, dict[str, list[int]]]
    footprint_to_people: FootprintToPeople
    radius_calibration: dict[SizeMeasure, RadiusCalibration]
    scipy_true: bool
    cv2_true: bool
    industrial_category_numbers: list[int]
    mix_use_categories: list[int]
    csv_files: bool
    sindex_concurrency: bool
    remove_industrial: bool
    return_boolean: bool
    force_country_rejoin: bool
    paths: Paths

    @model_validator(mode="after")
    def _calibrated(self):
        if self.dynamic_buffering and self.size_measure not in self.radius_calibration:
            raise ValueError(f"size_measure {self.size_measure!r} has no radius_calibration entry; choose one of "
                             f"{sorted(self.radius_calibration)} or add it with wwtp.calibration.service_radius")
        return self


class YearRange(Strict):
    start_year: int
    end_year: int
    include_2014: bool

    @model_validator(mode="after")
    def _ordered(self):
        if self.start_year > self.end_year:
            raise ValueError(f"start_year {self.start_year} is after end_year {self.end_year}")
        return self


# --------------------------------------------------------------------------- #
# Data merge and annotation                                                   #
# --------------------------------------------------------------------------- #
class CorrectLocationsWOSM(Section):
    version: Version
    rad: PositiveInt
    country_output_column: str
    country_boundary_column: str
    paths: Paths


class RunManifest(Section):
    hash_max_mb: NonNegativeInt
    paths: Paths


class MergeSegResults(Section):
    version: Version
    legacy_merge: bool
    paths: Paths


class FinalDataMerge(Section):
    version: Version
    legacy_merge: bool
    threshold: NonNegativeInt
    osm_threshold: NonNegativeInt
    paths: Paths


class CombineWatersheds(Section):
    level: Level
    paths: Paths


class GridAnnotations(Strict):
    cell_size: PositiveInt
    factor: PositiveFloat


class Generategrids(Section):
    version: Version
    annotations: GridAnnotations
    paths: Paths


class OverpassAnnotations(Strict):
    max_workers: PositiveInt
    retries: NonNegativeInt
    overpass_urls: list[str] = Field(min_length=1)
    overpass_pause_seconds: NonNegativeFloat


class ExtractOsm(Section):
    version: Version
    overwrite_existing: bool
    annotations: OverpassAnnotations
    paths: Paths


class BingAnnotations(Strict):
    cell_size: PositiveInt
    factor: PositiveFloat
    image_size_px: PositiveInt
    zoom_level: PositiveInt
    base_z17_resolution: PositiveFloat
    max_workers: PositiveInt
    georeferenced: bool
    fontsize: PositiveInt
    dpi: PositiveInt
    target_size_px: PositiveInt
    mercator_tile_size_px: PositiveInt
    imagery_reference_tile_size_px: PositiveInt
    earth_radius_m: PositiveInt
    random_seed: int


class DownloadBingAnnotate(Section):
    version: Version
    annotations: BingAnnotations
    paths: Paths


class MergeAnnotations(Section):
    version: Version
    paths: Paths


class InspectionAnnotations(Strict):
    n_sample_size: PositiveInt
    random_seed: int


class AnnotationsInspection(Section):
    annotations: InspectionAnnotations
    paths: Paths


class CopyFalsyImages(Section):
    version: Version
    paths: Paths


class Seed(Strict):
    random_seed: int


class Sampling(Strict):
    n_per_class: PositiveInt


class SampleAnnotationsByClass(Section):
    annotations: Seed
    sampling: Sampling
    paths: Paths


class WastewaterJoin(Section):
    version: Version
    overwrite_existing: bool
    paths: Paths


class GeotiffExport(Strict):
    id_field: str
    output_width_px: PositiveInt
    output_height_px: PositiveInt
    dpi: PositiveInt
    image_format: Literal["tif", "png"]


class ExportGeotiff(Section):
    version: Version
    export: GeotiffExport
    paths: Paths


# --------------------------------------------------------------------------- #
# Service areas and population                                                #
# --------------------------------------------------------------------------- #
class DownloadPop(Section, YearRange):
    worldpop_2014_url_template: str
    worldpop_yearly_url_template: str
    country_limit: NonNegativeInt
    paths: Paths


class CreateVoronoi(ServiceAreaParams):
    city_voronoi: bool
    duckdb_cond: bool


class AddPop(RunParams, YearRange):
    add_pop_max_workers: PositiveInt
    country_output_column: str
    overwrite_existing: bool
    paths: Paths


class RasterAnnotations(Strict):
    max_workers: PositiveInt
    random_seed: int


class CreateRasters(RunParams):
    zoom_level: PositiveInt
    min_pixels: PositiveInt
    served_all_touched: bool
    shards: PositiveInt
    basin_column_name: str
    country_boundary_column: str
    country_output_column: str
    annotations: RasterAnnotations
    figures: Figures
    paths: Paths


class FindUnservedPop(RunParams):
    country_output_column: str
    max_workers: PositiveInt
    is_parallel: bool
    zoom_level: PositiveInt
    threshold_value: NonNegativeInt
    urban_buffer: NonNegativeInt
    wwtp_buffer: NonNegativeInt
    tolerance: NonNegativeInt
    paths: Paths


class FindDiffPop(RunParams):
    max_workers: PositiveInt
    basin_column_name: str
    country_output_column: str
    country_boundary_column: str
    figures: Figures
    paths: Paths


# --------------------------------------------------------------------------- #
# Downstream risk                                                             #
# --------------------------------------------------------------------------- #
class AssignRiversToBasin(RunParams):
    paths: Paths


class FindIntersectionRiver(RunParams):
    x_distance: PositiveInt
    basin_column_name: str
    paths: Paths


class ImpactParams(Strict):
    """Plume model: organic load per person, decay, reach and the radii reported."""
    org_per_pop: PositiveFloat
    c_limit: PositiveFloat
    base_k: PositiveFloat
    theta: PositiveFloat
    step_m: PositiveFloat
    least_discharge_cms: PositiveFloat
    velocity_m_s: PositiveFloat
    water_temp_equator_c: float
    impact_radii: list[PositiveInt] = Field(min_length=1)


class ImpactPolygonsPop(RunParams):
    impact_polygons_pop_params: ImpactParams
    paths: Paths


class FindPopInDangerPop(RunParams):
    zoom_level: PositiveInt
    country_boundary_column: str
    country_output_column: str
    annotations: Workers
    paths: Paths


# --------------------------------------------------------------------------- #
# Industrial areas                                                            #
# --------------------------------------------------------------------------- #
class DownloadAndVectorize(Section):
    level: Level
    version: Version
    max_workers: PositiveInt
    industrial_batch_size: PositiveInt
    basin_column_name: str
    country_boundary_column: str
    country_output_column: str
    industrial_zenodo_url: str
    industrial_download_chunk_size: PositiveInt
    industrial_min_cells: PositiveInt
    industrial_persist_rasters: bool
    industrial_simplify_tolerance: NonNegativeFloat
    overwrite_existing: bool
    sindex_concurrency: bool
    paths: Paths


class FindUnconnectedIndustrialAreas(ServiceAreaParams):
    industrial_min_cells: PositiveInt
    unconnected_sjoin_workers: PositiveInt


# --------------------------------------------------------------------------- #
# Validation against reference data                                           #
# --------------------------------------------------------------------------- #
class VerificationScript(RunParams):
    percent_verification: Fraction
    paths: Paths


class HwComparison(RunParams):
    paths: Paths


class EuComparison(RunParams):
    threshold: NonNegativeInt
    eu_reference_factor: PositiveFloat
    paths: Paths


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
class VectorTiles(Strict):
    enabled: bool
    min_zoom: NonNegativeInt
    max_zoom: NonNegativeInt
    min_render_zoom: NonNegativeInt
    extent: PositiveInt
    layer_name: str
    simplify_tolerance_deg: NonNegativeFloat
    zip_tiles: bool

    @model_validator(mode="after")
    def _zooms(self):
        if self.min_zoom > self.max_zoom:
            raise ValueError(f"min_zoom {self.min_zoom} is above max_zoom {self.max_zoom}")
        return self


class InteractiveUnconnectedIndustrialMap(RunParams):
    industrial_min_cells: PositiveInt
    initial_zoom: NonNegativeInt
    vector_tiles: VectorTiles
    paths: Paths


class CompositeAreaPopulationPlots(RunParams):
    zonal_sum_default_column: str
    min_country_facility_count: NonNegativeInt
    plot_outlier_quantiles: list[Fraction]
    plot_outlier_iqr_factor: PositiveFloat
    figures: Figures
    paths: Paths


class PiechartInteractive(RunParams):
    min_total_size: NonNegativeInt
    zonal_sum_default_column: str
    industrial_category_numbers: list[int]
    mix_use_categories: list[int]
    figures: Figures
    paths: Paths


class PiechartFigure(PiechartInteractive):
    nodata_country_color: str
    nodata_country_label: str


class SizesInteractiveMap(RunParams):
    figures: Figures
    paths: Paths


class PopAtRiskFigures(RunParams):
    threshold_value: NonNegativeInt
    min_display_population: NonNegativeInt
    save_dpi: PositiveInt
    country_id_column: str
    nodata_country_color: str
    nodata_country_label: str
    plot_outlier_quantiles: list[Fraction]
    plot_outlier_iqr_factor: PositiveFloat
    zoom_level: PositiveInt
    max_workers: PositiveInt
    paths: Paths


class Figure1(Section):
    min_plants: PositiveInt
    match_radius_m: PositiveInt
    paths: Paths


# --------------------------------------------------------------------------- #
# Sensitivity sweeps                                                          #
# --------------------------------------------------------------------------- #
class Sweep(Section):
    sweep_levels: list[Annotated[int, Field(ge=1, le=12)]] = Field(min_length=1)
    sweep_weight_funcs: list[WeightFunc] = Field(min_length=1)
    sweep_weight_methods: list[WeightMethod] = Field(min_length=1)
    sweep_rigid_buffers: list[NonNegativeInt] = Field(min_length=1)
    sweep_size_measures: list[SizeMeasure] = Field(min_length=1)


class ComparePopSweepHwEu(Section):
    max_workers: PositiveInt
    threshold: NonNegativeInt
    hw_weight: Fraction
    eu_reference_factor: PositiveFloat
    paths: Paths


class SweepVerRanking(ComparePopSweepHwEu):
    percent_verification: Fraction
    figures: Figures


# --------------------------------------------------------------------------- #
# Snakemake                                                                   #
# --------------------------------------------------------------------------- #
class JobResources(Strict):
    threads: PositiveInt
    mem_mb: PositiveInt
    runtime: PositiveInt  # minutes


class Workflow(Section):
    bing_annotate_shards: PositiveInt
    resources: dict[str, JobResources]
    paths: Paths


SECTIONS: dict[str, type[Section]] = {
    "correct_locations_w_OSM": CorrectLocationsWOSM,
    "run_manifest": RunManifest,
    "merge_seg_results": MergeSegResults,
    "final_data_merge": FinalDataMerge,
    "combine_watersheds": CombineWatersheds,
    "NEW_01_GENERATEGRIDS": Generategrids,
    "NEW_02_EXTRACTOSMDATAFULL_GEOJSON": ExtractOsm,
    "download_bing_annotate": DownloadBingAnnotate,
    "merge_annotations": MergeAnnotations,
    "annotations_inspection": AnnotationsInspection,
    "copy_falsy_images": CopyFalsyImages,
    "sample_annotations_by_class": SampleAnnotationsByClass,
    "NEW_03_WASTEWATERJOIN_GEOJSON": WastewaterJoin,
    "NEW_04_EXPORTGEOTIFF": ExportGeotiff,
    "download_pop": DownloadPop,
    "create_voronoi": CreateVoronoi,
    "add_pop": AddPop,
    "create_rasters": CreateRasters,
    "find_unserved_pop": FindUnservedPop,
    "find_diff_pop": FindDiffPop,
    "assign_rivers_to_basin": AssignRiversToBasin,
    "find_intersection_river": FindIntersectionRiver,
    "impact_polygons_pop": ImpactPolygonsPop,
    "find_pop_in_danger_pop": FindPopInDangerPop,
    "download_and_vectorize": DownloadAndVectorize,
    "find_unconnected_industrial_areas": FindUnconnectedIndustrialAreas,
    "verification_script": VerificationScript,
    "hw_comparison": HwComparison,
    "eu_comparison": EuComparison,
    "interactive_unconnected_industrial_map": InteractiveUnconnectedIndustrialMap,
    "composite_area_population_plots": CompositeAreaPopulationPlots,
    "piechart_figure": PiechartFigure,
    "piechart_interactive": PiechartInteractive,
    "sizes_interactive_map": SizesInteractiveMap,
    "pop_at_risk_figures": PopAtRiskFigures,
    "sweep": Sweep,
    "compare_pop_sweep_hw_eu": ComparePopSweepHwEu,
    "sweep_ver_ranking": SweepVerRanking,
    "figure1": Figure1,
    "workflow": Workflow,
}


class SettingsError(ValueError):
    """A config section that does not fit its model; the message lists every problem as section.key: reason."""


def parse(section, values):
    """The validated model of ``values`` (a resolved section of config.yaml)."""
    if section not in SECTIONS:
        raise SettingsError(f"config section {section!r} has no settings model in wwtp/settings.py")
    try:
        return SECTIONS[section].model_validate(values)
    except ValidationError as err:
        problems = "\n".join(f"  {'.'.join(str(p) for p in (section, *e['loc']))}: {e['msg']}"
                             + ("" if e["type"] == "missing" or isinstance(e["input"], dict) else f" (got {e['input']!r})")
                             for e in err.errors())
        raise SettingsError(f"config section {section!r} is not valid:\n{problems}") from None


def validate(section, values):
    """``values`` checked against the section's model, returned as a dict in the same key order."""
    dumped = parse(section, values).model_dump()
    return {key: dumped[key] for key in values}
