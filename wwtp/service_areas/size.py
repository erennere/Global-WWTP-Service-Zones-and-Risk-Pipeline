"""Plant size in people and the service radius (config: size_measure, radius_calibration).

Part of the service-area stage (``python -m wwtp.create_voronoi``).
"""

import logging
import numpy as np
import pandas as pd
from ..geo_utils import parse_diameters_to_round_area

logger = logging.getLogger(__name__)


def calculate_area(df, only_round=False):
    """
    Calculate Voronoi region areas from assigned points.

    Computes area (in m²) for each Voronoi region based on point assignments.
    Optionally rounds values before calculating to handle overlapping regions.

    Args:
        df (pd.GeoDataFrame): Points with 'geometry' (Point), 'weight', and a site ID column
        only_round (bool): Round weight values before area calculation (default False)

    Returns:
        pd.DataFrame: The provided input dataframe with area-derived columns,
        including ``base_values`` required by ``create_weights``.

    Notes:
        Assumes 1 point per cell in Voronoi grid
        Area = point_count × cell_size (depends on Voronoi grid resolution)
        Rounding reduces numeric precision but may handle overlaps better
    """

    if df is None or df.empty:
        logger.warning("Input dataframe is empty, returning as-is")
        return df
    logger.debug(f"Calculating area for {len(df)} WWTP facilities")
    if "wwtp_area_rect" in df:
        df["wwtp_area_rect_2"] = df["wwtp_area_rect"].apply(
            lambda x: (
                np.sum([float(i) for i in str(x).strip().strip("[]").split() if i and i.lower() != "none"])
                if pd.notnull(x)
                else 0
            )
        )
        df["round_area"] = df["diameters"].apply(parse_diameters_to_round_area)
        # The service radius is calibrated on the full detected footprint, so it
        # does not change with only_round (which only affects the weights).
        df["footprint_m2"] = df["round_area"] + df["wwtp_area_rect_2"]
        if only_round:
            df["total_area"] = df["round_area"]
            logger.debug(f"Using round areas only for {len(df)} facilities")
        else:
            df["total_area"] = df["footprint_m2"]
            logger.debug(f"Combined round + building areas for {len(df)} facilities")
    else:
        logger.warning("No 'wwtp_area_rect' column found, using default area=1")
        df["total_area"] = 1
        df["footprint_m2"] = 0.0

    # Build detection-derived capacity proxy once so downstream weighting
    # logic can reuse these fields without recomputing.
    rect_counts = pd.to_numeric(
        df.get("num_detection_rect", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    circle_counts = pd.to_numeric(
        df.get("num_detection_circle", pd.Series(0, index=df.index)),
        errors="coerce",
    ).fillna(0)
    num_detections = (rect_counts.astype(int) + circle_counts.astype(int)).clip(lower=0)
    df["num_detections"] = num_detections
    df["capacity_proxy"] = df["total_area"] * np.sqrt(df["num_detections"])
    fallback_mean = df["capacity_proxy"].mean()
    df["base_values"] = df["capacity_proxy"].replace(0.0, np.nan).fillna(fallback_mean)

    return df


LAGOON_COLUMN = "lagoon_detected"


SIZE_MEASURES = ("reported", "footprint")


_WARNED_MISSING_SIZE_COLUMNS = set()  # warn once per process, not once per basin group


def _truthy(series):
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def reported_size_people(df, columns, skip_quality):
    """Reported plant size in people: the first positive value among ``columns``.

    Parameters
    ----------
    df : pandas.DataFrame
        Plant records.
    columns : sequence of str
        Columns in order of preference (e.g. ``POP_SERVED``, ``EPA POP 2022``,
        ``uwwLoadEnteringUWWTP``). Missing columns are skipped.
    skip_quality : dict
        ``{column: {quality_column: [codes]}}``: values of ``column`` are ignored
        on rows whose ``quality_column`` is one of ``codes`` (HydroWASTE
        ``QUAL_POP`` 4: people within 11 km of the plant, not a plant record).

    Returns
    -------
    numpy.ndarray
        People per row; NaN where no column holds a positive value.
    """
    for column in columns:
        if column not in df.columns and column not in _WARNED_MISSING_SIZE_COLUMNS:
            _WARNED_MISSING_SIZE_COLUMNS.add(column)
            logger.warning(
                "Reported size column %r is not in the plant data; those plants use the footprint estimate", column
            )
    values = {column: pd.to_numeric(df[column], errors="coerce") for column in columns if column in df.columns}
    for column, rules in skip_quality.items():
        for quality_column, codes in rules.items():
            if column in values and quality_column in df.columns:
                quality = pd.to_numeric(df[quality_column], errors="coerce")
                values[column] = values[column].where(~quality.isin([float(c) for c in codes]))
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for column in columns:
        if column in values:
            out = out.fillna(values[column].where(values[column] > 0))
    return out.to_numpy(dtype=float)


def plant_size_people(df, size_measure, footprint_to_people, reported_size_columns, reported_size_skip, **_):
    """Plant size in people, used for both the service radius and the Voronoi weight.

    Parameters
    ----------
    df : pandas.DataFrame
        Plant records with ``footprint_m2`` (from ``calculate_area``; falls back to
        ``total_area``) and optionally ``lagoon_detected`` and the reported columns.
    size_measure : {'reported', 'footprint'}
        ``reported``: the reported size where there is one (see
        ``reported_size_people``), the footprint estimate elsewhere.
        ``footprint``: the footprint estimate for every plant.
    footprint_to_people : dict
        ``intercept`` and ``slope`` of ``log10 people = intercept + slope * log10
        footprint_m2`` and ``unsized_people`` for lagoon plants and plants
        without detections, whose footprint says nothing about their size.
    reported_size_columns, reported_size_skip
        Passed to ``reported_size_people`` (used with ``reported`` only).

    Returns
    -------
    numpy.ndarray
        People per row (always > 0).
    """
    if size_measure not in SIZE_MEASURES:
        raise ValueError(f"size_measure must be one of {SIZE_MEASURES}, got {size_measure!r}")
    if "footprint_m2" in df.columns:
        footprint = pd.to_numeric(df["footprint_m2"], errors="coerce")
    elif "total_area" in df.columns:
        footprint = pd.to_numeric(df["total_area"], errors="coerce")
    else:
        footprint = pd.Series(0.0, index=df.index)
    footprint = np.nan_to_num(footprint.to_numpy(dtype=float), nan=0.0)
    lagoon = _truthy(df[LAGOON_COLUMN]).to_numpy() if LAGOON_COLUMN in df.columns else np.zeros(len(df), bool)
    sized = (footprint > 0) & ~lagoon
    people = np.where(
        sized,
        np.power(
            10.0,
            float(footprint_to_people["intercept"])
            + float(footprint_to_people["slope"]) * np.log10(np.where(sized, footprint, 1.0)),
        ),
        float(footprint_to_people["unsized_people"]),
    )
    if size_measure == "reported":
        reported = reported_size_people(df, reported_size_columns, reported_size_skip)
        people = np.where(np.isfinite(reported), reported, people)
    return people


def service_radius_m(size_people, radius_at_10k_people_km, exponent, max_m):
    """Service radius (m): ``radius_at_10k_people_km * (people / 10,000) ** exponent``, at most ``max_m``."""
    people = np.asarray(size_people, dtype=float)
    radius_km = float(radius_at_10k_people_km) * np.power(people / 1e4, float(exponent))
    return np.minimum(radius_km * 1000.0, float(max_m))


def calculate_buffer(df, weights, *args, **kwargs):
    """Per-site service radius in metres.

    Parameters
    ----------
    df : geopandas.GeoDataFrame
        Input sites (see ``plant_size_people`` for the columns used).
    weights : numpy.ndarray
        Voronoi weights; unused, kept for the ``calculate_buffer_fn`` signature.
    **kwargs
        ``buffer`` (m), ``dynamic_buffering``, ``max_buffer`` (m), and with
        dynamic buffering ``size_measure``, ``footprint_to_people``,
        ``reported_size_columns``, ``reported_size_skip``, ``radius_exponent``
        and ``radius_at_10k_people_km`` (starter picks the latter from
        ``radius_calibration`` for the size measure).

    Returns
    -------
    numpy.ndarray
        Radius in metres for each row of ``df``.

    Notes
    -----
    With ``dynamic_buffering`` false every site gets ``buffer`` (at most
    ``max_buffer``). Otherwise the radius grows with plant size in people. The
    exponent and the radius at 10,000 people were chosen on US census
    population in EPA sourced sewersheds (``wwtp.calibration.service_radius``);
    the GB catchments are the independent test.
    """
    max_buffer = kwargs["max_buffer"]
    if not kwargs["dynamic_buffering"]:
        fixed = float(min(kwargs["buffer"], max_buffer))
        return np.full(len(df), fixed, dtype=float)
    people = plant_size_people(df, **kwargs)
    radius = service_radius_m(people, kwargs["radius_at_10k_people_km"], kwargs["radius_exponent"], max_buffer)
    logger.debug(
        "Service radius %.0f-%.0f m (median %.0f m) for %d sites",
        radius.min(),
        radius.max(),
        np.median(radius),
        len(df),
    )
    return radius
