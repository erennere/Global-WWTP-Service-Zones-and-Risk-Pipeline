"""Column conventions shared by the static and interactive piechart figures.

These helpers encode figure-layer conventions (the ``IND_``/``RES_`` prefixes, the
``population_served_index`` fallback chain, the marker-size mapping) rather than
general geometry or config logic, so they live beside the figure scripts instead
of in ``utils.py``/``geo_utils.py``.

``piechart_figure`` and ``piechart_interactive`` had drifted copies of all three.
The divergences are kept, but as explicit parameters: ``stats`` for the
aggregation, ``scale``/``degenerate``/``floor_nonpositive`` for the size mapping.
"""

import numpy as np
import pandas as pd

#: Summary statistics the static figure aggregates; the interactive map needs sums only.
FULL_STATS = ("sum", "mean", "median", "std")
SUM_ONLY = ("sum",)


def aggregate_by_country(gdf, country_column, agg_column, industrial_column=None,
                         is_pop=False, stats=FULL_STATS):
    """Aggregate facility-level attributes to country-level summary statistics.

    With ``is_pop=True`` the frame is grouped by country alone. Otherwise it is
    also split on ``industrial_column`` and the two halves are merged side by side
    under ``IND_``/``RES_`` prefixes.

    ``stats`` selects which pandas aggregations are produced; the interactive map
    only ever reads the sums.
    """
    gdf = gdf.copy()
    agg_dict = {f"{agg_column}_{stat}": stat for stat in stats}

    if is_pop:
        gdf = gdf.dropna(subset=[country_column, agg_column])
        return gdf.groupby(country_column)[agg_column].agg(**agg_dict).reset_index()

    if industrial_column is None:
        raise ValueError("industrial_column must be provided when is_pop=False")

    gdf = gdf.dropna(subset=[country_column, agg_column, industrial_column])
    grouped = gdf.groupby([country_column, industrial_column])[agg_column].agg(**agg_dict).reset_index()

    ind = grouped[grouped[industrial_column] == True].drop(columns=[industrial_column]).reset_index(drop=True)
    res = grouped[grouped[industrial_column] != True].drop(columns=[industrial_column]).reset_index(drop=True)
    ind = ind.rename(columns={c: f"IND_{c}" for c in ind.columns if c != country_column})
    res = res.rename(columns={c: f"RES_{c}" for c in res.columns if c != country_column})
    return res.merge(ind, on=country_column, how="left")


def calculate_size(value, min_value, max_value, min_size, max_size, scale='log',
                   degenerate='mid', floor_nonpositive=False):
    """Map a value onto a marker size in ``[min_size, max_size]``.

    Parameters carrying the two callers' deliberate differences:

    ``scale``
        ``'log'`` (static figure) or ``'linear'`` (interactive map).
    ``degenerate``
        What to return when ``max_value <= min_value`` and the mapping is
        undefined: ``'mid'`` (static) or ``'min'`` (interactive).
    ``floor_nonpositive``
        Clamp non-positive values to ``min_size`` even in linear mode, as the
        interactive map does. Log mode always floors them - the logarithm is
        undefined there.

    The non-finite guard applies to both callers: the interactive copy lacked it
    and let ``NaN`` reach the emitted SVG as a marker radius.
    """
    if not np.isfinite(value) or not np.isfinite(min_value) or not np.isfinite(max_value):
        return min_size

    if floor_nonpositive and value <= 0:
        return min_size

    if max_value <= min_value:
        if degenerate == 'mid':
            return (min_size + max_size) / 2.0
        if degenerate == 'min':
            return min_size
        raise ValueError("Invalid degenerate")

    if scale == 'log':
        if value <= 0 or min_value <= 0 or max_value <= 0:
            return min_size
        return (np.log(value) - np.log(min_value)) / (np.log(max_value) - np.log(min_value)) * (max_size - min_size) + min_size
    if scale == 'linear':
        return (value - min_value) / (max_value - min_value) * (max_size - min_size) + min_size
    raise ValueError("Invalid scale")


def ensure_population_percentage_column(df, preferred_col="population_served_index",
                                        zonal_sum_col="2024_zonal_sum"):
    """Ensure a population-served percentage column exists and return its name."""
    if preferred_col in df.columns:
        return preferred_col

    # Common fallback: derive from absolute served/total population columns.
    if {'population_served', 'population_total'}.issubset(df.columns):
        denom = pd.to_numeric(df['population_total'], errors='coerce').replace(0, np.nan)
        num = pd.to_numeric(df['population_served'], errors='coerce')
        df[preferred_col] = (num / denom).fillna(0)
        return preferred_col

    # Secondary fallback: use the aggregated latest zonal sum as the served estimate.
    zonal_sum_sum_col = f"{zonal_sum_col}_sum"
    if {zonal_sum_sum_col, 'population_total'}.issubset(df.columns):
        denom = pd.to_numeric(df['population_total'], errors='coerce').replace(0, np.nan)
        num = pd.to_numeric(df[zonal_sum_sum_col], errors='coerce')
        df[preferred_col] = (num / denom).fillna(0)
        return preferred_col

    raise KeyError(
        "Could not derive population-served percentage. Expected one of: "
        "'population_served_index', ('population_served' + 'population_total'), "
        f"or ('{zonal_sum_sum_col}' + 'population_total')."
    )
