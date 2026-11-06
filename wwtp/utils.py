"""Generic, domain-agnostic helpers shared across pipeline modules.

Unlike geo_utils.py, nothing here depends on geometry/CRS concepts -
filesystem, SQL string-building, and reference-data lookups only.
"""

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pycountry
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG_LEVEL_ENV_VAR = "WWTP_SERVICE_PIPELINE_LOG_LEVEL"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

DEFAULT_REQUEST_TIMEOUT_SECONDS = 60


def requests_session_with_retries(total_retries=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504)):
    """Build a ``requests.Session`` with bounded automatic retry/backoff.

    Centralizes the retry policy that several download sites previously
    lacked entirely (no timeout, no retry at all) or hand-rolled unsafely (a
    module-global sleep counter reassigned from inside a thread pool). Callers
    still pass their own ``timeout`` to each request - a session does not set
    one implicitly.
    """
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def default_cpu_workers():
    """Return the SLURM-allocated CPU count (``SLURM_CPUS_PER_TASK``), else the machine's.

    The Snakemake workflow sets OMP/OpenBLAS/MKL/NUMEXPR thread counts to a job's
    threads the same way, so numerical libraries and Python worker pools size
    themselves to the allocation rather than the whole node.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass
    return os.cpu_count() or 1


def configure_logging(level=None):
    """Install the single project-wide logging configuration.

    Call this from an entry point's ``__main__`` block only - never at import
    time. ``logging.basicConfig`` is a no-op once the root logger has a handler,
    so per-module calls meant whichever module imported first silently decided
    the level for every other module.

    ``level`` wins if given; otherwise the ``WWTP_SERVICE_PIPELINE_LOG_LEVEL``
    environment variable is used, defaulting to INFO.
    """
    if level is None:
        level_name = os.environ.get(LOG_LEVEL_ENV_VAR, "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    return level


def ensure_output_dir_for_file(filepath):
    """Create the parent directory for an output file path if needed."""
    parent = Path(filepath).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def atomic_output(filepath):
    """Yield a temporary path for ``filepath`` and move it onto ``filepath`` when the block succeeds.

    A reader (another job, Snakemake, a crashed rerun) then sees either the old
    file or the complete new one, never a half-written one. The temporary file
    has the same name inside a private folder next to the target, so formats
    and GeoPackage layer names (taken from the file name) stay as they were, and
    the final rename stays on one filesystem. On an exception the folder is
    removed and ``filepath`` is left as it was.
    """
    target = os.path.abspath(filepath)
    ensure_output_dir_for_file(target)
    directory, name = os.path.split(target)
    tmpdir = tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=directory)
    try:
        tmp = os.path.join(tmpdir, name)
        yield tmp
        os.replace(tmp, target)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_iso_codes():
    """Build ISO-code lookup dictionaries used by population workflows.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names)``.
    """
    alpha_3_to_2 = {}
    alpha_2_to_3 = {}
    alpha_3_to_names = {}
    alpha_2_to_names = {}
    for country in pycountry.countries:
        alpha_3_to_2[country.alpha_3.upper()] = country.alpha_2.upper()
        alpha_2_to_3[country.alpha_2.upper()] = country.alpha_3.upper()
        alpha_3_to_names[country.alpha_3.upper()] = country.name
        alpha_2_to_names[country.alpha_2.upper()] = country.name
    return alpha_3_to_2, alpha_2_to_3, alpha_3_to_names, alpha_2_to_names


INDUSTRIAL_CATEGORY_COLUMN = "category_number"


def industrial_category_mask(df, industrial_categories, mix_use_categories, column=INDUSTRIAL_CATEGORY_COLUMN):
    """Return a boolean mask of rows whose category is industrial or mixed-use.

    Mixed-use sites count as industrial everywhere. Before this was shared, only
    the unconnected-industrial layer included them, so the pie charts' IND/RES
    split disagreed with it for every mixed-use plant.

    Returns ``None`` when ``column`` is absent, so callers can log and decide -
    rather than each site silently skipping, raising, or letting pandas raise.
    """
    if column not in df.columns:
        return None

    wanted = {str(c) for c in (industrial_categories or ())} | {str(c) for c in (mix_use_categories or ())}
    return df[column].astype(str).isin(wanted)


def select_industrial_categories(
    df, industrial_categories, mix_use_categories, keep=True, column=INDUSTRIAL_CATEGORY_COLUMN, logger=None
):
    """Keep only industrial/mixed-use rows (``keep=True``) or drop them.

    ``keep=True`` is the industrial pipeline's direction; ``keep=False`` is what
    the Voronoi pipeline wants when ``remove_industrial`` is set. Both use the
    same category set so the two layers cannot disagree.

    Rows are returned unchanged when the category column is missing, and the fact
    is logged rather than passing silently.
    """
    log = logger or logging.getLogger(__name__)
    mask = industrial_category_mask(df, industrial_categories, mix_use_categories, column=column)
    if mask is None:
        log.warning(
            "Column '%s' is absent; industrial %s filter skipped and all %d row(s) kept.",
            column,
            "keep-only" if keep else "discard",
            len(df),
        )
        return df

    selected = df[mask if keep else ~mask].copy()
    log.info(
        "Industrial %s filter on '%s': %d of %d row(s) retained.",
        "keep-only" if keep else "discard",
        column,
        len(selected),
        len(df),
    )
    return selected


def reported_population_mask(df, column="QUAL_POP"):
    """True where HydroWASTE marks POP_SERVED as reported (``QUAL_POP == 1``).

    The column is read as a number, so ``1``, ``1.0`` and ``'1.0'`` all count.
    A missing column raises: the comparisons used to backfill it with ``'1.0'``
    or skip the filter, which silently treated every estimated value as
    reported.
    """
    if column not in df.columns:
        raise KeyError(
            f"Column '{column}' is required to select plants with reported population; "
            "it is missing from the comparison input."
        )
    return pd.to_numeric(df[column], errors="coerce").eq(1)


@contextmanager
def duckdb_connection(in_memory=False, read_only=False):
    """Yield a DuckDB connection, owning the scratch database's whole lifecycle.

    The file is created with a collision-free name inside a temporary directory
    and the directory is removed on every exit path, including exceptions.
    Callers never name or delete a ``.db`` themselves - the previous per-site
    versions used a fixed ``temp_duckdb.db`` (two parallel workers clobbered each
    other) or a random name that was never cleaned up.

    Pass ``in_memory=True`` for queries that do not need a backing file.

    Load the SPATIAL extension on the yielded connection with
    ``geo_utils.ensure_duckdb_spatial(connection)``.
    """
    if in_memory:
        connection = duckdb.connect(":memory:")
        try:
            yield connection
        finally:
            connection.close()
        return

    with tempfile.TemporaryDirectory(prefix="wwtp_duckdb_") as tmpdir:
        database = os.path.join(tmpdir, "scratch.db")
        connection = duckdb.connect(database=database, read_only=read_only)
        try:
            yield connection
        finally:
            connection.close()


def quote_sql_identifier(name):
    """Quote a SQL identifier, escaping any embedded double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


ZONAL_SUM_SUFFIX = "_zonal_sum"


def parse_zonal_sum_year(column):
    """Return the year encoded in a ``*_zonal_sum`` column name, or ``None``.

    Columns come in two shapes: ``2024_zonal_sum`` and, once
    ``find_pop_in_danger_pop.rename_cols`` has prefixed them with the impact
    radius, ``500_2024_zonal_sum``. Every previous copy of this parsing read
    ``col.split('_')[0]``, so the prefixed form was read as year ``500`` by all
    five call sites. The year is the token directly before the suffix.
    """
    name = str(column)
    if not name.endswith(ZONAL_SUM_SUFFIX):
        return None
    token = name[: -len(ZONAL_SUM_SUFFIX)].rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError:
        return None


def resolve_latest_zonal_sum_column(
    df,
    preferred=None,
    exclude_years=(),
    lexicographic=False,
    keep_unparseable=True,
    required=True,
    missing_message=None,
):
    """Resolve the zonal-sum column to use, preferring ``preferred`` if present.

    The five previous implementations differed in exactly three ways, so those
    are the parameters rather than hidden behavior:

    - ``exclude_years`` - ``compare_pop_sweep_hw_eu`` ignores 2014, the others do not.
    - ``lexicographic`` - ``sizes_interactive_map`` picked ``sorted(cols)[-1]``
      instead of ranking by parsed year.
    - ``keep_unparseable`` - most callers rank a name whose year will not parse
      last (year ``-1``) but still accept it; the sweep skips it entirely.

    Returns ``(year, column)``. ``year`` is ``None`` when it could not be parsed
    (including every ``lexicographic`` result). Raises ``KeyError`` when nothing
    matches and ``required`` is set, otherwise returns ``(None, None)``.
    """
    if preferred is not None and preferred in df.columns:
        return parse_zonal_sum_year(preferred), preferred

    zonal_cols = [c for c in df.columns if str(c).endswith(ZONAL_SUM_SUFFIX)]

    if lexicographic:
        if zonal_cols:
            chosen = sorted(zonal_cols)[-1]
            return parse_zonal_sum_year(chosen), chosen
        candidates = []
    else:
        excluded = {int(y) for y in exclude_years}
        candidates = []
        for col in zonal_cols:
            year = parse_zonal_sum_year(col)
            if year is not None and year in excluded:
                continue
            if year is None:
                if not keep_unparseable:
                    continue
                year = -1
            candidates.append((year, col))

    if not candidates:
        if required:
            raise KeyError(missing_message or f"No '*{ZONAL_SUM_SUFFIX}' column found.")
        return None, None

    year, column = max(candidates, key=lambda pair: pair[0])
    return (None if year == -1 else year), column


def _clean_numeric_series(values, positive_only=False):
    """Coerce array-like input to a finite, NaN-free numeric Series."""
    clean = pd.Series(pd.to_numeric(values, errors="coerce")).replace([np.inf, -np.inf], np.nan).dropna()
    if positive_only:
        clean = clean[clean > 0]
    return clean


def _quantile_iqr_cut(clean, quantile_range, iqr_factor):
    """Return the (low, high) cut combining an IQR fence with a quantile clip.

    The IQR fence degenerates to the full range when the IQR is zero, and the
    tighter of the two cuts wins on each side.
    """
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    if iqr > 0:
        iqr_low = q1 - iqr_factor * iqr
        iqr_high = q3 + iqr_factor * iqr
    else:
        iqr_low = float(clean.min())
        iqr_high = float(clean.max())

    q_low = float(clean.quantile(quantile_range[0]))
    q_high = float(clean.quantile(quantile_range[1]))

    return max(iqr_low, q_low), min(iqr_high, q_high)


def robust_bounds(values, quantile_range=(0.02, 0.98), iqr_factor=1.5, positive_only=False):
    """Estimate robust plotting bounds by combining quantile and IQR filtering.

    Raises on empty input and widens a degenerate result back to the observed
    range, because a colour scale must always produce two usable numbers. The
    clipping sibling, :func:`clip_to_robust_bounds`, deliberately does neither -
    see its docstring.
    """
    clean = _clean_numeric_series(values, positive_only=positive_only)
    if clean.empty:
        raise ValueError("No valid values available to compute robust bounds.")

    low, high = _quantile_iqr_cut(clean, quantile_range, iqr_factor)

    if positive_only:
        low = max(low, np.finfo(float).tiny)

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(clean.min())
        high = float(clean.max())
        if positive_only:
            low = max(low, np.finfo(float).tiny)
        if high <= low:
            high = low * (10.0 if positive_only else 1.000001)

    return low, high


def clip_to_robust_bounds(series, lower_q, upper_q, iqr_factor=1.0):
    """Drop outliers with the same combined quantile/IQR filtering, and return the kept values.

    Two deliberate differences from :func:`robust_bounds`, which is why they are
    separate entry points rather than one function with flags:

    * the default ``iqr_factor`` is 1.0 rather than 1.5 - histogram trimming is
      tighter than colour-scale bounding, and this is a choice, not a typo;
    * empty input passes through as an empty Series instead of raising, and a
      degenerate cut is not widened - dropping every point is an acceptable
      outcome for a histogram, whereas a colour scale cannot work with it.
    """
    if not (0 <= lower_q < upper_q <= 1):
        raise ValueError("Quantile bounds must satisfy 0 <= lower_q < upper_q <= 1")

    clean = _clean_numeric_series(series)
    if clean.empty:
        return clean

    low, high = _quantile_iqr_cut(clean, (lower_q, upper_q), iqr_factor)
    return clean[(clean >= low) & (clean <= high)]
