"""Figure 1 settings from config.yaml (section ``figure1``), for the scripts in this folder.

``python fig1_config.py <key>`` prints one value for the render_*.sh wrappers:
``min_plants``, ``match_radius_m``, ``tag`` (the radius tag of the matched files,
e.g. r10000) or a path key (``match_dir``, ``cartogram_dir``, ``combined_dir``).
"""

import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from wwtp.starter import load_config  # noqa: E402

FIG1 = load_config(script_name="figure1")
MIN_PLANTS = int(FIG1["min_plants"])
MATCH_RADIUS_M = float(FIG1["match_radius_m"])
TAG = f"r{int(MATCH_RADIUS_M)}"  # suffix of the matched files: plants_with_units_<TAG>.csv
PATHS = FIG1["paths"]


def enough_plants(n, minimum=MIN_PLANTS):
    """Whether a unit has enough plants to be coloured (fewer: hatched). Works on numbers and Series."""
    return n >= minimum


def pfaf_lookup(hb_dir):
    """HYBAS_ID -> PFAF_ID from every ``hybas_lev<NN>_*attributes.csv`` in the HydroBASINS folder.

    Plants and areas carry the HYBAS_ID of the level their file was made at: level 9 (first release), level 8
    (the non-served areas of 2 Nov), level 6 (plant list v3, the pipeline's basins: hybas_lev06_pipeline_
    attributes.csv, written by 00a_release_inputs.py basins). Digits 2-3 of an ID are its level, so the
    tables of different levels never share an ID; the first k digits of any PFAF_ID are its level-k basin.
    """
    import pandas as pd

    tables = sorted(glob.glob(os.path.join(hb_dir, "hybas_lev[0-9][0-9]_*attributes.csv")))
    assert tables, f"no hybas_lev<NN>_*attributes.csv in {hb_dir}"
    t = pd.concat([pd.read_csv(f, usecols=["HYBAS_ID", "PFAF_ID"]) for f in tables], ignore_index=True)
    t = t.astype("int64").drop_duplicates()
    clash = t.HYBAS_ID.duplicated(keep=False)
    assert not clash.any(), f"{int(clash.sum())} HYBAS_IDs with two PFAF_IDs in {tables}"
    return dict(zip(t.HYBAS_ID, t.PFAF_ID))


def main():
    key = sys.argv[1]
    values = {"min_plants": MIN_PLANTS, "match_radius_m": int(MATCH_RADIUS_M), "tag": TAG}
    print(values[key] if key in values else PATHS[key])


if __name__ == "__main__":
    main()
