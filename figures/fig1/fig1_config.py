"""Figure 1 settings from config.yaml (section ``figure1``), for the scripts in this folder.

``python fig1_config.py <key>`` prints one value for the render_*.sh wrappers:
``min_plants``, ``match_radius_m``, ``tag`` (the radius tag of the matched files,
e.g. r10000) or a path key (``match_dir``, ``cartogram_dir``, ``combined_dir``).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.starter import load_config  # noqa: E402

FIG1 = load_config(script_name="figure1")
MIN_PLANTS = int(FIG1["min_plants"])
MATCH_RADIUS_M = float(FIG1["match_radius_m"])
TAG = f"r{int(MATCH_RADIUS_M)}"  # suffix of the matched files: plants_with_units_<TAG>.csv
PATHS = FIG1["paths"]


def enough_plants(n, minimum=MIN_PLANTS):
    """Whether a unit has enough plants to be coloured (fewer: hatched). Works on numbers and Series."""
    return n >= minimum


def main():
    key = sys.argv[1]
    values = {"min_plants": MIN_PLANTS, "match_radius_m": int(MATCH_RADIUS_M), "tag": TAG}
    print(values[key] if key in values else PATHS[key])


if __name__ == "__main__":
    main()
