"""River risk model against hand-computed answers.

- load ratio: people x organic load / discharge / limit, in units worked out by hand
- decay: a plume leaves its segment at C0 exp(-k L / v)
- confluence: mixing conserves the pollutant flux
"""
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st
from shapely.geometry import LineString

from wwtp.pop_at_risk_river_calculations import impact_polygons_pop as ip

pytestmark = pytest.mark.unit


def test_load_ratio_in_hand_computed_units():
    # 1,000 people x 60 g/day = 694.4 mg/s into 1 m3/s = 1,000 l/s -> 0.694 mg/l; limit 5 mg/l
    assert ip.calculate_load_ratio(1000, 1.0, org_per_pop=60.0, c_limit=5.0) == pytest.approx(60000 / 86.4 / 1000 / 5)


@given(q=st.one_of(st.floats(1e-4, 50), st.floats(-5, 0), st.just(float("nan"))),  # HydroRIVERS: >= 0.001 m3/s
       pop=st.floats(0, 1e6, allow_nan=False))
def test_scalar_and_vectorized_load_ratio_agree(q, pop):
    scalar = ip.calculate_load_ratio(pop, q)
    vector = ip.calculate_load_ratio(np.array([pop]), np.array([q]))[0]
    series = ip.calculate_load_ratio(pd.Series([pop]), pd.Series([q])).iloc[0]
    assert np.isfinite(scalar) and scalar >= 0
    assert scalar == pytest.approx(vector) == pytest.approx(series)


def test_plume_leaves_its_segment_at_exponential_decay_over_the_whole_length():
    length, start, lat, v = 1000.0, 3.0, 50.0, 0.01  # slow flow so the decay is visible
    ip.init_worker({1: (None, start)}, {1: LineString([(0, 0), (length, 0)])}, {1: lat}, {1: 0}, {1: 1.0})
    polygons, exit_load = ip.generate_single_segment_plume(1, lat, step_m=100.0, c_limit=5.0, velocity_m_s=v,
                                                           impact_radii=[1000])
    k = ip.calculate_kt(lat)
    assert polygons
    assert exit_load == pytest.approx(5.0 * start * np.exp(-k * length / (86400 * v)))


def test_decay_rate_follows_the_temperature_model():
    # 28 C at the equator -> k = 0.23 * 1.047 ** 8; 20 C (k = base) at arccos(20/28)
    assert ip.calculate_kt(0.0) == pytest.approx(0.23 * 1.047 ** 8)
    assert ip.calculate_kt(np.degrees(np.arccos(20 / 28))) == pytest.approx(0.23)


@given(c1=st.floats(0, 100), c2=st.floats(0, 100), q1=st.floats(0.3, 500), q2=st.floats(0.3, 500))
def test_confluence_conserves_pollutant_flux(c1, c2, q1, q2):
    q_down = q1 + q2
    c = ip.mix_downstream_concentration(0.0, c1, q1, q_down)
    c = ip.mix_downstream_concentration(c, c2, q2, q_down)
    assert c * q_down == pytest.approx(c1 * q1 + c2 * q2, rel=1e-9, abs=1e-9)


def test_inconsistent_discharge_never_concentrates_a_plume():
    # downstream discharge smaller than the tributary's (bad data): no concentration gain
    assert ip.mix_downstream_concentration(0.0, 10.0, 50.0, 5.0) == pytest.approx(10.0)
