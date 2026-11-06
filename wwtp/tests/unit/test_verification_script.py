"""Tests for verification_script (moved from the coverage-campaign files)."""

import pandas as pd
import pytest
from wwtp.pop_validation_scripts import verification_script

pytestmark = pytest.mark.unit


def test_find_verification_watersheds_marks_single_and_verification_flags():
    df = pd.DataFrame(
        {
            "HYBAS_ID": [101, 101, 202],
            "total_area": [10.0, 0.0, 5.0],
        }
    )

    result = verification_script.find_verification_watersheds(df, percent_verification=0.5)

    assert result["is_single_points"].tolist() == [False, False, True]
    assert result["use_verify"].tolist() == [True, False, False]
    assert result["watersheds_chosen"].tolist() == [True, True, False]
