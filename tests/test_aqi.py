import math

import pytest

from aqmonitor import aqi


def test_aqi_zero_concentration_is_zero() -> None:
    assert aqi.aqi(0.0) == 0


def test_aqi_top_of_good_band_is_fifty() -> None:
    assert aqi.aqi(9.0) == 50


def test_aqi_bottom_of_moderate_band_is_fifty_one() -> None:
    assert aqi.aqi(9.1) == 51


def test_aqi_top_of_moderate_band_is_one_hundred() -> None:
    assert aqi.aqi(35.4) == 100


def test_aqi_top_of_unhealthy_band_is_two_hundred() -> None:
    assert aqi.aqi(125.4) == 200


def test_aqi_above_scale_is_capped_at_five_hundred() -> None:
    assert aqi.aqi(900.0) == 500


def test_aqi_negative_concentration_is_treated_as_zero() -> None:
    assert aqi.aqi(-2.0) == 0


def test_aqi_missing_value_returns_none() -> None:
    assert aqi.aqi(math.nan) is None


def test_aqi_truncates_concentration_to_one_decimal() -> None:
    assert aqi.aqi(9.09) == aqi.aqi(9.0)


def test_aqi_never_decreases_as_concentration_rises() -> None:
    values = [aqi.aqi(step / 10.0) for step in range(0, 4000)]
    assert values == sorted(values)


@pytest.mark.parametrize(
    ("concentration", "expected"),
    [
        (4.0, "Good"),
        (9.0, "Good"),
        (9.1, "Moderate"),
        (35.5, "Unhealthy for Sensitive Groups"),
        (55.5, "Unhealthy"),
        (125.5, "Very Unhealthy"),
        (225.5, "Hazardous"),
        (5000.0, "Hazardous"),
    ],
)
def test_category_concentration_maps_to_expected_band(concentration: float, expected: str) -> None:
    assert aqi.category(concentration).name == expected


def test_category_missing_value_is_unknown() -> None:
    assert aqi.category(None) is aqi.UNKNOWN


def test_categories_bands_leave_no_gap_wider_than_one_tenth() -> None:
    gaps = [later.conc_low - earlier.conc_high for earlier, later in zip(aqi.CATEGORIES, aqi.CATEGORIES[1:])]
    assert all(math.isclose(gap, 0.1, abs_tol=1e-9) for gap in gaps)
