import pytest

from aqmonitor import region


@pytest.mark.parametrize(
    ("place", "longitude", "latitude"),
    [
        ("Halifax", -63.57, 44.65),
        ("Sydney", -60.17, 46.14),
        ("Yarmouth", -66.12, 43.84),
        ("Amherst", -64.21, 45.83),
        ("Pictou", -62.70, 45.68),
        ("Antigonish", -61.99, 45.62),
        ("Cheticamp", -61.01, 46.63),
        ("Sable Island", -60.00, 43.93),
    ],
)
def test_contains_nova_scotia_place_is_inside(place: str, longitude: float, latitude: float) -> None:
    assert region.contains(longitude, latitude), place


@pytest.mark.parametrize(
    ("place", "longitude", "latitude"),
    [
        ("Moncton", -64.79, 46.10),
        ("Saint John", -66.06, 45.27),
        ("Sackville NB", -64.37, 45.90),
        ("Charlottetown", -63.13, 46.24),
        ("Southampton PEI", -62.58, 46.39),
        ("Boston", -71.06, 42.36),
    ],
)
def test_contains_neighbouring_place_is_outside(place: str, longitude: float, latitude: float) -> None:
    assert not region.contains(longitude, latitude), place


def test_bounding_box_surrounds_every_outline_point() -> None:
    west, south, east, north = region.bounding_box()
    assert all(west <= lon <= east and south <= lat <= north for lon, lat in region.OUTLINE)
