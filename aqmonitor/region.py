"""
Decide whether a point on the map is inside the monitored region.

OpenAQ can only be searched with a rectangle, and a rectangle around Nova Scotia also catches Moncton, Saint John
and Prince Edward Island. ``OUTLINE`` is a rough hand-drawn outline of the province whose edges run down the
middle of the Bay of Fundy and the Northumberland Strait, so it separates Nova Scotia's stations from its
neighbours' without needing a detailed coastline. Sable Island is inside it.

``contains`` uses the ray-casting rule: draw a line from the point straight to the east and count how many edges
of the outline it crosses. An odd count means the point is inside.

To monitor a different place, replace ``OUTLINE`` with corner points around it, listed in order around the edge.
"""

OUTLINE: tuple[tuple[float, float], ...] = (
    (-66.50, 43.30),
    (-66.50, 44.75),
    (-65.30, 45.20),
    (-64.75, 45.50),
    (-64.30, 45.83),
    (-64.04, 46.01),
    (-63.50, 46.00),
    (-62.70, 45.85),
    (-62.20, 46.00),
    (-61.70, 46.50),
    (-61.20, 47.10),
    (-59.60, 47.10),
    (-59.60, 43.30),
)
"""Corner points as (longitude, latitude)."""


def bounding_box() -> tuple[float, float, float, float]:
    """The smallest rectangle around the outline, as (west, south, east, north)."""
    longitudes = [point[0] for point in OUTLINE]
    latitudes = [point[1] for point in OUTLINE]
    return min(longitudes), min(latitudes), max(longitudes), max(latitudes)


def contains(longitude: float, latitude: float) -> bool:
    """Return True when the point lies inside the outline."""
    inside = False
    for index, (lon_a, lat_a) in enumerate(OUTLINE):
        lon_b, lat_b = OUTLINE[index - 1]
        if (lat_a > latitude) != (lat_b > latitude):
            crossing = lon_a + (latitude - lat_a) * (lon_b - lon_a) / (lat_b - lat_a)
            if longitude < crossing:
                inside = not inside
    return inside
