# map_utils.py
import csv
from pathlib import Path

IMAGE_LOCATIONS_CSV = Path("image_locations_ap.csv")


def load_pothole_points():
    """Load all synthetic pothole locations from CSV."""
    points = []
    if not IMAGE_LOCATIONS_CSV.exists():
        return points

    with IMAGE_LOCATIONS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (KeyError, ValueError):
                continue
            points.append(
                {
                    "image_path": row.get("image_path", ""),
                    "place_name": row.get("place_name", ""),
                    "lat": lat,
                    "lon": lon,
                }
            )
    return points
