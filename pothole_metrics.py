import numpy as np


def classify_pothole_volume_m3(volume_m3: float) -> str:
    """
    Classify pothole severity based on volume in m³.
    """
    if volume_m3 < 0.02:
        return "Small"
    elif volume_m3 < 0.1:
        return "Medium"
    else:
        return "Large"


def estimate_volume_cm3(
    depth_region: np.ndarray,
    pixel_area_cm2: float,
    max_depth_cm: float,
) -> float:
    """
    Estimate volume in cm³ for a pothole region.

    depth_region: normalized depth values (0–1) inside bbox.
    pixel_area_cm2: area represented by each pixel.
    max_depth_cm: depth corresponding to normalized value 1.
    """
    if depth_region.size == 0:
        return 0.0

    mean_norm = float(depth_region.mean())
    mean_depth_cm = max(mean_norm * max_depth_cm, 0.1)
    volume_cm3 = mean_depth_cm * pixel_area_cm2 * depth_region.size
    return volume_cm3


def cost_time_from_volume(
    volume_cm3: float,
    rate_min_rs_per_m3: float,
    rate_max_rs_per_m3: float,
):
    """
    Compute severity, cost range (₹), and time (minutes) from volume.
    """
    volume_m3 = volume_cm3 / 1e6
    severity = classify_pothole_volume_m3(volume_m3)

    cost_min = volume_m3 * rate_min_rs_per_m3
    cost_max = volume_m3 * rate_max_rs_per_m3

    if severity == "Small":
        minutes = 15
    elif severity == "Medium":
        minutes = 30
    else:
        minutes = 45

    return severity, (round(cost_min), round(cost_max)), minutes
