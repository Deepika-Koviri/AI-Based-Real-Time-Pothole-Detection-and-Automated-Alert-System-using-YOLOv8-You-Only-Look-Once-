import numpy as np

# Example thresholds from IJISRT paper idea: volume‑based class. [file:23]
def classify_pothole(volume_cm3: float) -> str:
    if volume_cm3 < 1000:
        return "Small"
    elif volume_cm3 < 10000:
        return "Medium"
    else:
        return "Large"

def estimate_volume(depth_region: np.ndarray, pixel_area_cm2: float) -> float:
    """
    Approximate volume = mean_depth_cm * area_cm2.
    depth_region: normalized 0–1 depth; assume max depth ~ D_max_cm.
    """
    D_max_cm = 20.0  # tune later
    mean_norm = float(depth_region.mean())
    mean_depth_cm = mean_norm * D_max_cm
    return mean_depth_cm * pixel_area_cm2

def cost_time_rules(volume_cm3: float):
    """
    Simple rule‑based cost/time inspired by IJISRT:
    Small: 20 min, 1400–1600 INR
    Medium: 30–40 min, 2700–3000 INR
    Large: 50 min, 5000–7000 INR
    """
    cat = classify_pothole(volume_cm3)
    if cat == "Small":
        return cat, (1400, 1600), 20
    if cat == "Medium":
        return cat, (2700, 3000), 35
    return cat, (5000, 7000), 50
