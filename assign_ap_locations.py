import csv
import random
from pathlib import Path

DATA_ROOT = Path("reduced_dataset")
OUTPUT_CSV = "image_locations_ap.csv"

# ---- DEMO SEVERITY: random distribution ----
# 50% Small, 30% Medium, 20% Large
def random_severity() -> str:
    r = random.random()
    if r < 0.5:
        return "Small"
    elif r < 0.8:
        return "Medium"
    else:
        return "Large"


# ---- Visakhapatnam major road points (on roads, not water) ----
# lat, lon, place_name
VIZAG_ROAD_POINTS = [
    # NH‑16 Anandapuram – Hanumanthawaka stretch
    (17.9033, 83.3625, "Anandapuram Junction"),        # NH16 at Anandapuram
    (17.8700, 83.3520, "NH16 Near Madhurawada"),
    (17.8450, 83.3380, "NH16 Near Yendada"),
    (17.8200, 83.3240, "NH16 Near Hanumanthawaka"),

    # City core & junctions
    (17.7609, 83.3188, "Hanumanthawaka Junction"),
    (17.7341, 83.3181, "Maddilapalem Junction"),
    (17.7260, 83.3030, "Dwaraka Nagar Junction"),
    (17.7350, 83.3050, "Railway Station Road"),
    (17.7130, 83.3200, "Siripuram Junction"),
    (17.7125, 83.3055, "Poorna Market Road"),

    # Beach Road (RK Beach – Rushikonda)
    (17.7130, 83.3410, "Beach Road RK Beach"),
    (17.7300, 83.3470, "Beach Road Lawson's Bay"),
    (17.7470, 83.3540, "Beach Road MVP"),
    (17.7969, 83.3842, "Beach Road Rushikonda"),

    # Gajuwaka / Kurmannapalem / Kancharapalem
    (17.7004, 83.2168, "Gajuwaka Junction"),
    (17.6899, 83.1693, "Kurmannapalem Road"),
    (17.7430, 83.2450, "Kancharapalem Road"),
    (17.6640, 83.2060, "Pedagantyada Road"),

    # Seethammadhara / MVP / Venkojipalem
    (17.7430, 83.3230, "Seethammadhara Road"),
    (17.7480, 83.3380, "MVP Colony Road"),
    (17.7500, 83.3270, "Venkojipalem Road"),

    # Outer/approach roads
    (17.8220, 83.1030, "Sabbavaram Road"),
]

AP_PLACES = VIZAG_ROAD_POINTS

# No jitter: keep points exactly on these roads
LAT_JITTER_DEG = 0.0
LON_JITTER_DEG = 0.0


def find_label_for_image(img_path: Path) -> Path:
    split = img_path.parent.parent.name  # "train" or "valid"
    labels_dir = DATA_ROOT / split / "labels"
    return labels_dir / (img_path.stem + ".txt")


def get_severity_from_label(label_path: Path) -> str:
    # For demo, ignore label and assign severity randomly.
    # If you later want real severities, replace this with class->severity logic.
    return random_severity()


def collect_images_with_severity(root: Path):
    records = []
    for split in ["train", "valid"]:
        img_dir = root / split / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*.*")):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            label_path = find_label_for_image(p)
            severity = get_severity_from_label(label_path)
            rel = p.relative_to(root)
            records.append((str(rel).replace("\\", "/"), severity))
    return records


def assign_locations_to_images(image_records):
    items = image_records[:]
    random.shuffle(items)
    n_places = len(AP_PLACES)
    assignments = []
    for idx, (img_rel, severity) in enumerate(items):
        plat, plon, pname = AP_PLACES[idx % n_places]
        lat = plat  # no jitter
        lon = plon
        assignments.append((img_rel, pname, lat, lon, severity))
    return assignments


def write_csv(assignments, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "place_name", "lat", "lon", "severity"])
        for img, pname, lat, lon, sev in assignments:
            w.writerow([img, pname, f"{lat:.6f}", f"{lon:.6f}", sev])


def main():
    records = collect_images_with_severity(DATA_ROOT)
    if not records:
        print("No images found under", DATA_ROOT)
        return
    print(f"Found {len(records)} images.")
    assignments = assign_locations_to_images(records)
    write_csv(assignments, OUTPUT_CSV)
    print(f"Wrote {len(assignments)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
