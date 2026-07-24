import json
from pathlib import Path
from PIL import Image

# -----------------------------
# Class mapping
# -----------------------------
CLASS_MAP = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motor": 6,
    "bike": 7,
    "traffic light": 8,
    "traffic sign": 9,
}

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "datasets" / "BDD100K"

ANNOTATION_FILE = (
    DATASET_ROOT
    / "annotations"
    / "bdd100k_labels_images_val.json"
)

IMAGES_DIR = DATASET_ROOT / "images" / "all"
LABELS_DIR = DATASET_ROOT / "labels" / "all"

LABELS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load annotations
# -----------------------------
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

skipped_categories = set()
processed = 0

# -----------------------------
# Convert annotations
# -----------------------------
for item in data:

    image_name = item.get("name")
    if image_name is None:
        continue

    image_path = IMAGES_DIR / image_name

    if not image_path.exists():
        continue

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    lines = []

    for label in item.get("labels", []):

        category = label.get("category")
        box = label.get("box2d")

        if category not in CLASS_MAP or box is None:
            if category:
                skipped_categories.add(category)
            continue

        class_id = CLASS_MAP[category]

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h

        lines.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_file = LABELS_DIR / f"{Path(image_name).stem}.txt"

    with open(label_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    processed += 1

print(f"Processed images : {processed}")
print(f"Skipped categories: {sorted(skipped_categories)}")