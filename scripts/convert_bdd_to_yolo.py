import json
import os

# ----------------------------
# Paths
# ----------------------------
JSON_PATH = "datasets/BDD100K/labels/bdd100k_labels_images_val.json"
OUTPUT_DIR = "datasets/BDD100K/yolo_labels/val"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# BDD100K classes to YOLO IDs
# ----------------------------
CLASS_MAP = {
    "person": 0,
    "rider": 0,
    "car": 2,
    "bus": 5,
    "truck": 7,
    "bike": 1,
    "motor": 3,
    "traffic light": 9,
    "train": 6,
}

# ----------------------------
# Load JSON
# ----------------------------
with open(JSON_PATH, "r") as f:
    data = json.load(f)

count = 0

for item in data:
    image_name = item["name"]
    labels = item.get("labels", [])

    txt_name = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_name)

    lines = []

    for obj in labels:

        category = obj.get("category")

        if category not in CLASS_MAP:
            continue

        if "box2d" not in obj:
            continue

        box = obj["box2d"]

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        # BDD100K image size
        img_w = 1280
        img_h = 720

        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        class_id = CLASS_MAP[category]

        lines.append(
            f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        )

    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    count += 1

print(f"Done! Converted {count} images.")