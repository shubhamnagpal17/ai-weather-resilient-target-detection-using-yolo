import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "BDD100K"
ANNOTATION_FILE = DATASET_ROOT / "annotations" / "bdd100k_labels_images_val.json"
IMAGES_DIR = DATASET_ROOT / "images" / "all"
SPLITS_DIR = DATASET_ROOT / "splits"


def main() -> None:
    random.seed(SEED)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    with ANNOTATION_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)

    for item in data:
        image_name = item.get("name")
        attributes = item.get("attributes", {})
        weather = attributes.get("weather", "unknown")
        time_of_day = attributes.get("timeofday", "unknown")

        if image_name and (IMAGES_DIR / image_name).exists():
            groups[(weather, time_of_day)].append(image_name)

    train_names: list[str] = []
    val_names: list[str] = []
    test_names: list[str] = []

    for image_names in groups.values():
        random.shuffle(image_names)

        total = len(image_names)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_names.extend(image_names[:train_end])
        val_names.extend(image_names[train_end:val_end])
        test_names.extend(image_names[val_end:])

    random.shuffle(train_names)
    random.shuffle(val_names)
    random.shuffle(test_names)

    split_data = {
        "train.txt": train_names,
        "val.txt": val_names,
        "test.txt": test_names,
    }

    for filename, image_names in split_data.items():
        lines = [f"datasets/BDD100K/images/all/{name}" for name in image_names]
        (SPLITS_DIR / filename).write_text("\n".join(lines), encoding="utf-8")

    all_names = set(train_names) | set(val_names) | set(test_names)

    assert not (set(train_names) & set(val_names))
    assert not (set(train_names) & set(test_names))
    assert not (set(val_names) & set(test_names))

    print(f"Train images: {len(train_names)}")
    print(f"Validation images: {len(val_names)}")
    print(f"Test images: {len(test_names)}")
    print(f"Total unique images: {len(all_names)}")


if __name__ == "__main__":
    main()
