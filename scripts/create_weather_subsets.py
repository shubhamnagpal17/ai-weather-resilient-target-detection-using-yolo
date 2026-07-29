import json
from pathlib import Path

ROOT = Path("datasets/BDD100K")

ANNOTATION_JSON = ROOT / "annotations" / "bdd100k_labels_images_val.json"
TRAIN_SPLIT = ROOT / "splits" / "train.txt"
VAL_SPLIT = ROOT / "splits" / "val.txt"
TEST_SPLIT = ROOT / "splits" / "test.txt"

RAW_IMAGES_DIR = ROOT / "images" / "all"
PIPELINE_IMAGES_DIR = ROOT / "images" / "test_full_pipeline"

OUTPUT_SPLIT_DIR = ROOT / "weather_subsets"
OUTPUT_CONFIG_DIR = Path("configs") / "weather_subsets"

OUTPUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

required_paths = [
    ANNOTATION_JSON,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RAW_IMAGES_DIR,
    PIPELINE_IMAGES_DIR,
]

for path in required_paths:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")

test_image_names = {
    Path(line.strip()).name
    for line in TEST_SPLIT.read_text(encoding="utf-8").splitlines()
    if line.strip()
}

print(f"Images in held-out test split: {len(test_image_names)}")

with ANNOTATION_JSON.open("r", encoding="utf-8") as file:
    annotations = json.load(file)

subsets = {
    "night": [],
    "rain": [],
    "snow": [],
}

for item in annotations:
    image_name = item.get("name")

    if not image_name or image_name not in test_image_names:
        continue

    attributes = item.get("attributes", {})
    weather = str(attributes.get("weather", "unknown")).strip().lower()
    time_of_day = str(attributes.get("timeofday", "unknown")).strip().lower()

    if time_of_day == "night":
        subsets["night"].append(image_name)

    if weather in {"rainy", "rain"}:
        subsets["rain"].append(image_name)

    if weather in {"snowy", "snow"}:
        subsets["snow"].append(image_name)


def create_split(
    subset_name: str,
    image_names: list[str],
    image_directory: Path,
    variant_name: str,
) -> Path:
    valid_paths = []
    missing_paths = []

    for image_name in sorted(set(image_names)):
        image_path = image_directory / image_name

        if image_path.exists():
            valid_paths.append(image_path.as_posix())
        else:
            missing_paths.append(image_path.as_posix())

    output_file = OUTPUT_SPLIT_DIR / f"{subset_name}_{variant_name}.txt"
    output_file.write_text("\n".join(valid_paths), encoding="utf-8")

    print(
        f"{output_file.name}: "
        f"{len(valid_paths)} images, "
        f"{len(missing_paths)} missing"
    )

    if missing_paths:
        print("Missing examples:", missing_paths[:3])

    return output_file


def create_yaml(
    subset_name: str,
    variant_name: str,
    split_file: Path,
) -> Path:
    yaml_content = f"""path: .

train: {TRAIN_SPLIT.as_posix()}
val: {VAL_SPLIT.as_posix()}
test: {split_file.as_posix()}

names:
  0: person
  1: rider
  2: car
  3: truck
  4: train
  5: bike
"""

    yaml_file = OUTPUT_CONFIG_DIR / f"{subset_name}_{variant_name}.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    print(f"Created YAML: {yaml_file}")

    return yaml_file


created_yaml_files = []

for subset_name, image_names in subsets.items():
    unique_names = sorted(set(image_names))

    print("\n" + "=" * 70)
    print(f"{subset_name.upper()} SUBSET: {len(unique_names)} test images")
    print("=" * 70)

    raw_split = create_split(
        subset_name,
        unique_names,
        RAW_IMAGES_DIR,
        "raw",
    )

    pipeline_split = create_split(
        subset_name,
        unique_names,
        PIPELINE_IMAGES_DIR,
        "pipeline",
    )

    raw_yaml = create_yaml(
        subset_name,
        "raw",
        raw_split,
    )

    pipeline_yaml = create_yaml(
        subset_name,
        "pipeline",
        pipeline_split,
    )

    created_yaml_files.extend([raw_yaml, pipeline_yaml])

print("\n" + "=" * 70)
print("WEATHER SUBSET CREATION COMPLETE")
print("=" * 70)

for subset_name, image_names in subsets.items():
    print(f"{subset_name:8s}: {len(set(image_names))} images")

print(f"\nSplit files saved in: {OUTPUT_SPLIT_DIR}")
print(f"YAML files saved in : {OUTPUT_CONFIG_DIR}")

print("\nCreated YAML files:")
for yaml_file in created_yaml_files:
    print(yaml_file)