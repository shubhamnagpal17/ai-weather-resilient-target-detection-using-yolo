from pathlib import Path
import cv2

from preprocessing.clahe import ImageEnhancer
from preprocessing.adaptive_params_calc import AdaptiveParameterCalculator


INPUT_DIR = Path("datasets/BDD100K/images/val")
OUTPUT_DIR = Path("datasets/BDD100K/images/val_clahe")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder: {INPUT_DIR.resolve()}")
print(f"Input exists: {INPUT_DIR.exists()}")

enhancer = ImageEnhancer()
adaptive_calculator = AdaptiveParameterCalculator()

valid_extensions = {".jpg", ".jpeg", ".png"}

processed = 0
clahe_applied = 0
clahe_skipped = 0
failed = 0

for image_path in INPUT_DIR.rglob("*"):

    if image_path.suffix.lower() not in valid_extensions:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        failed += 1
        continue

    # Calculate adaptive parameters for this image
    params = adaptive_calculator.calculate_params(image)

    if params["apply_clahe"]:
        enhancer.set_clip_limit(params["clip_limit"])
        output_image = enhancer.apply_clahe(image)
        clahe_applied += 1
    else:
        # CLAHE skipped, original image is saved
        output_image = image
        clahe_skipped += 1

    relative_path = image_path.relative_to(INPUT_DIR)
    output_path = OUTPUT_DIR / relative_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_path), output_image)

    if not success:
        print(f"Could not save: {output_path}")
        failed += 1
        continue

    processed += 1

    if processed % 500 == 0:
        print(f"Processed: {processed}")

print("\nDataset generation completed")
print(f"Total images saved: {processed}")
print(f"CLAHE applied: {clahe_applied}")
print(f"CLAHE skipped: {clahe_skipped}")
print(f"Failed: {failed}")
print(f"Output folder: {OUTPUT_DIR}")