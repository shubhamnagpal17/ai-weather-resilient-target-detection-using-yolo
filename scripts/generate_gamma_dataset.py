from pathlib import Path
import cv2

from preprocessing.weather_enhancer import WeatherEnhancer
from preprocessing.adaptive_params_calc import AdaptiveParameterCalculator


INPUT_DIR = Path("dataset/images/val")
OUTPUT_DIR = Path("dataset/images/val_adaptive_gamma")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

weather_enhancer = WeatherEnhancer()
adaptive_calculator = AdaptiveParameterCalculator()

valid_extensions = {".jpg", ".jpeg", ".png"}

processed = 0
gamma_applied = 0
gamma_skipped = 0
failed = 0

for image_path in INPUT_DIR.rglob("*"):

    if image_path.suffix.lower() not in valid_extensions:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        failed += 1
        continue

    params = adaptive_calculator.calculate_params(image)

    if params["apply_gamma"]:
        output_image = weather_enhancer.improve_visibility(
            image,
            gamma=params["gamma_value"]
        )
        gamma_applied += 1
    else:
        output_image = image
        gamma_skipped += 1

    relative_path = image_path.relative_to(INPUT_DIR)
    output_path = OUTPUT_DIR / relative_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), output_image):
        print(f"Could not save: {output_path}")
        failed += 1
        continue

    processed += 1

    if processed % 500 == 0:
        print(f"Processed: {processed}")

print("\nAdaptive Gamma dataset completed")
print(f"Total images saved: {processed}")
print(f"Gamma applied: {gamma_applied}")
print(f"Gamma skipped: {gamma_skipped}")
print(f"Failed: {failed}")
print(f"Output folder: {OUTPUT_DIR}")