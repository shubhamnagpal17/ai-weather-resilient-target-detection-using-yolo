from pathlib import Path
import cv2

from preprocessing.denoise import Denoiser
from preprocessing.adaptive_params_calc import AdaptiveParameterCalculator


INPUT_DIR = Path("datasets/BDD100K/images/val")
OUTPUT_DIR = Path("datasets/BDD100K/images/val_denoise")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

denoiser = Denoiser()
adaptive_calculator = AdaptiveParameterCalculator()

valid_extensions = {".jpg", ".jpeg", ".png"}

processed = 0
denoise_applied = 0
denoise_skipped = 0
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

    if params["apply_denoise"]:
        denoiser.h = params["denoise_h"]
        output_image = denoiser.remove_noise(image)
        denoise_applied += 1
    else:
        output_image = image
        denoise_skipped += 1

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

print("\nAdaptive Denoise dataset completed")
print(f"Total images saved: {processed}")
print(f"Denoise applied: {denoise_applied}")
print(f"Denoise skipped: {denoise_skipped}")
print(f"Failed: {failed}")
print(f"Output folder: {OUTPUT_DIR}")