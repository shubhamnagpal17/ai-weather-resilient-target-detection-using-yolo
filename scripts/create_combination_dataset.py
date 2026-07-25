from pathlib import Path

import cv2

from preprocessing.adaptive_params_calc import AdaptiveParameterCalculator
from preprocessing.weather_enhancer import WeatherEnhancer
from preprocessing.denoise import Denoiser


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

ROOT = Path("datasets/BDD100K")

CLAHE_DIR = ROOT / "images" / "test_clahe"
GAMMA_DIR = ROOT / "images" / "test_gamma"

CLAHE_GAMMA_DIR = ROOT / "images" / "test_clahe_gamma"
CLAHE_DENOISE_DIR = ROOT / "images" / "test_clahe_denoise"
GAMMA_DENOISE_DIR = ROOT / "images" / "test_gamma_denoise"


# -------------------------------------------------------
# Create output directories
# -------------------------------------------------------

for folder in [
    CLAHE_GAMMA_DIR,
    CLAHE_DENOISE_DIR,
    GAMMA_DENOISE_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Preprocessing modules
# -------------------------------------------------------

adaptive = AdaptiveParameterCalculator()
weather = WeatherEnhancer()
denoiser = Denoiser()


def apply_gamma(image):
    """Apply adaptive gamma correction."""
    params = adaptive.calculate_params(image)

    gamma = params["gamma_value"]

    return weather.improve_visibility(
        image,
        gamma=gamma
    )


def apply_denoise(image):
    """Apply adaptive denoising."""
    params = adaptive.calculate_params(image)

    # If adaptive logic says denoising is unnecessary,
    # return the image unchanged.
    if not params["apply_denoise"]:
        return image.copy()

    denoiser.h = params["denoise_h"]

    return denoiser.remove_noise(image)


def process_folder(source_dir, output_dir, operation):
    images = list(source_dir.glob("*.jpg"))

    print()
    print("=" * 70)
    print(f"Source : {source_dir}")
    print(f"Output : {output_dir}")
    print(f"Images : {len(images)}")
    print("=" * 70)

    processed = 0
    failed = 0

    for i, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Could not read: {image_path}")
            failed += 1
            continue

        result = operation(image)

        output_path = output_dir / image_path.name

        success = cv2.imwrite(str(output_path), result)

        if success:
            processed += 1
        else:
            print(f"[WARNING] Could not save: {output_path}")
            failed += 1

        if i % 100 == 0 or i == len(images):
            print(f"{i}/{len(images)} processed")

    print()
    print(f"Successfully processed : {processed}")
    print(f"Failed                 : {failed}")


# =======================================================
# 1. CLAHE + Gamma
# =======================================================

process_folder(
    source_dir=CLAHE_DIR,
    output_dir=CLAHE_GAMMA_DIR,
    operation=apply_gamma,
)


# =======================================================
# 2. CLAHE + Denoise
# =======================================================

process_folder(
    source_dir=CLAHE_DIR,
    output_dir=CLAHE_DENOISE_DIR,
    operation=apply_denoise,
)


# =======================================================
# 3. Gamma + Denoise
# =======================================================

process_folder(
    source_dir=GAMMA_DIR,
    output_dir=GAMMA_DENOISE_DIR,
    operation=apply_denoise,
)


print()
print("=" * 70)
print("ALL COMBINED TEST DATASETS GENERATED")
print("=" * 70)