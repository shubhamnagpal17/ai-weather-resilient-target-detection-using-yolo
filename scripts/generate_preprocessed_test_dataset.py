import cv2
from pathlib import Path
import shutil

from preprocessing.clahe import ImageEnhancer
from preprocessing.weather_enhancer import WeatherEnhancer
from preprocessing.denoise import Denoiser
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.adaptive_params_calc import AdaptiveParameterCalculator


ROOT = Path(".")

test_split_file = ROOT / "datasets/BDD100K/splits/test.txt"
output_base = ROOT / "datasets/BDD100K/images"
labels_root = ROOT / "datasets/BDD100K/labels/all"
labels_output_root = ROOT / "datasets/BDD100K/labels"

adaptive_params = AdaptiveParameterCalculator()
clahe = ImageEnhancer()
gamma = WeatherEnhancer()
denoiser = Denoiser()
pipeline = PreprocessingPipeline()


def apply_adaptive_clahe(image):
    params = adaptive_params.calculate_params(image)

    if not params["apply_clahe"]:
        return image.copy()

    clahe.set_clip_limit(params["clip_limit"])
    return clahe.apply_clahe(image)


def apply_adaptive_gamma(image):
    params = adaptive_params.calculate_params(image)

    if not params["apply_gamma"]:
        return image.copy()

    return gamma.improve_visibility(
        image,
        gamma=params["gamma_value"]
    )


def apply_adaptive_denoise(image):
    params = adaptive_params.calculate_params(image)

    if not params["apply_denoise"]:
        return image.copy()

    denoiser = Denoiser(h=params["denoise_h"])
    return denoiser.remove_noise(image)


def apply_full_pipeline(image):
    preprocessed, _, _ = pipeline.process(image)
    return preprocessed


variants = {
    "test_clahe": apply_adaptive_clahe,
    "test_gamma": apply_adaptive_gamma,
    "test_denoise": apply_adaptive_denoise,
    "test_full_pipeline": apply_full_pipeline,
}


with open(test_split_file, "r", encoding="utf-8") as file:
    test_image_paths = [
        line.strip()
        for line in file
        if line.strip()
    ]

print(f"Total test images: {len(test_image_paths)}")


for variant_name, process_function in variants.items():
    output_dir = output_base / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    output_dir = output_base / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    label_output_dir = labels_output_root / variant_name
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for relative_path in test_image_paths:
        image_path = ROOT / relative_path
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path}")
            skipped_count += 1
            continue

        try:
            processed_image = process_function(image)

            output_path = output_dir / image_path.name
            saved = cv2.imwrite(
                str(output_path),
                processed_image
            )

            if saved:
                processed_count += 1
                    # Copy corresponding label
                label_name = image_path.with_suffix(".txt").name
                source_label = labels_root / label_name
                destination_label = label_output_dir / label_name

                if source_label.exists():
                    shutil.copy2(source_label, destination_label)
                else:
                    print(f"Label not found: {source_label}")
            else:
                print(f"Could not save: {output_path}")
                skipped_count += 1

        except Exception as error:
            print(f"Error processing {image_path.name}: {error}")
            skipped_count += 1

    print(
        f"{variant_name}: "
        f"{processed_count} processed, "
        f"{skipped_count} skipped"
    )