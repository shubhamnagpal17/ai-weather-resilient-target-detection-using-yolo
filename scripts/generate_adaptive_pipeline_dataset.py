from pathlib import Path
import cv2

from preprocessing.pipeline import PreprocessingPipeline


INPUT_DIR = Path("dataset/images/val")
OUTPUT_DIR = Path("dataset/images/val_adaptive_pipeline")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pipeline = PreprocessingPipeline()

processed = 0
failed = 0

for image_path in INPUT_DIR.rglob("*"):

    if image_path.suffix.lower() not in VALID_EXTENSIONS:
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        failed += 1
        continue

    try:
        # Uses the complete existing adaptive pipeline:
        # Adaptive CLAHE -> Adaptive Gamma -> Adaptive Denoising
        preprocessed, _, _ = pipeline.process(image)

        relative_path = image_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(output_path), preprocessed):
            print(f"Could not save: {output_path}")
            failed += 1
            continue

        processed += 1

        if processed % 500 == 0:
            print(f"Processed: {processed}")

    except Exception as error:
        print(f"Failed: {image_path}")
        print(f"Reason: {error}")
        failed += 1


print("\nAdaptive pipeline dataset completed")
print(f"Total images saved: {processed}")
print(f"Failed images: {failed}")
print(f"Output folder: {OUTPUT_DIR}")