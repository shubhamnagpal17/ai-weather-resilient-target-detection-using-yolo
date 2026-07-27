from pathlib import Path

import pandas as pd
from ultralytics import YOLO


# ======================================================
# Configuration
# ======================================================

MODEL_PATH = "models/best.pt"

PROJECT = "test_results"

VARIANTS = {
    "yolo_only": "configs/bdd_raw_split.yaml",
    "clahe": "configs/data_test_clahe.yaml",
    "gamma": "configs/data_test_gamma.yaml",
    "denoise": "configs/data_test_denoise.yaml",
    "clahe_gamma": "configs/data_test_clahe_gamma.yaml",
    "clahe_denoise": "configs/data_test_clahe_denoise.yaml",
    "gamma_denoise": "configs/data_test_gamma_denoise.yaml",
    "full_pipeline": "configs/data_test_full_pipeline.yaml",
}


# ======================================================
# Load model
# ======================================================

print("=" * 70)
print("Loading model...")
print("=" * 70)

model = YOLO(MODEL_PATH)

results_summary = []


# ======================================================
# Evaluate every dataset
# ======================================================

for name, yaml_path in VARIANTS.items():

    print("\n" + "=" * 70)
    print(f"Evaluating: {name}")
    print("=" * 70)

    metrics = model.val(
        data=yaml_path,
        split="test",
        imgsz=640,
        batch=8,
        workers=0,
        project=PROJECT,
        name=name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    results_summary.append(
        {
            "Variant": name,
            "Precision": precision,
            "Recall": recall,
            "mAP50": map50,
            "mAP50-95": map5095,
        }
    )


# ======================================================
# Print Summary
# ======================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for row in results_summary:
    print(
        f"{row['Variant']:18s}"
        f"P={row['Precision']:.3f}  "
        f"R={row['Recall']:.3f}  "
        f"mAP50={row['mAP50']:.3f}  "
        f"mAP50-95={row['mAP50-95']:.3f}"
    )


# ======================================================
# Save CSV
# ======================================================

df = pd.DataFrame(results_summary)

csv_path = Path(PROJECT) / "evaluation_summary.csv"

df.to_csv(csv_path, index=False)

print("\nSaved:", csv_path)