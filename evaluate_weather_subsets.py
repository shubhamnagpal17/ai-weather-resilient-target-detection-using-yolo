from pathlib import Path

import pandas as pd
from ultralytics import YOLO


MODEL_PATH = Path("models/best_pipelined.pt")
CONFIG_DIR = Path("configs") / "weather_subsets"
OUTPUT_DIR = Path("weather_results")

IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 0

EXPERIMENTS = {
    "night_raw": CONFIG_DIR / "night_raw.yaml",
    "night_pipeline": CONFIG_DIR / "night_pipeline.yaml",
    "rain_raw": CONFIG_DIR / "rain_raw.yaml",
    "rain_pipeline": CONFIG_DIR / "rain_pipeline.yaml",
    "snow_raw": CONFIG_DIR / "snow_raw.yaml",
    "snow_pipeline": CONFIG_DIR / "snow_pipeline.yaml",
}

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

for yaml_path in EXPERIMENTS.values():
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))
results = []

for experiment_name, yaml_path in EXPERIMENTS.items():
    print("\n" + "=" * 70)
    print(f"Evaluating: {experiment_name}")
    print(f"YAML      : {yaml_path}")
    print("=" * 70)

    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        project=str(OUTPUT_DIR),
        name=experiment_name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    speed = getattr(metrics, "speed", {}) or {}
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = 1000 / total_ms if total_ms > 0 else 0.0

    condition, input_type = experiment_name.rsplit("_", 1)

    results.append({
        "Condition": condition.capitalize(),
        "Input": input_type.capitalize(),
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "Preprocess_ms": preprocess_ms,
        "Inference_ms": inference_ms,
        "Postprocess_ms": postprocess_ms,
        "Total_ms_per_image": total_ms,
        "FPS": fps,
    })

results_df = pd.DataFrame(results)

numeric_columns = [
    "Precision",
    "Recall",
    "F1-Score",
    "mAP50",
    "mAP50-95",
    "Preprocess_ms",
    "Inference_ms",
    "Postprocess_ms",
    "Total_ms_per_image",
    "FPS",
]

results_df[numeric_columns] = results_df[numeric_columns].round(4)

detailed_csv = OUTPUT_DIR / "weather_subset_results.csv"
results_df.to_csv(detailed_csv, index=False)

comparison_rows = []

for condition in ["Night", "Rain", "Snow"]:
    subset = results_df[results_df["Condition"] == condition]

    raw_row = subset[subset["Input"] == "Raw"]
    pipeline_row = subset[subset["Input"] == "Pipeline"]

    if raw_row.empty or pipeline_row.empty:
        continue

    raw = raw_row.iloc[0]
    pipeline = pipeline_row.iloc[0]

    comparison_rows.append({
        "Condition": condition,
        "Raw_Precision": raw["Precision"],
        "Pipeline_Precision": pipeline["Precision"],
        "Raw_Recall": raw["Recall"],
        "Pipeline_Recall": pipeline["Recall"],
        "Raw_F1": raw["F1-Score"],
        "Pipeline_F1": pipeline["F1-Score"],
        "Raw_mAP50": raw["mAP50"],
        "Pipeline_mAP50": pipeline["mAP50"],
        "Absolute_mAP50_Gain": pipeline["mAP50"] - raw["mAP50"],
        "Relative_mAP50_Gain_Percent": (
            (pipeline["mAP50"] - raw["mAP50"]) / raw["mAP50"] * 100
            if raw["mAP50"] > 0
            else 0.0
        ),
        "Raw_mAP50-95": raw["mAP50-95"],
        "Pipeline_mAP50-95": pipeline["mAP50-95"],
        "Absolute_mAP50-95_Gain": pipeline["mAP50-95"] - raw["mAP50-95"],
        "Raw_FPS": raw["FPS"],
        "Pipeline_FPS": pipeline["FPS"],
    })

comparison_df = pd.DataFrame(comparison_rows).round(4)
comparison_csv = OUTPUT_DIR / "weather_subset_comparison.csv"
comparison_df.to_csv(comparison_csv, index=False)

excel_path = OUTPUT_DIR / "evaluation/weather_subset_results.xlsx"

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Detailed Results", index=False)
    comparison_df.to_excel(writer, sheet_name="Raw vs Pipeline", index=False)

print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))

print("\n" + "=" * 70)
print("RAW VS PIPELINE COMPARISON")
print("=" * 70)
print(comparison_df.to_string(index=False))

print(f"\nDetailed CSV : {detailed_csv}")
print(f"Comparison   : {comparison_csv}")
print(f"Excel file  : {excel_path}")