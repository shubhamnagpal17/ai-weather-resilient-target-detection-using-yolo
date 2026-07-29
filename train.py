from pathlib import Path
import shutil
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=project_root / "configs" / "bdd_raw_split.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        optimizer="AdamW",
        lr0=0.0005,
        patience=15,
        workers=0,
        device="cpu",
        pretrained=True,
        project=project_root / "training_runs",
        name="pipeline_final",
        exist_ok=True,
        save=True,
    )

    best_model = Path(results.save_dir) / "weights" / "best.pt"

    if best_model.exists():
        destination = models_dir / "best_pipelined.pt"
        shutil.copy2(best_model, destination)
        print(f"\nBest model saved to: {destination}")
    else:
        print("\nError: best.pt not found.")


if __name__ == "__main__":
    main()
