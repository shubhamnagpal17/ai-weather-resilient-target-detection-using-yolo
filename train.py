from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent

    model = YOLO(project_root / "yolov8n.pt")

    model.train(
        data=project_root / "configs" / "bdd_raw_split.yaml",
        epochs=30,
        imgsz=512,
        batch=8,
        workers=0,
        device="cpu",
        project=project_root / "training_runs",
        name="raw_baseline",
    )


if __name__ == "__main__":
    main()