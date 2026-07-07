import base64
import json
import time
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request

from input.image_loader import ImageLoader
from model.yolo import yolomodel
from preprocessing.letterbox import Letterbox
from preprocessing.pipeline import PreprocessingPipeline

app = Flask(__name__)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

pipeline = PreprocessingPipeline(apply_gaussian=False)
detector = yolomodel(model_path="yolov8n.pt", conf=0.25)


# ── Helpers ───────────────────────────────────────────────────────
def encode_image_b64(image):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode("utf-8")


def draw_detections(image, detections):
    """Draw red bounding boxes and labels on *image*."""
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{det['label']} {det['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated, (x1, y1 - th - baseline - 4), (x1 + tw, y1), (0, 0, 255), -1
        )
        cv2.putText(
            annotated, label, (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def save_outputs(original, preprocessed, annotated, detections, metrics):
    cv2.imwrite(str(OUTPUT_DIR / "original.jpg"), original)
    cv2.imwrite(str(OUTPUT_DIR / "preprocessed.jpg"), preprocessed)
    cv2.imwrite(str(OUTPUT_DIR / "detected.jpg"), annotated)
    payload = {"metrics": metrics, "detections": detections}
    with open(OUTPUT_DIR / "detections.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_image_from_request():
    """Accept multipart file upload or JSON base64 payload."""
    if "image" in request.files and request.files["image"].filename:
        return ImageLoader.load_uploaded(request.files["image"])
    data = request.get_json(silent=True)
    if data and "image" in data:
        return ImageLoader.load_base64(data["image"])
    raise ValueError("No image provided. Upload a file or send base64 JSON.")


# ── Routes ────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html", active_page="home")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")


@app.route("/detect", methods=["POST"])
def detect():
    try:
        original = load_image_from_request()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    h, w = original.shape[:2]
    preprocessed, letterboxed, meta = pipeline.process(original)
    t0 = time.perf_counter()
    results = detector.predict(letterboxed)
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)
    raw_detections = detector.parse(results)
    mapped_boxes = Letterbox.map_boxes_to_original(
        [d["bbox"] for d in raw_detections], meta
    )
    detections = []
    for raw, bbox in zip(raw_detections, mapped_boxes):
        detections.append({
            "label": raw["label"],
            "confidence": round(raw["confidence"], 4),
            "bbox": bbox,
        })
    annotated = draw_detections(preprocessed, detections)
    metrics = {
        "detection_count": len(detections),
        "image_size": f"{w}x{h}",
        "inference_time_ms": inference_ms,
        "model": detector.model_name,
    }
    save_outputs(original, preprocessed, annotated, detections, metrics)
    return jsonify({
        "status": "done",
        "metrics": metrics,
        "images": {
            "original": encode_image_b64(original),
            "preprocessed": encode_image_b64(preprocessed),
            "annotated": encode_image_b64(annotated),
        },
        "detections": detections,
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
