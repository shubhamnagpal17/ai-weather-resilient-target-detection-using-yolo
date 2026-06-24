from ultralytics import YOLO
import cv2
import os

# ----------------------------
# Load image
# ----------------------------

image_path = "image.png"

img = cv2.imread(image_path)

if img is None:
    print("Error: Could not load image.")
    exit()

print(f"Image Shape: {img.shape}")

# ----------------------------
# Load YOLO model
# ----------------------------

model = YOLO("yolov8n.pt")

# ----------------------------
# Run detection
# ----------------------------

results = model(
    image_path,
    conf=0.6
)

# ----------------------------
# Print detections
# ----------------------------

print("\nDetected Objects:")

for box in results[0].boxes:

    class_id = int(box.cls[0])

    confidence = float(box.conf[0])

    label = model.names[class_id]

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    print(
        f"{label} | {confidence:.2f} | "
        f"({x1}, {y1}, {x2}, {y2})"
    )

# ----------------------------
# Save output image
# ----------------------------

annotated_image = results[0].plot()

output_path = "output.jpg"

cv2.imwrite(output_path, annotated_image)

print(f"\nSaved output image: {output_path}")

print(f"Absolute Path: {os.path.abspath(output_path)}")