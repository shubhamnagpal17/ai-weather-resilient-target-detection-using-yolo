from ultralytics import YOLO
import cv2

class yolomodel:
    def __init__(self, model_path="yolov8n.pt", conf=0.6):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf = conf

    @property
    def model_name(self):
        return self.model_path

    # ----------------------------
    # Run inference
    # ----------------------------
    def predict(self, image):
        """
        image: file path OR numpy array
        returns: raw YOLO results
        """
        results = self.model(image, conf=self.conf)
        return results
    
    # ----------------------------
    # Parse results into clean format
    # ----------------------------
    def parse(self, results):
        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = self.model.names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]})
        return detections
    
    # ----------------------------
    # Draw detections on image
    # ----------------------------
    def draw(self, results):
        return results[0].plot()
    
    # ----------------------------
    # Full detection pipeline
    # ----------------------------
    def detect(self, image):
        results = self.predict(image)
        detections = self.parse(results)
        annotated = self.draw(results)

        return detections, annotated