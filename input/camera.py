import cv2
class Camera:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

    def capture(self):
        """
        Capture a single image.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image.")
        return frame

    def release(self):
        self.cap.release()