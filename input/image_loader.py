import base64
import cv2
import numpy as np


class ImageLoader:

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    @staticmethod
    def load(image_path):
        """Load image from a file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return image

    @staticmethod
    def load_uploaded(uploaded_file):
        """Decode an uploaded Flask file object (JPG / PNG / JPEG)."""
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid uploaded image.")
        return image

    @staticmethod
    def load_base64(b64_string):
        """Decode a base64-encoded image string (webcam fallback)."""
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        raw = base64.b64decode(b64_string)
        arr = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid base64 image.")
        return image
