import cv2
import numpy as np


class ImageLoader:

    @staticmethod
    def load(image_path):
        """
        Load image from a file path.
        """

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        return image

    @staticmethod
    def load_uploaded(uploaded_file):
        """
        Load image uploaded through Streamlit.
        """

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid uploaded image.")

        return image