import cv2
import numpy as np

class Normalizer:

    def normalize(self, image):
        # Convert to float and scale to [0,1]
        img = image.astype(np.float32) / 255.0

        # Optional: back to uint8 for YOLO compatibility
        img = (img * 255).astype(np.uint8)

        return img