import cv2
import numpy as np

class WeatherEnhancer:

    def __init__(self):
        self.gamma = 1.5
    def improve_visibility(self, image, gamma=None):
        """
        Apply gamma correction.
        Uses the provided gamma value if available,
        otherwise falls back to the stored value.
        """

        if gamma is None:
            gamma = self.gamma
        inv_gamma = 1.0 / gamma

        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8
        )

        return cv2.LUT(image, table)