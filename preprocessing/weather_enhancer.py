import cv2
import numpy as np

class WeatherEnhancer:
    def improve_visibility(self, image):
        # simple gamma correction (helps fog/low light)
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        table = cv2.LUT(image, np.array(table, dtype="uint8"))
        return table