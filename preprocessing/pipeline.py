import cv2

from preprocessing.clahe import ImageEnhancer
from preprocessing.letterbox import Letterbox
from preprocessing.normalize import Normalizer
from preprocessing.weather_enhancer import WeatherEnhancer


class PreprocessingPipeline:
    """
    Default pipeline (per project spec):
        CLAHE → Gamma Correction → (optional) Gaussian Blur→ Normalization → Letterbox 640×640

    Denoiser is kept as an optional module (preprocessing/denoise.py)
    but is NOT part of this default chain.
    """

    def __init__(self, apply_gaussian=False):
        self.enhancer = ImageEnhancer()
        self.weather = WeatherEnhancer()
        self.normalizer = Normalizer()
        self.letterboxer = Letterbox(size=640)
        self.apply_gaussian = apply_gaussian

    def process(self, image):
        """
        Returns:
            preprocessed  — enhanced image at original resolution (for display)
            letterboxed   — 640×640 YOLO input
            meta          — letterbox padding/ratio for bbox remapping
        """
        image = self.enhancer.apply_clahe(image)
        image = self.weather.improve_visibility(image)

        if self.apply_gaussian:
            image = cv2.GaussianBlur(image, (5, 5), 0)

        image = self.normalizer.normalize(image)
        preprocessed = image.copy()

        letterboxed, meta = self.letterboxer.resize(image)
        return preprocessed, letterboxed, meta
