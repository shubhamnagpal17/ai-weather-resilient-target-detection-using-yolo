import cv2

from preprocessing.clahe import ImageEnhancer
from preprocessing.letterbox import Letterbox
from preprocessing.normalize import Normalizer
from preprocessing.weather_enhancer import WeatherEnhancer
from preprocessing.denoise import Denoiser
from preprocessing.adaptive_params_calc import AdaptiveParameterCalculator


class PreprocessingPipeline:
    def __init__(self):
        self.adaptiveparams= AdaptiveParameterCalculator()
        self.enhancer = ImageEnhancer()
        self.weather = WeatherEnhancer()
        self.denoiser= Denoiser()
        self.normalizer = Normalizer()
        self.letterboxer = Letterbox(size=640)

    def process(self, image):
        """
        Returns:
            preprocessed  — enhanced image at original resolution (for display)
            letterboxed   — 640×640 YOLO input
            meta          — letterbox padding/ratio for bbox remapping
        """
        #calculating parameters
        config_params=self.adaptiveparams.calculate_params(image)

        # 1. CLAHE step
        if config_params["apply_clahe"]:
            self.enhancer.set_clip_limit(config_params["clip_limit"])
            image = self.enhancer.apply_clahe(image)

        # 2. Gamma Correction / Visibility Improve step
        if config_params["apply_gamma"]:
            image = self.weather.improve_visibility(image,gamma=config_params["gamma_value"])

        # 3. Denoising step
        if config_params["apply_denoise"]:
            self.denoiser.h=config_params["denoise_h"]
            image = self.denoiser.remove_noise(image)

        # Enhanced image for dashboard display
        preprocessed = image.copy()

        # 4. Letterbox resize
        letterboxed, meta = self.letterboxer.resize(image)

        # 5. Normalize final model input
        letterboxed = self.normalizer.normalize(letterboxed)

        return preprocessed, letterboxed, meta
