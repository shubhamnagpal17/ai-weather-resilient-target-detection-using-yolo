from preprocessing.clahe import ImageEnhancer
from preprocessing.denoise import Denoiser
from preprocessing.normalize import Normalizer
from preprocessing.weather_enhancer import WeatherEnhancer

class PreprocessingPipeline:
    def __init__(self):
        self.enhancer=ImageEnhancer()
        self.denoiser=Denoiser()
        self.normalizer=Normalizer()
        self.weather=WeatherEnhancer()

    def process(self, image):
        
        # 1. Remove noise
        image = self.denoiser.remove_noise(image)

        # 2. Improve visibility
        image = self.weather.improve_visibility(image)

        # 3. Enhance contrast
        image = self.enhancer.apply_clahe(image)

        # 4. Normalize
        image = self.normalizer.normalize(image)

        return image