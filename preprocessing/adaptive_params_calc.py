import cv2
import numpy as np


class AdaptiveParameterCalculator:

    def calculate_params(self, image):
        gray = self._to_grayscale(image)
        config = {}
        config.update(self._calculate_clahe_params(gray))
        config.update(self._calculate_gamma_params(gray))
        config.update(self._calculate_denoise_params(gray))
        return config

    def _to_grayscale(self, image):
        """Convert BGR image to grayscale when required."""
        if image is None:
            raise ValueError("Input image cannot be None.")
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _calculate_clahe_params(self, gray):
        """Calculate adaptive CLAHE settings."""
        contrast = np.std(gray)
        clamped_contrast = np.clip(contrast, 20, 150)
        clip_limit = 4.0 - ((clamped_contrast - 20) / (150 - 20)) * (4.0 - 1.2)
        apply_clahe = contrast <= 150

        return {
            "apply_clahe": apply_clahe,
            "clip_limit": (
                round(float(clip_limit), 2)
                if apply_clahe
                else 1.2
            )
        }

    def _calculate_gamma_params(self, gray):
        """Calculate adaptive gamma settings."""
        mean_brightness = np.mean(gray)
        clamped_brightness = np.clip(mean_brightness,30,200)
        gamma_value = 0.5 + ((clamped_brightness - 30) / (200 - 30)) * (1.5 - 0.5)
        if mean_brightness < 15:
            gamma_value = 0.4
        elif mean_brightness > 230:
            gamma_value = 1.6

        return {
            "apply_gamma": True,
            "gamma_value": round(
                float(gamma_value),
                2
            )
        }

    def _calculate_denoise_params(self, gray):
        """Calculate adaptive denoising settings."""
        smooth = cv2.GaussianBlur(gray,(3, 3),0)
        noise = (gray.astype(np.float32)- smooth.astype(np.float32))
        noise_std = np.std(noise)
        if noise_std < 10:
            return {
                "apply_denoise": False,
                "denoise_h": 0.0
            }
        clamped_noise_std = np.clip(noise_std,10,30)
        denoise_h = 10.0 + ((clamped_noise_std - 10.0)/ (30.0 - 10.0)) * (20.0 - 10.0)

        return {
            "apply_denoise": True,
            "denoise_h": round(
                float(denoise_h),
                2
            )
        }