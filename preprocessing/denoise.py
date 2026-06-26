import cv2

class Denoiser:
    def __init__(self, h=10):
        self.h = h

    def remove_noise(self, image):
        # Fast Non-Local Means Denoising
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            self.h,
            self.h,
            7,
            21
        )