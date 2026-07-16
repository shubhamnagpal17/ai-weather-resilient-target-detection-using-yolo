import cv2


class ImageEnhancer:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid_size)

    # ----------------------------
    # Update CLAHE Clip Limit
    # ----------------------------
    def set_clip_limit(self, clip_limit):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=self.tile_grid_size)

    # ----------------------------
    # CLAHE Contrast Enhancement
    # ----------------------------
    def apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)