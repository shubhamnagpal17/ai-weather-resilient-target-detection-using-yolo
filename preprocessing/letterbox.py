import cv2
import numpy as np


class Letterbox:
    """Standard YOLO letterbox resize with coordinate mapping metadata."""

    def __init__(self, size=640, color=(114, 114, 114)):
        self.size = size
        self.color = color

    def resize(self, image):
        """
        Letterbox *image* to (size × size).
        Returns (letterboxed_image, meta) where meta holds ratio and padding
        for mapping detections back to the original resolution.
        """
        h, w = image.shape[:2]
        target = self.size

        ratio = min(target / h, target / w)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        if (w, h) != (new_w, new_h):
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_w = target - new_w
        pad_h = target - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        letterboxed = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.color,
        )

        meta = {"ratio": ratio, "pad": (left, top)}
        return letterboxed, meta

    @staticmethod
    def map_boxes_to_original(boxes, meta):
        """Convert bounding boxes from letterbox space back to original image coords."""
        ratio = meta["ratio"]
        left, top = meta["pad"]
        mapped = []

        for box in boxes:
            x1, y1, x2, y2 = box
            mapped.append([
                int((x1 - left) / ratio),
                int((y1 - top) / ratio),
                int((x2 - left) / ratio),
                int((y2 - top) / ratio),
            ])

        return mapped
