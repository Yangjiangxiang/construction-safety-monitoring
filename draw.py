import cv2
import numpy as np


class ImageDrawer:
    """Draw a relative restricted area and flag detections entering it."""

    # Original polygon converted from a 1912 x 792 reference frame.
    DEFAULT_POINTS = [
        (1533 / 1912, 541 / 792),
        (1260 / 1912, 500 / 792),
        (1260 / 1912, 792 / 792),
        (1533 / 1912, 792 / 792),
    ]

    def __init__(self, normalized_points=None):
        self.normalized_points = normalized_points or self.DEFAULT_POINTS

    def get_points(self, image):
        height, width = image.shape[:2]
        return [
            (int(x * width), int(y * height))
            for x, y in self.normalized_points
        ]

    def draw_polygon(self, image):
        points = np.array(self.get_points(image), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            image,
            [points],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )
        return image

    def draw_tracked_box(self, image, bbox, track_id, class_id, score, intruding):
        xmin, ymin, xmax, ymax = bbox
        color = (0, 0, 255) if intruding else (0, 200, 0)
        label = (
            f"ID {track_id} | class {class_id} | {score:.2f}"
            + (" | WARNING" if intruding else "")
        )
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        text_y = max(ymin - 10, 20)
        cv2.putText(
            image,
            label,
            (xmin, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        return image

    def is_inside_polygon(self, image, x, y):
        points = np.array(self.get_points(image), dtype=np.int32)
        return cv2.pointPolygonTest(points, (x, y), False) >= 0
