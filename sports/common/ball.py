from collections import deque

import cv2
import numpy as np
import supervision as sv


class BallAnnotator:
    def __init__(self, radius: int, buffer_size: int = 5, thickness: int = 2):
        self.color_palette = sv.ColorPalette.from_matplotlib('jet', buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thickness = thickness

    def interpolate_radius(self, i: int, max_i: int) -> int:
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)

        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame


class BallTracker:
    def __init__(self, buffer_size: int = 10):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections: sv.Detections) -> sv.Detections:
        # 検出が無ければそのまま返す
        if len(detections) == 0:
            return detections

        xy = detections.get_anchors_coordinates(sv.Position.CENTER)

        # 念のための安全チェック
        if xy is None or len(xy) == 0:
            return detections

        # 検出があるときだけ buffer に追加
        self.buffer.append(xy)

        # buffer が空でないことを保証
        if len(self.buffer) == 0:
            return detections

        centroid = np.mean(np.concatenate(list(self.buffer)), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]