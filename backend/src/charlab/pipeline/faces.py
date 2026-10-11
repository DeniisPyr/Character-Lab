"""Find faces in generated images.

Anime faces: deepghs/anime_face_detection, a YOLOv8 model (MIT), run with onnxruntime. The
pre- and post-processing follow dghs-imgutils 0.19.0 (MIT, imgutils/generic/yolo.py). The
package itself is not a dependency because it requires numpy below 2.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from charlab.config import QualitySettings
from charlab.pipeline.onnx import load_onnx

# The model file has no input size in its metadata, so imgutils' default applies.
INPUT_SIZE = (640, 640)
# Overlapping boxes above this IoU are the same face (imgutils' default).
NMS_IOU = 0.7


@dataclass(frozen=True)
class Face:
    box: tuple[int, int, int, int]  # left, top, right, bottom in image pixels
    confidence: float

    @property
    def size(self) -> int:
        """The shorter side of the box, in pixels."""
        left, top, right, bottom = self.box
        return min(right - left, bottom - top)


def encode(image: Image.Image) -> np.ndarray:
    """The model's input: the image squashed to 640x640, RGB in 0..1, shape (1, 3, H, W)."""
    pixels = np.asarray(image.convert("RGB").resize(INPUT_SIZE), dtype=np.float32) / 255
    return pixels.transpose(2, 0, 1)[None]


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Indices of the boxes to keep, best first, dropping boxes that overlap a better one."""
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0 + 1) * (y1 - y0 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        width = np.maximum(0, np.minimum(x1[best], x1[rest]) - np.maximum(x0[best], x0[rest]) + 1)
        height = np.maximum(0, np.minimum(y1[best], y1[rest]) - np.maximum(y0[best], y0[rest]) + 1)
        overlap = width * height
        iou = overlap / (areas[best] + areas[rest] - overlap)
        order = rest[iou <= NMS_IOU]
    return keep


def decode(output: np.ndarray, image_size: tuple[int, int], confidence: float) -> list[Face]:
    """Faces from the model's raw output, shape (5, N): centre x, centre y, width, height and
    score per candidate, in input pixels. Most confident face first."""
    candidates = output[:, output[4] > confidence].T
    if not len(candidates):
        return []
    cx, cy, w, h, scores = candidates.T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    scale_x, scale_y = image_size[0] / INPUT_SIZE[0], image_size[1] / INPUT_SIZE[1]
    faces = []
    for index in _nms(boxes, scores):
        x0, y0, x1, y1 = boxes[index] * [scale_x, scale_y, scale_x, scale_y]
        box = (
            int(np.clip(x0, 0, image_size[0]).round()),
            int(np.clip(y0, 0, image_size[1]).round()),
            int(np.clip(x1, 0, image_size[0]).round()),
            int(np.clip(y1, 0, image_size[1]).round()),
        )
        faces.append(Face(box, float(scores[index])))
    return faces


# Takes the model file to load and returns an onnxruntime session. Tests pass a fake.
Loader = Callable[[Any], Any]


class FaceDetector:
    """Finds anime faces; the model loads on first use."""

    def __init__(self, settings: QualitySettings, load: Loader = load_onnx):
        self.settings = settings
        self._load = load
        self._session = None

    def __call__(self, image: Image.Image) -> list[Face]:
        if self._session is None:
            self._session = self._load(self.settings.face_detector)
        (output,) = self._session.run(["output0"], {"images": encode(image)})
        return decode(output[0], image.size, self.settings.face_confidence)
