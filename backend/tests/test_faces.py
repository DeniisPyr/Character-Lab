import sys

import numpy as np
import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline.faces import Face, FaceDetector, decode, encode
from charlab.pipeline.onnx import load_onnx
from charlab.pipeline.stages import StageUnavailable

SETTINGS = load_settings().quality


def output(*candidates):
    """Raw model output, shape (5, N): one column per (cx, cy, w, h, score) in 640x640."""
    return np.array(candidates, dtype=np.float32).T


def test_input_is_640_square_rgb_in_0_to_1():
    data = encode(Image.new("RGBA", (1280, 720), (255, 0, 0, 0)))

    assert data.shape == (1, 3, 640, 640)
    assert data.dtype == np.float32
    assert (data[0, 0].max(), data[0, 1].max()) == (1.0, 0.0)


def test_box_is_scaled_back_to_the_image():
    faces = decode(output((320, 320, 64, 64, 0.9)), (1280, 720), 0.3)

    assert faces == [Face((576, 324, 704, 396), pytest.approx(0.9))]
    assert faces[0].size == 72


def test_candidates_below_the_confidence_are_dropped():
    assert decode(output((320, 320, 64, 64, 0.2)), (640, 640), 0.3) == []


def test_overlapping_boxes_count_once_and_the_best_comes_first():
    raw = output(
        (100, 100, 50, 50, 0.6),  # a second face
        (320, 320, 64, 64, 0.8),
        (322, 321, 64, 64, 0.9),  # the same face as above, more confident
    )

    faces = decode(raw, (640, 640), 0.3)

    assert [face.confidence for face in faces] == pytest.approx([0.9, 0.6])
    assert faces[0].box == (290, 289, 354, 353)


def test_boxes_are_clipped_to_the_image():
    faces = decode(output((10, 630, 40, 40, 0.9)), (640, 640), 0.3)

    assert faces[0].box == (0, 610, 30, 640)


class FakeSession:
    def __init__(self, raw):
        self.raw, self.calls = raw, []

    def run(self, outputs, inputs):
        self.calls.append((outputs, {name: value.shape for name, value in inputs.items()}))
        return [self.raw[None]]


def test_detector_loads_once_and_runs_the_model():
    session, loads = FakeSession(output((320, 320, 64, 64, 0.9))), []
    detector = FaceDetector(SETTINGS, lambda model: loads.append(model) or session)

    detector(Image.new("RGB", (1280, 720)))
    faces = detector(Image.new("RGB", (640, 640)))

    assert loads == [SETTINGS.face_detector]
    assert session.calls[0] == (["output0"], {"images": (1, 3, 640, 640)})
    assert faces[0].box == (288, 288, 352, 352)


def test_detector_uses_the_configured_confidence():
    settings = SETTINGS.model_copy(update={"face_confidence": 0.95})
    detector = FaceDetector(settings, lambda model: FakeSession(output((320, 320, 64, 64, 0.9))))

    assert detector(Image.new("RGB", (640, 640))) == []


def test_missing_onnxruntime_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(StageUnavailable, match=r"need onnxruntime; .* \[gpu\] extra"):
        load_onnx(SETTINGS.face_detector)
