import numpy as np
import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline.identity import MEAN, STD, Ccip, encode

SETTINGS = load_settings().quality


def test_input_is_the_whole_image_at_384_with_clip_normalisation():
    data = encode(Image.new("RGBA", (832, 1216), (255, 255, 255, 0)))

    assert data.shape == (3, 384, 384)
    assert data.dtype == np.float32
    assert np.allclose(data[:, 0, 0], ((1 - MEAN) / STD)[:, 0, 0])


class FakeFeatures:
    def __init__(self):
        self.inputs = []

    def run(self, outputs, inputs):
        assert outputs == ["output"]
        self.inputs.append(inputs["input"].shape)
        return [np.full((1, 768), inputs["input"].mean(), dtype=np.float32)]


class FakeMetric:
    def __init__(self):
        self.inputs = []

    def run(self, outputs, inputs):
        a, b = inputs["input"]
        self.inputs.append(inputs["input"].shape)
        difference = abs(float(a[0] - b[0]))
        return [np.array([[0.0, difference], [difference, 0.0]], dtype=np.float32)]


@pytest.fixture
def ccip():
    sessions = {
        SETTINGS.ccip_features.file: FakeFeatures(),
        SETTINGS.ccip_metric.file: FakeMetric(),
    }
    loads = []

    def load(model):
        loads.append(model.file)
        return sessions[model.file]

    return Ccip(SETTINGS, load), sessions, loads


def test_features_run_one_image_through_the_feature_model(ccip):
    model, sessions, _ = ccip

    features = model.features(Image.new("RGB", (64, 64)))

    assert features.shape == (768,)
    assert sessions[SETTINGS.ccip_features.file].inputs == [(1, 3, 384, 384)]


def test_difference_compares_two_features_with_the_metric_model(ccip):
    model, sessions, _ = ccip
    white = model.features(Image.new("RGB", (64, 64), "white"))
    black = model.features(Image.new("RGB", (64, 64), "black"))

    assert model.difference(white, white) == 0
    assert model.difference(white, black) > 0
    assert sessions[SETTINGS.ccip_metric.file].inputs == [(2, 768), (2, 768)]


def test_each_model_loads_once_when_first_needed(ccip):
    model, _, loads = ccip
    assert loads == []

    features = model.features(Image.new("RGB", (64, 64)))
    model.features(Image.new("RGB", (64, 64)))
    model.difference(features, features)
    model.difference(features, features)

    assert loads == [SETTINGS.ccip_features.file, SETTINGS.ccip_metric.file]
