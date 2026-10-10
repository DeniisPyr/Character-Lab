import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from charlab.config import load_settings
from charlab.pipeline import generate
from charlab.pipeline.generate import QwenGenerator, load_pipeline
from charlab.pipeline.stages import StageUnavailable

SETTINGS = load_settings().generation


class FakePipe:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (kwargs["width"], kwargs["height"]))])


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(generate, "_generator", lambda seed: f"generator({seed})")
    return FakePipe()


def make(pipe, settings=SETTINGS):
    loads = []

    def load(settings):
        loads.append(settings)
        return pipe

    return QwenGenerator(settings, load), loads


def test_pipeline_loads_once_on_the_first_call(pipe):
    generator, loads = make(pipe)
    assert loads == []

    generator("A.", [Image.new("RGB", (64, 64))], (832, 1216), 1)
    generator("B.", [Image.new("RGB", (64, 64))], (832, 1216), 2)

    assert loads == [SETTINGS]


def test_call_passes_images_size_steps_and_seed(pipe):
    images = [Image.new("RGB", (64, 64)), Image.new("RGB", (32, 32))]

    result = make(pipe)[0]("Turn around.", images, (832, 1216), 7)

    call = pipe.calls[0]
    assert call["image"] == images
    assert call["prompt"] == "Turn around."
    assert (call["width"], call["height"]) == (832, 1216)
    assert result.size == (832, 1216)
    assert (call["num_inference_steps"], call["true_cfg_scale"]) == (4, 1.0)
    assert call["negative_prompt"] is None
    assert call["generator"] == "generator(7)"


def test_guidance_above_1_sends_a_negative_prompt(pipe):
    settings = SETTINGS.model_copy(update={"cfg": 4.0})

    make(pipe, settings)[0]("A.", [Image.new("RGB", (64, 64))], (1024, 1024), 1)

    assert pipe.calls[0]["negative_prompt"] == " "


def test_missing_gpu_packages_are_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "diffusers", None)

    with pytest.raises(StageUnavailable, match=r"needs torch and diffusers; .* \[gpu\] extra"):
        load_pipeline(SETTINGS)
